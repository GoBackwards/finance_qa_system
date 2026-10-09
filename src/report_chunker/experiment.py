"""Run reproducible chunk-size/overlap experiments on the parsed corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import time

from .chunking import ChunkConfig, chunk_document, get_tokenizer
from .retrieval import RETRIEVER, build_index, evaluate
from .validation import validate_document_chunks


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def engine_signature(configs):
    directory=Path(__file__).parent
    payload={'configs':[asdict(c) for c in configs],
             'tokenizer_version':importlib.metadata.version('tiktoken'),
             'code':{name:digest(directory/name) for name in ('chunking.py','validation.py')}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def _document_worker(job):
    folder,part_root,configs,signature,force=job
    folder,part_root=Path(folder),Path(part_root)
    metadata=json.loads((folder/'document.json').read_text(encoding='utf-8'))
    if not metadata.get('complete') or metadata.get('partial'):
        raise ValueError(f'Full completed parse required: {folder}')
    identity=metadata['document_id']
    source={name:digest(folder/name) for name in ('document.json','pages.jsonl','tables.jsonl')}
    for name in ('pages.jsonl','tables.jsonl'):
        if source[name]!=metadata['artifacts'][name]['sha256']:
            raise ValueError(f'Parsed artifact hash mismatch: {folder/name}')
    target=part_root/identity
    if not target.resolve().is_relative_to(part_root.resolve()):
        raise ValueError('Document identifier would escape the experiment folder')
    target.mkdir(parents=True,exist_ok=True)
    marker=target/'complete.json'
    if marker.exists() and not force:
        old=json.loads(marker.read_text(encoding='utf-8'))
        if old.get('signature')==signature and old.get('source')==source:
            if all((target/name).is_file() and digest(target/name)==sha
                   for name,sha in old.get('artifacts',{}).items()) and len(old.get('artifacts',{}))==len(configs):
                return {**old,'reused':True}
    pages=list(read_jsonl(folder/'pages.jsonl'))
    tables=list(read_jsonl(folder/'tables.jsonl'))
    tokenizer=get_tokenizer(configs[0].tokenizer_name)
    strategies={}
    artifacts={}
    for config in configs:
        chunks=list(chunk_document(metadata,pages,tables,config,tokenizer))
        validation=validate_document_chunks(metadata,pages,tables,chunks)
        path=target/(config.strategy_id+'.jsonl')
        temporary=path.with_suffix('.jsonl.tmp')
        with temporary.open('w',encoding='utf-8',newline='\n') as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk,ensure_ascii=False,separators=(',',':'))+'\n')
        temporary.replace(path)
        artifacts[path.name]=digest(path)
        strategies[config.strategy_id]={'file':str(path),'validation':validation,'chunks':len(chunks)}
    result={'document_id':identity,'source_file':metadata['source_file'],
            'ticker':metadata.get('ticker'),'fiscal_year':metadata.get('fiscal_year'),
            'source':source,'signature':signature,'artifacts':artifacts,
            'strategies':strategies,'reused':False}
    atomic_json(marker,result)
    return result


def _percentile(values, fraction):
    if not values:
        return 0
    ordered=sorted(values)
    index=(len(ordered)-1)*fraction
    low=int(index)
    high=min(len(ordered)-1,low+1)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)


def merge_strategy(results,config,output):
    """Merge immutable document parts in deterministic document order."""
    directory=output/config.strategy_id
    directory.mkdir(parents=True,exist_ok=True)
    chunks_path=directory/'chunks.jsonl'
    temporary=directory/'chunks.jsonl.tmp'
    lengths=[]
    characters=0
    kinds=Counter()
    kind_tokens=Counter()
    oversized=Counter()
    overlap_sum=0
    header_tokens=0
    validations=[]
    with temporary.open('wb') as destination:
        for report in sorted(results,key=lambda r:r['document_id']):
            part=report['strategies'][config.strategy_id]
            validations.append(part['validation'])
            with Path(part['file']).open('rb') as source:
                for raw in source:
                    destination.write(raw)
                    chunk=json.loads(raw)
                    count=chunk['token_count']
                    lengths.append(count)
                    characters+=chunk['character_count']
                    kinds[chunk['kind']]+=1
                    kind_tokens[chunk['kind']]+=count
                    if chunk['oversized']:
                        oversized[chunk.get('oversized_reason','unspecified')]+=1
                    if chunk['kind']=='prose':
                        overlap_sum+=chunk.get('overlap_tokens',0)
                    header_tokens+=chunk.get('repeated_header_tokens',0)
    temporary.replace(chunks_path)
    n=len(lengths)
    stats={'strategy_id':config.strategy_id,'chunk_size':config.chunk_size,'overlap':config.overlap,
           'tokenizer':config.tokenizer_name,'document_count':len(results),'chunk_count':n,
           'total_tokens':sum(lengths),'average_tokens':sum(lengths)/max(1,n),
           'average_characters':characters/max(1,n),'median_tokens':statistics.median(lengths) if lengths else 0,
           'p95_tokens':_percentile(lengths,.95),'max_tokens':max(lengths,default=0),
           'min_tokens':min(lengths,default=0),'oversized_chunks':sum(oversized.values()),
           'oversized_reasons':dict(oversized),'chunk_kinds':dict(kinds),
           'average_tokens_by_kind':{kind:kind_tokens[kind]/count for kind,count in kinds.items()},
           'average_prose_overlap_tokens':overlap_sum/max(1,kinds['prose']),
           'repeated_table_header_tokens':header_tokens,
           'row_split_violations':0,'source_coverage_validation':'passed',
           'validated_documents':len(validations),'chunks_sha256':digest(chunks_path),
           'chunks_size_bytes':chunks_path.stat().st_size,
           'document_validations':validations}
    atomic_json(directory/'statistics.json',stats)
    return stats


def bootstrap_mean_interval(values, *, seed=20261008, iterations=2000):
    if not values:
        return [None,None]
    generator=random.Random(seed)
    means=[sum(generator.choices(values,k=len(values)))/len(values) for _ in range(iterations)]
    return [_percentile(means,.025),_percentile(means,.975)]


def validate_benchmark(benchmark,reports):
    """Validate labels against original parsed sources, independently of chunks."""
    folders={json.loads((p/'document.json').read_text(encoding='utf-8'))['document_id']:p for p in reports}
    questions=benchmark['questions']
    if not questions:
        raise ValueError('Benchmark must contain questions')
    ids=[q['question_id'] for q in questions]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate benchmark question IDs')
    needed={e['document_id'] for q in questions for unit in q['evidence']
            for e in [unit,*unit.get('alternatives',[])]}
    data={}
    for docid in needed:
        if docid not in folders:
            raise ValueError(f'Benchmark source document is missing: {docid}')
        folder=folders[docid]
        meta=json.loads((folder/'document.json').read_text(encoding='utf-8'))
        elements={e['element_id']:e for p in read_jsonl(folder/'pages.jsonl') for e in p['elements']}
        tables={t['table_id']:t for t in read_jsonl(folder/'tables.jsonl')}
        data[docid]=(meta,elements,tables)
    units=0
    for question in questions:
        if not question.get('question','').strip() or not question.get('evidence'):
            raise ValueError('Each question needs text and evidence')
        units+=len(question['evidence'])
        for unit in question['evidence']:
            for gold in [unit,*unit.get('alternatives',[])]:
                if gold.get('kind') == 'table_source':
                    from report_chunker.retrieval import _validate_gold
                    _validate_gold(gold)
                meta,elements,tables=data[gold['document_id']]
                if gold.get('source_sha256',gold.get('sha256'))!=meta['sha256']:
                    raise ValueError(f"Benchmark source hash mismatch: {question['question_id']}")
                if gold.get('table_id'):
                    source=tables[gold['table_id']]
                    if gold.get('kind') == 'table_source':
                        if source.get('structured_reliable') is not False:
                            raise ValueError('Benchmark table source requires an unstructured fallback')
                        source_text=source.get('source_text')
                        if (not isinstance(source_text,str) or gold['char_end'] > len(source_text)
                                or source_text[gold['char_start']:gold['char_end']] != gold['quote']):
                            raise ValueError('Benchmark table source quote/range changed')
                    else:
                        if source.get('structured_reliable') is not True:
                            raise ValueError('Benchmark relies on unverified structured cells')
                        row=source['rows'][gold['row_index']]
                        if gold.get('row_cells',row)!=row or gold['row_text']!=' | '.join('' if v is None else str(v) for v in row):
                            raise ValueError('Benchmark table row changed')
                    citation=source['citation']
                else:
                    source=elements[gold['element_id']]
                    if source['text'][gold['char_start']:gold['char_end']]!=gold['quote']:
                        raise ValueError('Benchmark quote/range changed')
                    citation=source['citation']
                if citation['pdf_page']!=gold['pdf_page']:
                    raise ValueError('Benchmark PDF page changed')
    return {'valid':True,'questions':len(questions),'required_evidence_units':units,
            'source_documents':len(needed),'judgment':'canonical source spans/rows; not answer accuracy'}


def _query_values(evaluation):
    return {row['question_id']:row['recall_at_5'] for row in evaluation['per_query']}


def _top5_tokens(evaluation):
    return statistics.mean(sum(chunk.get('token_count',0) for chunk in row.get('results',row.get('retrieved',[])))
                           for row in evaluation['per_query'])


def comparison_report(output,summary):
    baseline=summary['baseline_strategy_id']
    lines=['# Chunking experiment comparison','',
           f"Corpus: {summary['document_count']} reports. Sizes and overlaps are **tokens**, measured with `{summary['tokenizer']}`.",
           f"Retrieval: SQLite FTS5 BM25, same {summary['question_count']}-question starter benchmark and top-k=5 in every group.",'',
           '| Size | Prose overlap | Chunks | Avg tokens | Avg characters | Oversized | Recall@5 | Delta (pp) | Avg top-5 tokens |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['strategies']:
        lines.append(f"| {row['chunk_size']:,} | {row['overlap']} | {row['chunk_count']:,} | {row['average_tokens']:.1f} | {row['average_characters']:.1f} | {row['oversized_chunks']:,} | {row['recall_at_5']:.2%} | {row['delta_recall_pp']:+.2f} | {row['average_top5_tokens']:.1f} |")
    lines.extend(['',f"Delta is relative to `{baseline}`. All groups use the same table-preserving strategy; every document passes source-character and complete-row coverage validation.",'',
                  '## Interpretation','',
                  '- This is an agent-curated exploratory benchmark, not held-out production evaluation. BM25 scores do not predict a future embedding or hybrid retriever.',
                  '- Recall@5 counts unique required source evidence units recovered across the five chunks, then averages per question. Repeated overlap or headers cannot inflate recovered evidence counts. It is distinct from Hit@5 for questions needing multiple units.',
                  '- Gold relevance is tied to source spans and table rows, not strategy-dependent chunk IDs. Unjudged equivalent passages may be valid answers yet receive no credit.',
                  '- Larger chunks supply more tokens at the same top-k; average top-5 tokens reports that cost. Compare at a fixed context-token budget in the next retrieval experiment too.',
                  '- Prose can cross page boundaries while preserving per-fragment citations. Overlap is a measured maximum for prose; tables repeat source headers/context without duplicating body rows.',
                  '- Chunk size is a soft limit for an indivisible table row, unverified whole table, or page whose table extraction failed. Oversized counts are explicit; no source text is silently truncated.',
                  '- Row preservation covers recognized tables and known failed-table pages. Tables missed entirely by PDF layout detection cannot be guaranteed by the chunker.',
                  '', 'The JSON comparison includes narrative/table subset metrics, 95% paired bootstrap intervals, a separately labeled company/year-filtered diagnostic, source/benchmark hashes, and per-strategy artifact paths. Each strategy folder contains the reusable `chunks.jsonl`, statistics, and per-question retrieval results.'])
    path=output/'comparison.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('data/processed'))
    parser.add_argument('--output',type=Path,default=Path('data/experiments/chunking'))
    parser.add_argument('--benchmark',type=Path,default=Path('data/evaluation/chunking_questions.json'))
    parser.add_argument('--sizes',type=int,nargs='+',default=[512,1024,2048])
    parser.add_argument('--overlaps',type=int,nargs='+',default=[50,100,200])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--keep-indexes',action='store_true',help='Keep all SQLite indexes after evaluation')
    parser.add_argument('--force',action='store_true',help='Rebuild cached document chunk parts')
    args=parser.parse_args(argv)
    if args.workers<1:
        parser.error('workers must be positive')
    try:
        configs=[ChunkConfig(size,overlap) for size in sorted(set(args.sizes)) for overlap in sorted(set(args.overlaps))]
    except ValueError as exc:
        parser.error(str(exc))
    folders=sorted(p.parent for p in args.input.glob('*/*/document.json'))
    if not folders:
        parser.error('No parsed document folders found')
    benchmark=json.loads(args.benchmark.read_text(encoding='utf-8'))
    benchmark_validation=validate_benchmark(benchmark,folders)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    signature=engine_signature(configs)
    parts=output/'_document_parts'
    jobs=[(folder.resolve(),parts,configs,signature,args.force) for folder in folders]
    results=[]
    started=time.monotonic()
    print(f"Chunking {len(jobs)} documents across {len(configs)} strategies; benchmark validated ({benchmark_validation['questions']} questions)",flush=True)
    if args.workers==1:
        for job in jobs:
            result=_document_worker(job)
            results.append(result)
            print(f"[{len(results)}/{len(jobs)}] {result['source_file']}: all strategies validated",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(_document_worker,job):job[0] for job in jobs}
            for future in as_completed(futures):
                result=future.result()
                results.append(result)
                print(f"[{len(results)}/{len(jobs)}] {result['source_file']}: all strategies validated",flush=True)
    atomic_json(output/'chunking_manifest.json',{'signature':signature,'documents':results})
    comparisons=[]
    evaluations={}
    benchmark_sha=digest(args.benchmark)
    for index,config in enumerate(configs,1):
        print(f"[{index}/{len(configs)} strategies] Merge/index/evaluate {config.strategy_id}",flush=True)
        stats=merge_strategy(results,config,output)
        directory=output/config.strategy_id
        index_path=directory/'retrieval.sqlite'
        index_info=build_index(directory/'chunks.jsonl',index_path)
        evaluation=evaluate(index_path,benchmark['questions'])
        filtered=evaluate(index_path,benchmark['questions'],filtered=True)
        evaluation.update(benchmark_sha256=benchmark_sha,benchmark_validation=benchmark_validation)
        filtered.update(benchmark_sha256=benchmark_sha,diagnostic_only=True)
        atomic_json(directory/'retrieval_results.json',evaluation)
        atomic_json(directory/'retrieval_filtered_diagnostic.json',filtered)
        atomic_json(directory/'index_statistics.json',index_info)
        values=_query_values(evaluation)
        row={k:v for k,v in stats.items() if k!='document_validations'}
        row.update(**evaluation['aggregate'],by_type=evaluation['by_type'],
                   recall_95ci=bootstrap_mean_interval(list(values.values())),
                   filtered_recall_at_5=filtered['aggregate']['recall_at_5'],
                   average_top5_tokens=_top5_tokens(evaluation),
                   artifacts={'chunks':str(directory/'chunks.jsonl'),
                              'retrieval':str(directory/'retrieval_results.json'),
                              'statistics':str(directory/'statistics.json')})
        comparisons.append(row)
        evaluations[config.strategy_id]=values
        if not args.keep_indexes:
            # This exact temporary index belongs to this run and is reproducible
            # from chunks.jsonl; never delete a computed path outside output.
            if not index_path.resolve().is_relative_to(output):
                raise ValueError('Unsafe index cleanup path')
            index_path.unlink()
        print(f"  chunks={stats['chunk_count']:,}; avg={stats['average_tokens']:.1f} tokens; Recall@5={row['recall_at_5']:.2%}",flush=True)
    baseline=configs[0].strategy_id
    for row in comparisons:
        paired=[evaluations[row['strategy_id']][qid]-value for qid,value in evaluations[baseline].items()]
        row['delta_recall_pp']=statistics.mean(paired)*100
        row['delta_95ci_pp']=[v*100 for v in bootstrap_mean_interval(paired)]
    summary={'schema_version':'1.0','created_at_utc':datetime.now(timezone.utc).isoformat(),
             'document_count':len(results),'question_count':len(benchmark['questions']),
             'tokenizer':configs[0].tokenizer_name,'tokenizer_version':importlib.metadata.version('tiktoken'),
             'engine_signature':signature,'retriever':RETRIEVER,'baseline_strategy_id':baseline,
             'benchmark':str(args.benchmark.resolve()),'benchmark_sha256':benchmark_sha,
             'benchmark_validation':benchmark_validation,'strategies':comparisons,
             'duration_seconds':round(time.monotonic()-started,3),
             'bootstrap':{'iterations':2000,'seed':20261008,'unit':'question','paired_for_deltas':True},
             'scope':'Exploratory source-evidence retrieval; no answer-generation or embedding evaluation.'}
    atomic_json(output/'comparison.json',summary)
    comparison_report(output,summary)
    print(f"Completed all {len(configs)} strategies in {summary['duration_seconds']}s. Report: {output/'comparison.md'}",flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
