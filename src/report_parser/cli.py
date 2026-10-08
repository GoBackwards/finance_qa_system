"""Resumable corpus parsing with process isolation for PDF engines."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path
import re
import time

from .pipeline import ParserConfig, parse_document


def parse_page_selection(value):
    pages=set()
    try:
        for item in value.split(','):
            bounds=item.strip().split('-')
            if len(bounds)==1:
                pages.add(int(bounds[0]))
            elif len(bounds)==2:
                start,end=map(int,bounds)
                if start>end or end-start>10000:
                    raise ValueError()
                pages.update(range(start,end+1))
            else:
                raise ValueError()
        if not pages or min(pages)<1:
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError('Use positive 1-based pages, e.g. 1-5,20,50-55')
    return tuple(sorted(pages))


def _worker(job):
    logging.getLogger('pdfminer').setLevel(logging.ERROR)
    path,out,metadata,config,resume=job
    try:
        result=parse_document(path,out,source_metadata=metadata,config=config,resume=resume)
        return {key:result[key] for key in ['document_id','source_file','sha256','output_dir','total_pdf_pages',
                'partial','selected_pdf_pages','complete','counts','duration_seconds','reused','config_signature']}
    except Exception as exc:
        return {'source_file':Path(path).name,'output_dir':str(out),'complete':False,
                'error':f'{type(exc).__name__}: {exc}'}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Extract page-grounded text and tables from annual reports.')
    parser.add_argument('--manifest',type=Path,default=Path('data/manifest.json'))
    parser.add_argument('--input',type=Path,nargs='+',help='Specific PDFs; defaults to the source manifest')
    parser.add_argument('--output',type=Path,default=Path('data/processed'))
    parser.add_argument('--tickers',nargs='+')
    parser.add_argument('--years',type=int,nargs='+')
    parser.add_argument('--pages',type=parse_page_selection,help='1-based PDF pages; partial output gets a separate folder')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--table-mode',choices=['auto','all','off'],default='auto')
    parser.add_argument('--force',action='store_true',help='Reparse even if source and parser fingerprints match')
    args=parser.parse_args(argv)
    if args.workers<1 or (args.limit is not None and args.limit<1):
        parser.error('workers and limit must be positive')
    project_root=Path(__file__).resolve().parents[2]
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig')) if args.manifest.exists() else []
    if args.input:
        # A matching basename is insufficient evidence of provenance for an
        # unrelated external file. Only trust the manifest's actual source path.
        by_path={(project_root/row['relative_path']).resolve():row for row in manifest}
        candidates=[(path,by_path.get(path,{})) for path in dict.fromkeys(p.resolve() for p in args.input)]
    else:
        if not manifest:
            parser.error('No manifest found; specify --manifest or --input')
        candidates=[((project_root/row['relative_path']).resolve(),row) for row in manifest]
    candidates=[(path,meta) for path,meta in candidates if
                (not args.tickers or meta.get('ticker') in args.tickers) and
                (not args.years or meta.get('fiscal_year') in args.years)]
    candidates.sort(key=lambda item:item[0].name)
    if args.limit:
        candidates=candidates[:args.limit]
    if not candidates:
        parser.error('No reports match the requested filters')
    for path,_ in candidates:
        if not path.is_file() or path.suffix.lower()!='.pdf':
            parser.error(f'Not a PDF file: {path}')
    config=ParserConfig(table_mode=args.table_mode,pages=args.pages)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    destinations=set()
    for path,meta in candidates:
        ticker=re.sub(r'[^A-Za-z0-9_-]','_',str(meta.get('ticker','other')))
        stem=re.sub(r'[^A-Za-z0-9_.-]','_',path.stem)
        if args.pages:
            stem+='__pages_'+hashlib.sha256(str(args.pages).encode()).hexdigest()[:8]
        destination=output/ticker/stem
        if destination in destinations:
            parser.error(f'Multiple source PDFs map to {destination}; give the inputs distinct filenames')
        destinations.add(destination)
        jobs.append((path,destination,meta,config,not args.force))
    started=time.monotonic()
    results=[]
    registry_path=output/'parse_manifest.json'
    previous=json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {'documents':[]}
    registry={item['output_dir']:item for item in previous['documents']}
    print(f'Parsing {len(jobs)} report(s); workers={args.workers}; table mode={args.table_mode}',flush=True)
    def record(result):
        results.append(result)
        registry[result['output_dir']]=result
        state='reused' if result.get('reused') else 'parsed' if result['complete'] else 'FAILED'
        counts=result.get('counts',{})
        print(f"[{len(results)}/{len(jobs)}] {result['source_file']} {state}; pages={counts.get('pages',0)} tables={counts.get('tables',0)} review={counts.get('pages_needing_review',0)}" + (f" {result['error']}" if 'error' in result else ''),flush=True)
        content={'schema_version':'1.0','documents':sorted(registry.values(),key=lambda r:r['output_dir'])}
        temp=registry_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(content,indent=2,ensure_ascii=False),encoding='utf-8')
        temp.replace(registry_path)
    if args.workers==1:
        for job in jobs:
            record(_worker(job))
    else:
        # PyMuPDF is not thread-safe. Workers open their own documents.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(_worker,job) for job in jobs]
            for future in as_completed(futures):
                record(future.result())
    failures=sum(not result['complete'] for result in results)
    print(f'Finished in {time.monotonic()-started:.1f}s; completed={len(results)-failures}, failed={failures}',flush=True)
    return 1 if failures else 0
