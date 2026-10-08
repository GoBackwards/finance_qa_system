"""Build a complete dense index and evaluate the unchanged evidence benchmark."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from .embedding import (DenseEncoder, MODEL_SPECS, build_embeddings,
                        preflight_chunks, sha256_file)
from .faiss_index import build_faiss_index, benchmark_indexes, search_faiss_index
from report_chunker.retrieval import judge_retrieval


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def consolidate_vectors(directory: Path, manifest: dict) -> Path:
    """Validate shard hashes/order before creating a single FAISS-ready matrix."""
    if manifest['status'] != 'complete':
        raise ValueError('Embedding run is incomplete')
    target = directory / 'vectors.npy'
    marker = directory / 'vectors_manifest.json'
    identity = {'embedding_fingerprint': manifest['fingerprint'],
                'shards': [{k: s[k] for k in ('start_index', 'count', 'vectors_sha256')}
                           for s in manifest['shards']],
                'shape': [manifest['vector_count'], manifest['dimension']]}
    cursor = 0
    for shard in manifest['shards']:
        path = (directory / shard['vectors_path']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Embedding shard points outside its directory')
        if (shard['start_index'] != cursor or shard['count'] < 1
                or sha256_file(path) != shard['vectors_sha256']):
            raise ValueError('Embedding shard order or hash mismatch')
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        if data.shape != (shard['count'], manifest['dimension']) or data.dtype != np.float32:
            raise ValueError('Embedding shard shape/dtype mismatch')
        cursor += shard['count']
        del data
    if cursor != manifest['vector_count']:
        raise ValueError('Embedding shard coverage mismatch')
    if marker.exists() and target.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous.get('identity') == identity and previous.get('sha256') == sha256_file(target):
            return target
    temporary = target.with_name(target.name + '.building')
    vectors = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                        shape=tuple(identity['shape']))
    cursor = 0
    try:
        for shard in manifest['shards']:
            path = (directory / shard['vectors_path']).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Embedding shard points outside its directory')
            if shard['start_index'] != cursor or sha256_file(path) != shard['vectors_sha256']:
                raise ValueError('Embedding shard order or hash mismatch')
            data = np.load(path, mmap_mode='r', allow_pickle=False)
            count = shard['count']
            if data.shape != (count, manifest['dimension']) or data.dtype != np.float32:
                raise ValueError('Embedding shard shape/dtype mismatch')
            vectors[cursor:cursor + count] = data
            cursor += count
            del data
        if cursor != manifest['vector_count']:
            raise ValueError('Embedding shard coverage mismatch')
        vectors.flush()
    finally:
        del vectors
    os.replace(temporary, target)
    write_json(marker, {'identity': identity, 'sha256': sha256_file(target),
                        'path': str(target), 'dtype': 'float32'})
    return target


def _aggregate(rows: list[dict], top_k: int) -> dict:
    keys = [f'recall_at_{top_k}', f'hit_at_{top_k}', f'all_evidence_hit_at_{top_k}']
    return {'question_count': len(rows),
            **{key: statistics.mean(row[key] for row in rows) if rows else None for key in keys},
            'gold_units': sum(row['gold_units'] for row in rows),
            'recovered_units': sum(row['recovered_units'] for row in rows)}


def evaluate_dense(index_path: Path, chunks_path: Path, benchmark: dict,
                   query_vectors: np.ndarray, *, top_k: int = 5, threads: int = 4) -> dict:
    """Judge retrieved original payloads, never source labels supplied to search."""
    questions = benchmark['questions']
    if len(query_vectors) != len(questions) or not questions:
        raise ValueError('Queries and benchmark question count differ, or benchmark is empty')
    if len({q['question_id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate benchmark question IDs')
    index_manifest = json.loads(index_path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if index_manifest['chunks_sha256'] != sha256_file(chunks_path):
        raise ValueError('Source chunks no longer match the FAISS vector mapping')
    scores, ids = search_faiss_index(index_path, query_vectors, k=top_k, threads=threads)
    wanted = {int(i) for i in ids.flat if i >= 0}
    selected = {}
    count = 0
    with chunks_path.open('rb') as handle:
        for line in handle:
            if not line.strip():
                raise ValueError('Blank lines are not valid vector mapping entries')
            if count in wanted:
                selected[count] = json.loads(line)
            count += 1
    if count != index_manifest['vector_count'] or set(selected) != wanted:
        raise ValueError('Retrieved vector IDs do not map to the complete source corpus')
    per_query = []
    by_type = defaultdict(list)
    for question, row_scores, row_ids in zip(questions, scores, ids):
        results = [selected[int(i)] for i in row_ids if i >= 0]
        judged = judge_retrieval(question, results, top_k)
        compact = []
        for rank, (score, vector_id) in enumerate(zip(row_scores, row_ids), 1):
            if vector_id < 0:
                continue
            chunk = selected[int(vector_id)]
            citations = {json.dumps(e['citation'], sort_keys=True): e['citation']
                         for e in chunk.get('evidence', []) if e.get('citation')}
            compact.append({'rank': rank, 'vector_id': int(vector_id),
                            'score': float(score), 'chunk_id': chunk['chunk_id'],
                            'document_id': chunk['document_id'], 'kind': chunk['kind'],
                            'token_count': chunk['token_count'],
                            'citations': list(citations.values()), 'text_excerpt': chunk['text'][:1200]})
        result = {'question_id': question['question_id'], 'question': question['question'],
                  'type': question.get('type', 'unspecified'), 'filters': None,
                  **judged, 'retrieved': compact}
        per_query.append(result)
        by_type[result['type']].append(result)
    return {'scope': 'full_corpus_unfiltered', 'top_k': top_k,
            'metric': 'macro canonical source-evidence recall; distinct from ANN neighbor overlap',
            'aggregate': _aggregate(per_query, top_k),
            'by_type': {key: _aggregate(rows, top_k) for key, rows in by_type.items()},
            f'average_top{top_k}_tokens': statistics.mean(sum(c['token_count'] for c in r['retrieved'])
                                                        for r in per_query),
            'per_query': per_query}


def _progress(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=True), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks', type=Path,
                        default=Path('data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', choices=list(MODEL_SPECS), default='BAAI/bge-m3')
    parser.add_argument('--revision', help='Immutable 40-character model commit; defaults to pinned release')
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--shard-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--index-types', nargs='+', choices=['flat', 'ivf', 'hnsw'], default=['flat'])
    parser.add_argument('--nlist', type=int, default=256)
    parser.add_argument('--nprobe', type=int, default=16)
    parser.add_argument('--hnsw-m', type=int, default=32)
    parser.add_argument('--ef-construction', type=int, default=200)
    parser.add_argument('--ef-search', type=int, default=64)
    parser.add_argument('--benchmark', type=Path, default=Path('data/evaluation/chunking_questions.json'))
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--skip-evaluation', action='store_true')
    args = parser.parse_args(argv)
    if args.threads < 1 or args.shard_size < 1:
        parser.error('threads and shard-size must be positive')
    chunks = args.chunks.resolve()
    if not chunks.is_file():
        parser.error(f'Missing chunks file: {chunks}')
    output = (args.output or Path('data/indexes') / args.model.split('/')[-1] / chunks.parent.name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder = DenseEncoder(args.model, args.revision, device, args.batch_size)
    embeddings_dir = output / 'embeddings'
    if args.preflight_only:
        audit = preflight_chunks(chunks, embeddings_dir, encoder, on_progress=_progress)
        print(json.dumps(audit, indent=2), flush=True)
        return 0
    start = time.perf_counter()
    embeddings = build_embeddings(chunks, embeddings_dir, encoder, args.shard_size, on_progress=_progress)
    vectors_path = consolidate_vectors(embeddings_dir, embeddings)
    indexes = {}
    index_manifests = {}
    for kind in dict.fromkeys(['flat', *args.index_types]):
        path = output / f'{kind}.faiss'
        print(f'Building {kind}: {embeddings["vector_count"]:,} vectors, {embeddings["dimension"]} dimensions', flush=True)
        index_manifests[kind] = build_faiss_index(
            vectors_path, path, kind=kind, chunks_path=chunks,
            nlist=args.nlist, nprobe=args.nprobe, hnsw_m=args.hnsw_m,
            ef_construction=args.ef_construction, ef_search=args.ef_search, threads=args.threads)
        indexes[kind] = path
    evaluation_summary = {}
    if not args.skip_evaluation:
        benchmark = json.loads(args.benchmark.read_text(encoding='utf-8'))
        query_start = time.perf_counter()
        query_vectors = encoder.encode_queries([q['question'] for q in benchmark['questions']])
        query_seconds = time.perf_counter() - query_start
        np.save(output / 'query_vectors.npy', query_vectors, allow_pickle=False)
        query_identity = {'benchmark_path': str(args.benchmark.resolve()),
                          'benchmark_sha256': sha256_file(args.benchmark),
                          'encoder': encoder.identity,
                          'question_ids': [q['question_id'] for q in benchmark['questions']],
                          'query_embedding_seconds': query_seconds,
                          'query_vectors_sha256': sha256_file(output / 'query_vectors.npy')}
        write_json(output / 'query_manifest.json', query_identity)
        comparison = benchmark_indexes(indexes, query_vectors, k=5, threads=args.threads)
        write_json(output / 'ann_comparison.json', comparison)
        for kind, index_path in indexes.items():
            result = evaluate_dense(index_path, chunks, benchmark, query_vectors, threads=args.threads)
            result.update(benchmark_sha256=query_identity['benchmark_sha256'], encoder=encoder.identity,
                          index_kind=kind, query_embedding_seconds=query_seconds)
            write_json(output / f'{kind}_retrieval_results.json', result)
            evaluation_summary[kind] = {'aggregate': result['aggregate'], 'by_type': result['by_type'],
                                        'average_top5_tokens': result['average_top5_tokens']}
            print(f'{kind} evidence Recall@5: {result["aggregate"]["recall_at_5"]:.2%}', flush=True)
    summary = {'schema_version': '1.0', 'status': 'complete',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'model': args.model, 'model_revision': encoder.revision,
               'dimension': embeddings['dimension'], 'document_count': embeddings['document_count'],
               'chunk_count': embeddings['chunk_count'], 'vector_count': embeddings['vector_count'],
               'source_chunks_path': str(chunks), 'source_sha256': embeddings['source_sha256'],
               'embedding_manifest': str(embeddings_dir / 'embedding_manifest.json'),
               'indexes': index_manifests, 'evaluation': evaluation_summary,
               'duration_seconds': time.perf_counter() - start,
               'scope': 'Dense embeddings only; starter benchmark is agent-curated exploratory evidence retrieval.'}
    write_json(output / 'index_creation.json', summary)
    print(f'Complete: {output / "index_creation.json"}', flush=True)
    return 0
