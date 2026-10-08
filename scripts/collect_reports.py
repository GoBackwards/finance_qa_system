"""Download source-mapped annual reports without modifying the original PDF bytes.

Usage: python scripts/collect_reports.py [--sources data/sources_*.json] [--workers 4]
Requires Poppler pdfinfo and the pypdf and pypdfium2 Python packages.
Existing files are validated and reused; failures remain visible in the manifest.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from pypdf import PdfReader
import pypdfium2 as pdfium
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AnnualReportResearch/1.0)", "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8"}
PDF_LOCK = threading.Lock()  # PDFium is not thread-safe.
ATTEMPTS = 2


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30)


def inspect(path, ticker, year):
    with path.open('rb') as f:
        if not f.read(1024).lstrip().startswith(b'%PDF-'):
            raise ValueError('Downloaded content is not a PDF')
    info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
    if info.returncode:
        raise ValueError('PDF parser failed: ' + info.stderr[:300])
    match = re.search(r'^Pages:\s+(\d+)', info.stdout, re.M)
    pages = int(match.group(1)) if match else 0
    if pages < 25:
        raise ValueError(f'Only {pages} pages; inspect whether this is a complete annual report')
    reader = PdfReader(path)
    if len(reader.pages) != pages:
        raise ValueError('PDF parsers disagree on page count')
    with PDF_LOCK:
        with pdfium.PdfDocument(str(path)) as doc:
            texts = []
            for page in doc:
                textpage = page.get_textpage()
                try:
                    texts.append(textpage.get_text_range())
                finally:
                    textpage.close()
                    page.close()
    txt = '\f'.join(texts)
    if len(txt.strip()) < 10000:
        raise ValueError('Insufficient extractable text; OCR or source review required')
    snippets = re.findall(r'(?:for\s+the\s+)?(?:fiscal\s+)?year\s+ended\s+[A-Za-z]+\s+\d{1,2},?\s+20\d{2}', txt, re.I)
    unique = list(dict.fromkeys(re.sub(r'\s+', ' ', s).strip() for s in snippets))[:12]
    first = re.sub(r'\s+', ' ', txt[:2500]).strip()
    audit = ROOT / 'tmp' / 'report_audit'
    audit.mkdir(parents=True, exist_ok=True)
    # Retain opening text and fiscal-period evidence for inspection, not a second corpus.
    (audit / f'{ticker}_{year}.txt').write_text(txt[:16000] + '\n\nFISCAL PERIOD MENTIONS\n' + '\n'.join(unique), encoding='utf-8')
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'page_count': pages, 'size_bytes': path.stat().st_size, 'sha256': sha,
            'text_characters': len(txt), 'pages_with_text': sum(bool(p.strip()) for p in txt.split('\f')),
            'fiscal_period_mentions': unique, 'opening_text': first,
            'text_extraction': 'passed', 'parser_warnings': info.stderr[:1000]}


def collect(row, replace_existing=False, previous=None):
    result = dict(row)
    ticker, year = row['ticker'], int(row['fiscal_year'])
    if not re.fullmatch(r'[A-Z]{1,5}', ticker) or year not in range(2021, 2026):
        raise ValueError('Unexpected ticker or fiscal year')
    path = ROOT / 'data' / 'raw' / ticker / f'{ticker}_FY{year}_annual_report.pdf'
    path.parent.mkdir(parents=True, exist_ok=True)
    result['relative_path'] = path.relative_to(ROOT).as_posix()
    result['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    try:
        if path.exists() and not replace_existing:
            if previous:
                # Retain provenance for the actual bytes, even if a source map changed.
                for key in ['url', 'source_page', 'download_url', 'resolved_url', 'content_type', 'downloaded_at_utc']:
                    if key in previous:
                        result[key] = previous[key]
            result.update(inspect(path, ticker, year))
            result['status'] = 'validated'
            result['reused_existing'] = True
            return result
        errors = []
        for url in [row['url']] + row.get('fallback_urls', []):
            for attempt in range(ATTEMPTS):
                part = path.with_suffix('.pdf.part')
                try:
                    with request(url) as response, part.open('wb') as output:
                        final_url = response.url
                        content_type = response.headers.get('Content-Type', '')
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    result.update(inspect(part, ticker, year))
                    part.replace(path)
                    result.update(status='validated', download_url=url, resolved_url=final_url,
                                  content_type=content_type, downloaded_at_utc=datetime.now(timezone.utc).isoformat())
                    return result
                except Exception as exc:
                    errors.append(f'{url}: {type(exc).__name__}: {exc}')
                    if part.exists():
                        part.unlink()
                    if attempt + 1 < ATTEMPTS:
                        time.sleep(1)
        raise RuntimeError(' | '.join(errors))
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    return result


def main():
    global ATTEMPTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='*')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tickers', nargs='*')
    parser.add_argument('--only-missing', action='store_true')
    parser.add_argument('--attempts', type=int, default=2)
    parser.add_argument('--replace', nargs='*', default=[], metavar='TICKER:YEAR',
                        help='Download a replacement, retaining the existing PDF until the new one validates')
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1:
        parser.error('workers and attempts must be positive')
    replacement_keys = set()
    for key in args.replace:
        if not re.fullmatch(r'[A-Z]{1,5}:202[1-5]', key):
            parser.error(f'Invalid replacement key: {key}')
        ticker, year = key.split(':')
        replacement_keys.add((ticker, int(year)))
    ATTEMPTS = args.attempts
    source_paths = [Path(p) for p in args.sources] if args.sources else [ROOT / 'data' / f'sources_{group}.json' for group in ['semiconductors','software','consumer','industrial']]
    records = {}
    for source in source_paths:
        raw = json.loads(source.read_text(encoding='utf-8-sig'))
        if isinstance(raw, dict):
            raw = raw.get('reports', raw.get('sources', []))
        for row in raw:
            if not args.tickers or row['ticker'] in args.tickers:
                records[(row['ticker'], int(row['fiscal_year']))] = row
    manifest_path = ROOT / 'data' / 'manifest.json'
    prior = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    merged = {(r['ticker'], int(r['fiscal_year'])): r for r in prior}
    if replacement_keys - records.keys():
        parser.error('Every replacement must exist in the selected source maps')
    if args.only_missing:
        records = {key: row for key, row in records.items() if key in replacement_keys or key not in merged or merged[key].get('status') != 'validated' or not (ROOT / merged[key]['relative_path']).exists()}
    if not records:
        print('No missing reports in the selected source maps. Existing files and manifests retained.', flush=True)
        return
    print(f'Collecting {len(records)} reports with {args.workers} workers', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect, row, key in replacement_keys, merged.get(key)) for key, row in records.items()]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            merged[(row['ticker'], int(row['fiscal_year']))] = row
            print(f"[{i}/{len(records)}] {row['ticker']} FY{row['fiscal_year']} {row['status']} " + (str(row.get('page_count', '')) + ' pages' if row['status'] == 'validated' else row.get('error', '')[:240]), flush=True)
            temp_manifest = manifest_path.with_suffix('.json.tmp')
            temp_manifest.write_text(json.dumps(sorted(merged.values(), key=lambda x: (x['ticker'], x['fiscal_year'])), indent=2, ensure_ascii=False), encoding='utf-8')
            temp_manifest.replace(manifest_path)
    fields = ['ticker','company','sector','fiscal_year','relative_path','status','source_page','url','download_url','resolved_url','page_count','size_bytes','sha256','text_characters','pages_with_text','fiscal_period_mentions','checked_at_utc','error']
    with (ROOT / 'data' / 'manifest.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=lambda x: (x['ticker'], x['fiscal_year'])))
    failures = [r for r in merged.values() if r['status'] != 'validated']
    print(f'TOTAL: {len(merged) - len(failures)} validated, {len(failures)} failed', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
