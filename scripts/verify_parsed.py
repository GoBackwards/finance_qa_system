"""Audit parsed artifact integrity and provenance, not semantic accuracy.

Reads one document at a time and never changes parsed artifacts or source PDFs.
Only the requested JSON audit report is written. Run after parsing has stopped;
concurrent writers can make the snapshot inconsistent and cause audit failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


ARTIFACTS = ('pages.jsonl', 'tables.jsonl', 'document.md')
HASH_RE = re.compile(r'^[0-9a-f]{64}$')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def bbox_valid(value):
    return (isinstance(value, list) and len(value) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in value)
            and value[0] <= value[2] and value[1] <= value[3])


class Audit:
    def __init__(self):
        self.errors = []
        self.error_count = 0

    def check(self, condition, message):
        if not condition:
            self.error_count += 1
            # Cap details for a damaged file; the total still counts every error.
            if len(self.errors) < 200:
                self.errors.append(message)
        return bool(condition)

    def objects(self, path):
        if not self.check(path.is_file(), f'Missing artifact: {path.name}'):
            return
        with path.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                try:
                    value = json.loads(line)
                except ValueError as exc:
                    self.check(False, f'{path.name}:{number}: invalid JSON: {exc}')
                    continue
                if self.check(isinstance(value, dict), f'{path.name}:{number}: expected object'):
                    yield value

    def citation(self, record, meta, page, label):
        box = record.get('bbox')
        self.check(bbox_valid(box), f'{label}: invalid bbox')
        cite = record.get('citation')
        if not self.check(isinstance(cite, dict), f'{label}: missing citation'):
            return
        for key in ('document_id', 'sha256', 'source_file', 'ticker', 'fiscal_year'):
            self.check(cite.get(key) == meta.get(key), f'{label}: citation {key} mismatch')
        self.check(cite.get('pdf_page') == page, f'{label}: citation page mismatch')
        self.check(cite.get('bbox') == box, f'{label}: citation bbox mismatch')


def table_structure(table, audit, label):
    """Check record shape, including source-text fallbacks; no value accuracy claim."""
    reliable = table.get('structured_reliable')
    if 'structured_reliable' in table:
        audit.check(isinstance(reliable, bool), f'{label}: structured_reliable must be boolean')
    if reliable is False:
        audit.check(table.get('rows') == [], f'{label}: fallback must not expose structured rows')
        audit.check(table.get('cells') == [], f'{label}: fallback must not expose structured cells')
        source_text, markdown = table.get('source_text'), table.get('markdown')
        audit.check(isinstance(source_text, str) and bool(source_text.strip()),
                    f'{label}: fallback must retain original source_text')
        audit.check(isinstance(markdown, str) and isinstance(source_text, str)
                    and bool(source_text) and source_text in markdown and '```' in markdown,
                    f'{label}: fallback markdown must include fenced source_text')
        audit.check(isinstance(table.get('warnings'), list) and bool(table['warnings']),
                    f'{label}: fallback must carry a review warning')
        audit.check(isinstance(table.get('source_lines'), list) and bool(table['source_lines']),
                    f'{label}: fallback must retain source_lines')
        # Candidates are debugging evidence, never accepted structured output.
        # Validate their basic grid shape but deliberately do not judge their values.
        _table_grid(table.get('candidate_rows'), table.get('candidate_cells'), audit,
                    f'{label} debug candidate')
        return
    _table_grid(table.get('rows'), table.get('cells'), audit, label)


def _table_grid(rows, cells, audit, label):
    if not audit.check(isinstance(rows, list) and bool(rows)
                       and all(isinstance(row, list) and row for row in rows),
                       f'{label}: rows must be nonempty arrays'):
        return
    columns = len(rows[0])
    audit.check(all(len(row) == columns for row in rows), f'{label}: ragged table rows')
    audit.check(all(value is None or isinstance(value, str) for row in rows for value in row),
                f'{label}: cell values must be strings or null')
    if not audit.check(isinstance(cells, list), f'{label}: missing cells array'):
        return
    seen = set()
    for index, cell in enumerate(cells):
        loc = f'{label} cell {index}'
        if not audit.check(isinstance(cell, dict), f'{loc}: expected object'):
            continue
        row, column = cell.get('row'), cell.get('column')
        if not audit.check(integer(row) and integer(column) and 0 <= row < len(rows)
                           and 0 <= column < len(rows[row]), f'{loc}: invalid row/column'):
            continue
        audit.check((row, column) not in seen, f'{loc}: duplicate row/column')
        seen.add((row, column))
        audit.check(cell.get('text') == rows[row][column], f'{loc}: text disagrees with rows')
        audit.check(cell.get('bbox') is None or bbox_valid(cell['bbox']), f'{loc}: invalid bbox')
    expected = {(r, c) for r, row in enumerate(rows) for c in range(len(row))}
    audit.check(seen == expected, f'{label}: cells do not cover every row/column position')


def audit_document(path):
    audit = Audit()
    folder = path.parent
    result = {'metadata_file': str(path), 'output_dir': str(folder), 'pages': 0,
              'tables': 0, 'elements': 0, 'native_source_lines': 0,
              'fallback_tables': 0, 'reliable_structured_tables': 0,
              'legacy_tables_without_reliability_flag': 0,
              'active_native_lines': 0, 'accounted_native_lines': 0,
              'warning_histogram': {}, 'table_error_pages': [], 'sparse_pages': []}
    try:
        meta = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(meta, dict):
            raise ValueError('document.json must contain an object')
        result.update(document_id=meta.get('document_id'), source_file=meta.get('source_file'),
                      sha256=meta.get('sha256'), partial=meta.get('partial'),
                      ticker=meta.get('ticker'), fiscal_year=meta.get('fiscal_year'),
                      total_pdf_pages=meta.get('total_pdf_pages'))
        audit.check(meta.get('complete') is True, 'Document completion marker is not true')
        for key in ('document_id', 'source_file', 'coordinate_system'):
            audit.check(isinstance(meta.get(key), str) and bool(meta[key]), f'Missing metadata: {key}')
        for key in ('sha256', 'config_signature', 'source_signature'):
            audit.check(isinstance(meta.get(key), str) and bool(HASH_RE.fullmatch(meta[key])),
                        f'Invalid metadata hash: {key}')
        artifacts = meta.get('artifacts', {})
        if not isinstance(artifacts, dict):
            audit.check(False, 'artifacts must be an object')
            artifacts = {}
        for name in ARTIFACTS:
            artifact = folder / name
            info = artifacts.get(name, {})
            if not isinstance(info, dict):
                info = {}
            if audit.check(artifact.is_file(), f'Missing artifact: {name}'):
                audit.check(digest(artifact) == info.get('sha256'), f'{name}: SHA-256 mismatch')
                audit.check(artifact.stat().st_size == info.get('size_bytes'), f'{name}: size mismatch')

        total, selected = meta.get('total_pdf_pages'), meta.get('selected_pdf_pages')
        valid_selection = (integer(total) and total >= 1 and isinstance(selected, list)
                           and bool(selected) and all(integer(p) and 1 <= p <= total for p in selected))
        if not audit.check(valid_selection, 'Invalid total_pdf_pages/selected_pdf_pages'):
            selected = []
        else:
            audit.check(selected == sorted(set(selected)), 'selected_pdf_pages must be sorted and unique')
            audit.check(meta.get('partial') is (len(selected) != total), 'partial flag disagrees with page selection')
            if meta.get('partial') is False:
                audit.check(selected == list(range(1, total + 1)), 'Full report does not select every PDF page')

        tables = {}
        for table in audit.objects(folder / 'tables.jsonl'):
            result['tables'] += 1
            if table.get('structured_reliable') is False:
                result['fallback_tables'] += 1
            elif table.get('structured_reliable') is True:
                result['reliable_structured_tables'] += 1
            elif 'structured_reliable' not in table:
                result['legacy_tables_without_reliability_flag'] += 1
            tid, pg = table.get('table_id'), table.get('pdf_page')
            label = f'Table {tid!r}'
            audit.citation(table, meta, pg, label)
            audit.check(integer(pg) and pg in selected, f'{label}: page not in selected PDF pages')
            table_structure(table, audit, label)
            if audit.check(isinstance(tid, str) and bool(tid), f'{label}: invalid table_id'):
                audit.check(tid not in tables, f'{label}: duplicate table_id')
                tables[tid] = table

        page_numbers, element_ids, referenced_tables = [], set(), Counter()
        warnings = Counter()
        actual = Counter()
        for page in audit.objects(folder / 'pages.jsonl'):
            result['pages'] += 1
            pg = page.get('page_number')
            label = f'Page {pg!r}'
            page_numbers.append(pg)
            audit.check(integer(pg) and pg in selected, f'{label}: unexpected PDF page')
            audit.check(page.get('document_id') == meta.get('document_id'), f'{label}: document_id mismatch')
            for dimension in ('width', 'height'):
                value = page.get(dimension)
                audit.check(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and value > 0,
                            f'{label}: invalid {dimension}')
            page_warnings = page.get('warnings', [])
            if not isinstance(page_warnings, list) or not all(isinstance(w, str) for w in page_warnings):
                audit.check(False, f'{label}: warnings must be strings')
                page_warnings = []
            warnings.update(w.partition(':')[0] for w in set(page_warnings))
            actual['pages_needing_review'] += bool(page_warnings)
            actual['pages_needing_ocr_or_review'] += any('OCR/review' in w for w in page_warnings)
            actual['pages_table_scanned'] += page.get('table_scan') == 'scanned'
            if page.get('table_scan') == 'error':
                result['table_error_pages'].append(pg)
            if any(w.startswith('low_native_text:') for w in page_warnings):
                result['sparse_pages'].append(pg)

            source = {}
            for line in page.get('source_lines', []):
                sid = line.get('source_line_id')
                if audit.check(isinstance(sid, str) and bool(sid), f'{label}: invalid native source_line_id'):
                    audit.check(sid not in source, f'{label}: duplicate native source_line_id {sid}')
                    source[sid] = line
            expected = {sid for sid, line in source.items()
                        if not line.get('duplicate') and line.get('kind') not in {'header', 'footer'}}
            result['native_source_lines'] += len(source)
            result['active_native_lines'] += len(expected)
            actual['duplicate_lines'] += sum(bool(line.get('duplicate')) for line in source.values())
            actual['removed_margin_lines'] += sum(line.get('kind') in {'header', 'footer'} for line in source.values())
            margin_ids = {line.get('source_line_id') for line in page.get('removed_margins', [])}
            audit.check(margin_ids == {sid for sid, line in source.items() if line.get('kind') in {'header', 'footer'}},
                        f'{label}: removed_margins disagree with native margin labels')
            accounted = set()

            def table_reference(tid, context):
                if audit.check(isinstance(tid, str) and tid in tables, f'{context}: unresolved table_id {tid!r}'):
                    audit.check(tables[tid].get('pdf_page') == pg, f'{context}: table points to another page')

            table_ids = page.get('table_ids', [])
            audit.check(isinstance(table_ids, list) and all(isinstance(t, str) for t in table_ids),
                        f'{label}: invalid table_ids')
            for tid in table_ids:
                table_reference(tid, label)
                if isinstance(tid, str):
                    referenced_tables[tid] += 1
            table_elements = []
            elements = page.get('elements', [])
            for index, element in enumerate(elements, 1):
                result['elements'] += 1
                eid = element.get('element_id')
                loc = f'{label} element {index}'
                audit.check(isinstance(eid, str) and bool(eid) and eid not in element_ids,
                            f'{loc}: missing/duplicate element_id')
                if isinstance(eid, str):
                    element_ids.add(eid)
                audit.citation(element, meta, pg, loc)
                audit.check(element.get('reading_order') == index, f'{loc}: inconsistent reading_order')
                if element.get('kind') == 'table':
                    tid = element.get('table_id')
                    table_reference(tid, loc)
                    table_elements.append(tid)
                    if isinstance(tid, str) and tid in tables:
                        audit.check(element.get('bbox') == tables[tid].get('bbox'), f'{loc}: table bbox mismatch')
                        audit.check(element.get('text', '').strip() == tables[tid].get('markdown', '').strip(),
                                    f'{loc}: table text disagrees with table markdown')
                    continue
                actual['text_blocks'] += 1
                for line in element.get('lines', []):
                    sid = line.get('source_line_id')
                    audit.check(isinstance(sid, str) and sid in expected, f'{loc}: invalid active source line {sid!r}')
                    if isinstance(sid, str):
                        accounted.add(sid)
            for line in page.get('table_source_lines', []):
                sid = line.get('source_line_id')
                audit.check(isinstance(sid, str) and sid in expected, f'{label}: invalid table source line {sid!r}')
                if isinstance(sid, str):
                    accounted.add(sid)
                table_reference(line.get('table_id'), f'{label} table source line {sid}')
            missing = expected - accounted
            audit.check(not missing, f'{label}: omitted active native lines {sorted(missing)[:20]}')
            result['accounted_native_lines'] += len(expected & accounted)
            audit.check(Counter(table_elements) == Counter(table_ids), f'{label}: table elements and table_ids disagree')
            audit.check(page.get('text') == '\n\n'.join(element.get('text', '') for element in elements),
                        f'{label}: page text disagrees with elements')
        audit.check(page_numbers == selected, 'pages.jsonl page sequence/count disagrees with selected_pdf_pages')
        for tid in tables:
            audit.check(referenced_tables[tid] == 1, f'Table {tid}: must be referenced by exactly one page')
        actual.update(pages=result['pages'], tables=result['tables'])
        metadata_counts = meta.get('counts', {})
        for key in ('pages', 'tables', 'text_blocks', 'removed_margin_lines', 'duplicate_lines',
                    'pages_needing_review', 'pages_needing_ocr_or_review', 'pages_table_scanned'):
            audit.check(metadata_counts.get(key) == actual[key], f'Metadata count mismatch: {key}')
        for key in ('fallback_tables', 'reliable_structured_tables'):
            if key in metadata_counts:
                audit.check(metadata_counts[key] == result[key], f'Metadata count mismatch: {key}')
        markdown_path = folder / 'document.md'
        if markdown_path.is_file():
            markdown_pages = [int(m.group(1)) for m in re.finditer(
                r'^## PDF page (\d+)\s*$', markdown_path.read_text(encoding='utf-8'), re.M)]
            audit.check(markdown_pages == selected, 'document.md PDF page headings disagree with selection')
        result['warning_histogram'] = dict(sorted(warnings.items()))
    except Exception as exc:
        audit.check(False, f'Audit could not finish: {type(exc).__name__}: {exc}')
    result.update(valid=audit.error_count == 0, error_count=audit.error_count, errors=audit.errors,
                  errors_truncated=audit.error_count > len(audit.errors))
    return result


def check_source_manifest(path, reports):
    """Require one valid full output per source; partial outputs do not add coverage."""
    path = Path(path).resolve()
    errors = []
    result = {'path': str(path), 'valid': False, 'expected_sources': 0,
              'matched_valid_full_sources': 0, 'missing_sources': [], 'extra_full_sources': [],
              'duplicate_full_sources': [], 'metadata_mismatches': [], 'invalid_full_sources': [],
              'ignored_partial_outputs': sum(r.get('partial') is True for r in reports)}
    expected = {}
    try:
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(rows, list) or not rows:
            raise ValueError('Source manifest must be a nonempty array')
        result['sha256'] = digest(path)
        for index, row in enumerate(rows, 1):
            if (not isinstance(row, dict) or not isinstance(row.get('relative_path'), str)
                    or not row['relative_path'].strip()):
                errors.append(f'Manifest row {index}: missing relative_path')
                continue
            # Accept manifests written with either Windows or POSIX separators.
            name = row['relative_path'].replace('\\', '/').rsplit('/', 1)[-1]
            if not name:
                errors.append(f'Manifest row {index}: relative_path has no basename')
                continue
            if name in expected:
                errors.append(f'Manifest has duplicate basename: {name}')
                continue
            expected[name] = row
            if not isinstance(row.get('sha256'), str) or not HASH_RE.fullmatch(row['sha256']):
                errors.append(f'Manifest {name}: invalid sha256')
            if not isinstance(row.get('ticker'), str) or not row['ticker']:
                errors.append(f'Manifest {name}: missing ticker')
            if not integer(row.get('fiscal_year')):
                errors.append(f'Manifest {name}: invalid fiscal_year')
            if not integer(row.get('page_count')) or row['page_count'] < 1:
                errors.append(f'Manifest {name}: invalid page_count')
    except (ValueError, OSError) as exc:
        errors.append(f'Cannot read source manifest: {type(exc).__name__}: {exc}')
    result['expected_sources'] = len(expected)
    result['expected_tickers'] = sorted({r['ticker'] for r in expected.values()
                                        if isinstance(r.get('ticker'), str)})
    result['expected_fiscal_years'] = sorted({r['fiscal_year'] for r in expected.values()
                                             if integer(r.get('fiscal_year'))})
    full = {}
    for report in reports:
        if report.get('partial') is False:
            name = report.get('source_file')
            if not isinstance(name, str):
                errors.append(f'Full output lacks source_file: {report.get("output_dir")}')
                continue
            full.setdefault(name, []).append(report)
    result['full_outputs'] = sum(map(len, full.values()))
    result['missing_sources'] = sorted(set(expected) - set(full))
    result['extra_full_sources'] = sorted(set(full) - set(expected))
    errors.extend(f'Missing full output: {name}' for name in result['missing_sources'])
    errors.extend(f'Unexpected full source: {name}' for name in result['extra_full_sources'])
    for name, outputs in sorted(full.items()):
        if len(outputs) != 1:
            result['duplicate_full_sources'].append({'source_file': name, 'count': len(outputs),
                                                      'output_dirs': [r.get('output_dir') for r in outputs]})
            errors.append(f'Duplicate full outputs for {name}: {len(outputs)}')
        if name not in expected:
            continue
        row = expected[name]
        matched = len(outputs) == 1
        for report in outputs:
            if report.get('valid') is not True:
                result['invalid_full_sources'].append({'source_file': name, 'output_dir': report.get('output_dir')})
                errors.append(f'Full output failed artifact audit: {name}')
                matched = False
            for output_key, manifest_key in (('sha256', 'sha256'), ('ticker', 'ticker'),
                                              ('fiscal_year', 'fiscal_year'), ('total_pdf_pages', 'page_count')):
                observed, required = report.get(output_key), row.get(manifest_key)
                if observed != required or type(observed) is not type(required):
                    result['metadata_mismatches'].append({'source_file': name, 'output_dir': report.get('output_dir'),
                                                          'field': output_key, 'expected': required, 'observed': observed})
                    errors.append(f'{name}: source manifest {output_key} mismatch')
                    matched = False
        result['matched_valid_full_sources'] += matched
    result.update(valid=not errors, error_count=len(errors), errors=errors)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path, default=Path('data/parsed_validation_report.json'))
    parser.add_argument('--source-manifest', type=Path,
                        help='Require one valid full output per source and verify source metadata')
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f'Parsed root does not exist: {root}')
    # Avoid replacing a parsed artifact even if the caller supplies an unsafe output path.
    if output.suffix.lower() != '.json':
        parser.error('Audit output must be a JSON file')
    if output.name in (*ARTIFACTS, 'document.json', 'parse_manifest.json', 'manifest.json'):
        parser.error('Audit output must not overwrite a parsed artifact or manifest')
    if args.source_manifest and output == args.source_manifest.resolve():
        parser.error('Audit output must not overwrite the source manifest')
    paths = sorted(root.rglob('document.json'))
    reports = []
    histogram = Counter()
    errors = []
    if not paths:
        errors.append('No document.json files found')
    known_folders = {path.parent for path in paths}
    orphan_folders = {path.parent for name in ARTIFACTS for path in root.rglob(name)} - known_folders
    errors.extend(f'Artifacts without document.json: {folder}' for folder in sorted(orphan_folders))
    identities = {}
    for index, path in enumerate(paths, 1):
        report = audit_document(path)
        # Full and partial selections of the same document may coexist. IDs are
        # unique within an artifact set, and the same document ID must retain its source.
        identity = report.get('document_id')
        source = (report.get('source_file'), report.get('sha256'))
        if isinstance(identity, str) and identity in identities and identities[identity] != source:
            errors.append(f'Document ID reused for different sources: {identity}')
        elif isinstance(identity, str):
            identities[identity] = source
        reports.append(report)
        histogram.update(report['warning_histogram'])
        print(f"[{index}/{len(paths)}] {path.parent.name}: {'PASS' if report['valid'] else 'FAIL'}; "
              f"pages={report['pages']} tables={report['tables']} errors={report['error_count']}", flush=True)
    source_manifest_check = None
    if args.source_manifest:
        source_manifest_check = check_source_manifest(args.source_manifest, reports)
        errors.extend(f'Source manifest: {message}' for message in source_manifest_check['errors'])
    failed = sum(not report['valid'] for report in reports)
    summary = {'schema_version': '1.0', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
               'root': str(root), 'scope': 'Artifact integrity, citation consistency and source-line accountability only; not semantic accuracy.',
               'valid': failed == 0 and not errors, 'reports': len(reports), 'reports_failed': failed,
               'full_reports': sum(report.get('partial') is False for report in reports),
               'partial_reports': sum(report.get('partial') is True for report in reports),
               'pages': sum(report['pages'] for report in reports),
               'tables': sum(report['tables'] for report in reports),
               'fallback_tables': sum(report['fallback_tables'] for report in reports),
               'reliable_structured_tables': sum(report['reliable_structured_tables'] for report in reports),
               'legacy_tables_without_reliability_flag': sum(report['legacy_tables_without_reliability_flag'] for report in reports),
               'native_source_lines': sum(report['native_source_lines'] for report in reports),
               'active_native_lines': sum(report['active_native_lines'] for report in reports),
               'accounted_native_lines': sum(report['accounted_native_lines'] for report in reports),
               'warning_histogram': dict(sorted(histogram.items())),
               'table_error_pages': [{'document_id': report.get('document_id'), 'pdf_page': pg}
                                     for report in reports for pg in report['table_error_pages']],
               'sparse_pages': [{'document_id': report.get('document_id'), 'pdf_page': pg}
                                for report in reports for pg in report['sparse_pages']],
               'error_count': len(errors) + sum(report['error_count'] for report in reports),
               'errors': errors, 'documents': reports}
    if source_manifest_check is not None:
        summary['source_manifest_check'] = source_manifest_check
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(output)
    print(f"Audited {len(reports)} report(s), {summary['pages']} pages, {summary['tables']} tables; "
          f"errors={summary['error_count']}. Report: {output}", flush=True)
    return 0 if summary['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
