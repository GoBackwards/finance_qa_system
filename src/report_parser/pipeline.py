"""Native PDF text + geometric layout + separate table structure extraction.

No neural layout inference or OCR is performed. Every source line is retained for
audit, including lines suppressed from the cleaned reading stream.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf

from . import __version__
from .layout import annotate_margins, lines_to_blocks, order_lines
from .tables import extract_tables


@dataclass(frozen=True)
class ParserConfig:
    table_mode: str = "auto"
    pages: tuple[int, ...] | None = None
    min_text_characters: int = 40

    def __post_init__(self):
        if self.table_mode not in {"auto", "all", "off"}:
            raise ValueError('table_mode must be auto, all or off')
        if self.pages is not None and (not self.pages or any(p < 1 for p in self.pages)):
            raise ValueError('pages must contain positive 1-based PDF page numbers')


def _rounded(bbox):
    return [round(float(value), 3) for value in bbox]


def _hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _versions():
    return {'parser': __version__, 'pymupdf': pymupdf.VersionBind,
            'pdfplumber': importlib.metadata.version('pdfplumber')}


def config_signature(config: ParserConfig) -> str:
    # Changes to implementation invalidate cached output even before a release bump.
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    payload = {'config': asdict(config), 'versions': _versions(), 'code_sha256': code.hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _extract_lines(page) -> list[dict[str, Any]]:
    result = []
    # Text-only flags avoid embedding image bytes in memory/output.
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    data = page.get_text('dict', flags=flags, sort=False)
    for block_index, block in enumerate(data.get('blocks', [])):
        if block.get('type') != 0:
            continue
        for line_index, line in enumerate(block.get('lines', [])):
            spans = line.get('spans', [])
            text = ''.join(span.get('text', '') for span in spans).strip()
            if not text:
                continue
            bbox = pymupdf.Rect(line['bbox']) * page.rotation_matrix
            sizes = [span['size'] for span in spans if span.get('text', '').strip()]
            dx,dy = line.get('dir',(1,0))
            matrix = page.rotation_matrix
            direction = [round(dx*matrix.a+dy*matrix.c,6),round(dx*matrix.b+dy*matrix.d,6)]
            result.append({'text': text, 'bbox': _rounded(bbox),
                           'font_size': round(statistics.median(sizes), 3) if sizes else 0,
                           'bold': any(span.get('flags', 0) & 16 for span in spans),
                           'block_id': block_index, 'line_id': line_index,
                           'source_line_id': f'{block_index}:{line_index}',
                           'direction': direction})
    # Some PDFs draw the same text twice to simulate bold. Keep the first and
    # explicitly retain duplicates, instead of silently destroying evidence.
    seen = set()
    for line in result:
        key = (line['text'], tuple(round(x, 1) for x in line['bbox']))
        line['duplicate'] = key in seen
        seen.add(key)
    return result


def _text_quality(lines: list[dict], minimum: int):
    text = '\n'.join(line['text'] for line in lines)
    nonspace = sum(not char.isspace() for char in text)
    broken = sum(char in '\ufffd\ufffe\uffff' or (ord(char) < 32 and char not in '\r\n\t') for char in text)
    words = re.findall(r'[A-Za-z]{2,}', text.lower())
    common = {'the','and','of','to','in','for','we','our','is','as','that','with','was','are','by','on','year','company'}
    common_ratio = sum(word in common for word in words) / max(1, len(words))
    warnings = []
    if nonspace < minimum:
        warnings.append('low_native_text: scanned, image-only or intentionally sparse page; OCR/review may be needed')
    if broken / max(1, len(text)) > .01:
        warnings.append('suspect_text_encoding: invalid/control characters; OCR/review may be needed')
    elif len(words) > 150 and common_ratio < .025:
        warnings.append('unusual_text_distribution: inspect tables, letter spacing or font encoding')
    return {'nonspace_characters': nonspace, 'broken_character_ratio': round(broken/max(1,len(text)),5),
            'common_word_ratio': round(common_ratio,5), 'warnings': warnings}


def _table_candidate(lines):
    # Cheap native-text gate; qualitative or exceptionally sparse tables can be
    # missed. --table-mode all bypasses this performance tradeoff explicitly.
    numeric = sum(bool(re.search(r'\d', line['text'])) for line in lines)
    return numeric >= 6


def _covered(bbox, region):
    x0,y0,x1,y1 = bbox
    a0,b0,a1,b1 = region
    area = max(0,x1-x0)*max(0,y1-y0)
    intersection = max(0,min(x1,a1)-max(x0,a0))*max(0,min(y1,b1)-max(y0,b0))
    return intersection/max(area, .001)


def _citation(meta, page_number, bbox):
    return {'document_id':meta['document_id'], 'source_file':meta['source_file'],
            'sha256':meta['sha256'], 'ticker':meta.get('ticker'), 'fiscal_year':meta.get('fiscal_year'),
            'pdf_page':page_number, 'bbox':_rounded(bbox)}


def parse_document(pdf_path: str | Path, output_dir: str | Path, *,
                   source_metadata: dict | None = None, config: ParserConfig | None = None,
                   resume: bool = True) -> dict:
    started = time.monotonic()
    pdf_path, output_dir = Path(pdf_path).resolve(), Path(output_dir).resolve()
    config = config or ParserConfig()
    source_metadata = source_metadata or {}
    signature, sha = config_signature(config), _hash(pdf_path)
    provenance = {key:source_metadata[key] for key in
                  ['ticker','company','sector','fiscal_year','download_url','resolved_url','source_page']
                  if key in source_metadata}
    source_signature = hashlib.sha256(json.dumps(
        {'path':str(pdf_path), 'metadata':provenance},sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    document_path = output_dir/'document.json'
    if resume and document_path.exists():
        try:
            old = json.loads(document_path.read_text(encoding='utf-8'))
            if (old.get('sha256') == sha and old.get('config_signature') == signature
                and old.get('source_signature') == source_signature and old.get('complete')
                and all((output_dir/p).is_file() and _hash(output_dir/p) == old.get('artifacts',{}).get(p,{}).get('sha256')
                        for p in ['pages.jsonl','tables.jsonl','document.md'])):
                return {**old, 'reused': True, 'output_dir': str(output_dir)}
        except (ValueError, OSError):
            pass
    meta = dict(provenance)
    meta.update(schema_version='1.0', parser_version=__version__, versions=_versions(),
                document_id=f'{pdf_path.stem}-{sha[:12]}', source_file=pdf_path.name,
                source_path=str(pdf_path), sha256=sha, config=asdict(config), config_signature=signature,
                source_signature=source_signature,
                coordinate_system='PDF points, origin top-left of displayed page; 1-based PDF page numbers',
                parsed_at_utc=datetime.now(timezone.utc).isoformat(), complete=False)
    outputs = {name:(output_dir/(name+'.tmp')).open('w',encoding='utf-8',newline='\n')
               for name in ['pages.jsonl','tables.jsonl','document.md']}
    counts = {'pages':0,'tables':0,'text_blocks':0,'removed_margin_lines':0,'duplicate_lines':0,
              'pages_needing_review':0,'pages_needing_ocr_or_review':0,'pages_table_scanned':0,
              'fallback_tables':0,'reliable_structured_tables':0}
    review_pages = []
    plumber = None
    try:
        with pymupdf.open(pdf_path) as doc:
            if doc.needs_pass:
                raise ValueError('Password-protected PDF cannot be parsed without a password')
            selected = list(config.pages) if config.pages is not None else list(range(1,len(doc)+1))
            if any(p > len(doc) for p in selected):
                raise ValueError(f'Requested PDF page exceeds document length {len(doc)}')
            selected = sorted(set(selected))
            meta.update(total_pdf_pages=len(doc), selected_pdf_pages=selected, partial=len(selected)!=len(doc))
            # Analyze every page for robust recurring margins, including subset runs.
            pages = []
            for index in range(len(doc)):
                page = doc[index]
                pages.append({'page_number':index+1,'width':float(page.rect.width),
                              'height':float(page.rect.height),'rotation':page.rotation,
                              'pdf_page_label':page.get_label(), 'lines':_extract_lines(page)})
            annotate_margins(pages)
            if config.table_mode != 'off':
                plumber = pdfplumber.open(pdf_path)
            outputs['document.md'].write(f"# {meta.get('company',pdf_path.stem)} — FY{meta.get('fiscal_year','unknown')}\n\nSource: `{pdf_path.name}`\n\n")
            for page_number in selected:
                page = pages[page_number-1]
                lines = page.pop('lines')
                quality = _text_quality(lines, config.min_text_characters)
                warnings = list(quality['warnings'])
                if page['rotation']:
                    warnings.append('rotated_page: coordinates are display-aligned; reading order needs review')
                active = [line for line in lines if line.get('kind','text') not in {'header','footer'} and not line['duplicate']]
                excluded = [line for line in lines if line.get('kind') in {'header','footer'}]
                tables = []
                scan = config.table_mode == 'all' or (config.table_mode == 'auto' and _table_candidate(active))
                table_status = 'disabled' if config.table_mode == 'off' else 'skipped_no_numeric_candidates'
                if scan and plumber is not None:
                    pp = plumber.pages[page_number-1]
                    try:
                        # The table module aligns pdfplumber's MediaBox/CropBox
                        # with PyMuPDF's displayed, crop-relative coordinates.
                        tables = extract_tables(pp,active,page['width'],page['height'])
                        table_status = 'scanned'
                        counts['pages_table_scanned'] += 1
                    except Exception as exc:
                        table_status = 'error'
                        warnings.append(f'table_extraction_error: {type(exc).__name__}: {exc}')
                    finally:
                        pp.close()
                for index, table in enumerate(tables,1):
                    table_id = f"{meta['document_id']}:p{page_number}:table{index}"
                    table.update(table_id=table_id, pdf_page=page_number,
                                 citation=_citation(meta,page_number,table['bbox']))
                    outputs['tables.jsonl'].write(json.dumps(table,ensure_ascii=False)+'\n')
                # Keep full source lines even when a structured table replaces them.
                prose = []
                table_source_lines = []
                for line in active:
                    matches = [t for t in tables if _covered(line['bbox'],t['bbox'])>=.8]
                    if matches:
                        table_source_lines.append({**line,'table_id':matches[0]['table_id']})
                    else:
                        prose.append(line)
                # Treat a table as one indivisible layout region.
                placeholders = [{'text':t['markdown'],'bbox':t['bbox'],'font_size':0,
                                 'block_id':-(index+1),'line_id':0,'source_line_id':f'table:{index}',
                                 'kind':'table','table_id':t['table_id']} for index,t in enumerate(tables)]
                ordered = order_lines(prose+placeholders,page['width'],page['height'])
                blocks = lines_to_blocks(ordered)
                for index, block in enumerate(blocks,1):
                    block['element_id'] = f"{meta['document_id']}:p{page_number}:e{index}"
                    block['citation'] = _citation(meta,page_number,block['bbox'])
                    block['reading_order'] = index
                warnings.extend(w for line in ordered for w in line.get('layout_warnings',[]))
                warnings.extend(f"table_review: {w}" for table in tables for w in table.get('warnings',[]))
                page.update(document_id=meta['document_id'], elements=blocks, source_lines=lines,
                            removed_margins=excluded, table_source_lines=table_source_lines,
                            table_ids=[t['table_id'] for t in tables], table_scan=table_status,
                            text_quality=quality, warnings=sorted(set(warnings)),
                            text='\n\n'.join(block['text'] for block in blocks))
                outputs['pages.jsonl'].write(json.dumps(page,ensure_ascii=False)+'\n')
                outputs['document.md'].write(f"## PDF page {page_number}\n\n{page['text']}\n\n")
                counts['pages']+=1
                counts['tables']+=len(tables)
                counts['fallback_tables']+=sum(table.get('structured_reliable') is False for table in tables)
                counts['reliable_structured_tables']+=sum(table.get('structured_reliable') is True for table in tables)
                counts['text_blocks']+=sum(block.get('kind','text')!='table' for block in blocks)
                counts['removed_margin_lines']+=len(excluded)
                counts['duplicate_lines']+=sum(line['duplicate'] for line in lines)
                if warnings:
                    counts['pages_needing_review']+=1
                    review_pages.append({'pdf_page':page_number,'warnings':page['warnings']})
                if any('OCR/review' in warning for warning in warnings):
                    counts['pages_needing_ocr_or_review']+=1
            meta.update(complete=True,counts=counts,review_pages=review_pages,
                        duration_seconds=round(time.monotonic()-started,3))
    finally:
        if plumber is not None:
            plumber.close()
        for output in outputs.values():
            output.close()
    # Invalidate the old commit marker before any artifact replacement. A crash
    # here causes a clean reparse, never reuse of files from mixed generations.
    (output_dir/'document.json.tmp').write_text(json.dumps({**meta,'complete':False},indent=2,ensure_ascii=False),encoding='utf-8')
    (output_dir/'document.json.tmp').replace(document_path)
    for name in outputs:
        (output_dir/(name+'.tmp')).replace(output_dir/name)
    meta['artifacts'] = {name:{'sha256':_hash(output_dir/name),
                               'size_bytes':(output_dir/name).stat().st_size} for name in outputs}
    (output_dir/'document.json.tmp').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    (output_dir/'document.json.tmp').replace(document_path)
    return {**meta,'reused':False,'output_dir':str(output_dir)}
