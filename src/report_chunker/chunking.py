"""Token-budgeted, citation-preserving chunks for parsed annual reports.

Prose may cross PDF pages, but never crosses a table. A structured table row is
an atomic unit; unverified table structure is preserved as one source-text unit.
The requested size is a soft limit only for those atomic units (or one Unicode
character whose encoding is larger than the budget). This module never reads
the parser's diagnostic candidate_rows, candidate_cells or raw_rows.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 512
    overlap: int = 50
    tokenizer_name: str = "cl100k_base"

    def __post_init__(self):
        if not isinstance(self.chunk_size, int) or isinstance(self.chunk_size, bool) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer token count")
        if not isinstance(self.overlap, int) or isinstance(self.overlap, bool) or not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must be an integer from 0 through chunk_size - 1")

    @property
    def strategy_id(self) -> str:
        return f"table_preserving_s{self.chunk_size}_o{self.overlap}"


def get_tokenizer(name: str = "cl100k_base"):
    """Return a real BPE encoding, never a characters/4 approximation."""
    import tiktoken
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".cache" / "tiktoken"))
    return tiktoken.get_encoding(name)


def token_count(text: str, tokenizer=None) -> int:
    tokenizer = tokenizer or get_tokenizer()
    # Literal special-token strings inside a financial document are plain text.
    return len(tokenizer.encode(text, disallowed_special=()))


def _base(metadata: dict, config: ChunkConfig, kind: str) -> dict:
    return {key: metadata.get(key) for key in
            ("document_id", "source_file", "ticker", "fiscal_year", "company")} | {
                "schema_version": "1.0", "strategy_id": config.strategy_id,
                "chunk_size": config.chunk_size, "requested_overlap": config.overlap,
                "tokenizer": config.tokenizer_name, "kind": kind}


def _finish(chunk: dict, config: ChunkConfig, tokenizer) -> dict:
    chunk["token_count"] = token_count(chunk["text"], tokenizer)
    chunk["character_count"] = len(chunk["text"])
    chunk["oversized"] = chunk["token_count"] > config.chunk_size
    if not chunk["oversized"]:
        chunk.pop("oversized_reason", None)
    chunk["source_pages"] = sorted({e["citation"]["pdf_page"] for e in chunk["evidence"]
                                     if e.get("citation", {}).get("pdf_page") is not None})
    chunk["warnings"] = sorted(set(chunk.get("warnings", [])))
    return chunk


def _token_character_boundaries(text: str, tokenizer) -> list[int]:
    """Map each BPE boundary to a valid Unicode character boundary.

    BPE tokens can contain only part of a UTF-8 codepoint. Decoding a sliced
    token list therefore corrupts some non-ASCII names; slicing the original
    string at mapped character positions avoids that failure.
    """
    ids = tokenizer.encode(text, disallowed_special=())
    char_bytes = array("I", [0])
    byte_position = 0
    for char in text:
        byte_position += len(char.encode("utf-8"))
        char_bytes.append(byte_position)
    positions = array("I", [0])
    byte_position = 0
    for item in ids:
        byte_position += len(tokenizer.decode_single_token_bytes(item))
        positions.append(bisect_right(char_bytes, byte_position) - 1)
    return positions


def _windows(text: str, paragraph_breaks: list[int], config: ChunkConfig, tokenizer):
    boundaries = _token_character_boundaries(text, tokenizer)
    breaks = sorted(set(paragraph_breaks + [m.end() for m in re.finditer(r"[.!?][\"')\]]*\s+", text)]))
    start, previous_end = 0, 0
    while start < len(text):
        approximate_token = bisect_left(boundaries, start)
        end = boundaries[min(len(boundaries) - 1, approximate_token + config.chunk_size)]
        end = max(start + 1, end)
        # Local BPE boundaries may change after a cut. Enforce the actual count,
        # not the count of a slice from the original tokenization.
        while end > start + 1 and token_count(text[start:end], tokenizer) > config.chunk_size:
            index = bisect_left(boundaries, end) - 1
            end = max(start + 1, boundaries[max(0, index)])
        if end < len(text):
            index = bisect_right(breaks, end) - 1
            if index >= 0:
                natural = breaks[index]
                if natural > start and natural >= start + (end - start) * .6:
                    end = natural
        actual_overlap = token_count(text[start:min(previous_end, end)], tokenizer) if start < previous_end else 0
        yield start, end, actual_overlap
        if end >= len(text):
            break
        # Reuse a suffix of this exact prose stream. No overlap is carried over
        # a table boundary or from the preceding document.
        if config.overlap:
            token_end = bisect_left(boundaries, end)
            next_start = max(start + 1, boundaries[max(0, token_end - config.overlap)])
            next_start = min(next_start, end)
            while next_start < end and token_count(text[next_start:end], tokenizer) > config.overlap:
                next_start += 1
        else:
            next_start = end
        previous_end, start = end, next_start


def _prose_chunks(elements: list[dict], metadata: dict, config: ChunkConfig, tokenizer, *, atomic=False):
    if not elements:
        return
    parts, segments, breaks = [], [], []
    position = 0
    for element in elements:
        text = element.get("text", "")
        if not text:
            continue
        if parts:
            parts.append("\n\n")
            position += 2
            breaks.append(position)
        start = position
        parts.append(text)
        position += len(text)
        segments.append((start, position, element))
    text = "".join(parts)
    segment_ends = [item[1] for item in segments]
    windows = [(0, len(text), 0)] if atomic and text else _windows(text, breaks, config, tokenizer)
    for start, end, overlap in windows:
        evidence = []
        quality_flags = set()
        first = bisect_right(segment_ends, start)
        for source_start, source_end, element in segments[first:]:
            if source_start >= end:
                break
            low, high = max(start, source_start), min(end, source_end)
            if high <= low:
                continue
            quality_flags.update(element.get("_source_quality_flags", []))
            evidence.append({"kind": "prose", "evidence_id": element["element_id"],
                             "element_id": element["element_id"],
                             "text": text[low:high],
                             "char_start": low - source_start, "char_end": high - source_start,
                             "chunk_char_start": low - start, "chunk_char_end": high - start,
                             "citation": dict(element.get("citation", {}))})
        chunk = _base(metadata, config, "page_fallback" if atomic else "prose") | {
            "text": text[start:end], "evidence": evidence, "overlap_tokens": overlap,
            "warnings": ["table_scan_failed_page_atomic: possible unparsed rows retained whole"] if atomic else [],
            "source_quality_flags": sorted(quality_flags),
            "oversized_reason": "failed_table_scan_page_kept_whole" if atomic else "single_unicode_character"}
        yield _finish(chunk, config, tokenizer)


_YEAR = re.compile(r"^(?:19|20)\d{2}(?:\s*\([a-z0-9]+\))?$", re.I)
_DATE_OR_HEADER = re.compile(
    r"\b(?:years?|quarters?|months?|ended|ending|description|item|amount|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
_UNITS = re.compile(r"\b(?:in (?:thousands|millions|billions)|except per.share|"
                    r"dollars in|shares in|unaudited)\b", re.I)


def header_row_indices(table: dict) -> list[int]:
    """Identify source headers conservatively; a data row is never the default.

    Explicit header metadata is authoritative. In its absence, only leading
    date/year rows and clearly labelled, non-numeric column rows are repeated.
    Ambiguous headers remain ordinary rows and are preserved exactly once.
    """
    rows = table.get("rows", [])
    explicit = table.get("header_row_indices")
    if explicit is not None:
        if not isinstance(explicit, list) or any(not isinstance(i, int) or isinstance(i, bool)
                                               or not 0 <= i < len(rows) for i in explicit):
            raise ValueError("Invalid table header_row_indices")
        return sorted(set(explicit))
    result = []
    for index, row in enumerate(rows[:5]):
        values = [str(value).strip() for value in row if value is not None and str(value).strip()]
        if not values:
            break
        all_years = all(_YEAR.fullmatch(value) for value in values)
        date_label = any(_DATE_OR_HEADER.search(value) for value in values)
        # A financial value mixed into the row excludes a header inference.
        numeric_cells = [value for value in values if re.search(r"\d", value)]
        dates_only = all(_YEAR.fullmatch(value) or
                         (bool(_DATE_OR_HEADER.search(value)) and not re.search(r"[$€£¥%]", value))
                         for value in numeric_cells)
        if all_years or (date_label and dates_only and all(len(value.split()) <= 12 for value in values)):
            result.append(index)
        else:
            break
    return result


def _escape(value) -> str:
    return "" if value is None else str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _row_markdown(row: list, columns: int) -> str:
    return "| " + " | ".join(_escape(value) for value in row + [None] * (columns - len(row))) + " |"


def _table_prefix(table: dict) -> tuple[str, list[dict]]:
    parts, evidence = [], []
    caption = table.get("caption")
    if caption:
        parts.append(caption)
        caption_citation = dict(table.get("citation", {}))
        boxes = [item["bbox"] for item in table.get("context", [])
                 if item.get("position") == "above" and item.get("bbox")]
        if boxes:
            caption_citation["bbox"] = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                        max(b[2] for b in boxes), max(b[3] for b in boxes)]
        evidence.append({"kind": "table_context", "evidence_id": table["table_id"] + ":caption",
                         "table_id": table["table_id"], "role": "caption",
                         "text": caption, "citation": caption_citation})
    present_lines = set(caption.splitlines()) if caption else set()
    for index, context in enumerate(table.get("context", [])):
        text = context.get("text", "")
        if text and text not in present_lines and _UNITS.search(text):
            parts.append(text)
            present_lines.add(text)
            citation = dict(table.get("citation", {}))
            if context.get("bbox"):
                citation["bbox"] = context["bbox"]
            evidence.append({"kind": "table_context", "evidence_id": f"{table['table_id']}:context:{index}",
                             "table_id": table["table_id"], "role": "unit_context", "text": text, "citation": citation})
    return "\n".join(parts), evidence


def _with_nearby_unit_context(table: dict, page: dict, table_map: dict) -> dict:
    """Attach a literal unit heading above the table on this same PDF page.

    Native extraction's small table-caption window can miss financial statement
    headings. Search at most half a page upward in the same horizontal region,
    and never cross another table. These are quoted context, not inferred units.
    """
    box = table.get("bbox", table.get("citation", {}).get("bbox"))
    if not box:
        return table
    preceding_bottom = max(0, box[1] - page.get("height", 792) * .5)
    for identifier in page.get("table_ids", []):
        other = table_map.get(identifier, {})
        other_box = other.get("bbox", other.get("citation", {}).get("bbox"))
        if identifier == table["table_id"] or not other_box:
            continue
        x_overlap = min(box[2], other_box[2]) - max(box[0], other_box[0])
        if x_overlap > 0 and other_box[3] <= box[1]:
            preceding_bottom = max(preceding_bottom, other_box[3])
    candidates = []
    for line in page.get("source_lines", []):
        bounds, text = line.get("bbox"), line.get("text", "")
        if not bounds or line.get("duplicate") or line.get("kind") in {"header", "footer"} or not _UNITS.search(text):
            continue
        x_overlap = min(box[2], bounds[2]) - max(box[0], bounds[0])
        if x_overlap >= .25 * min(box[2] - box[0], bounds[2] - bounds[0]) and preceding_bottom <= bounds[1] and bounds[3] <= box[1] + 2:
            candidates.append(line)
    if not candidates:
        return table
    nearest = max(candidates, key=lambda line: line["bbox"][3])
    existing = table.get("context", [])
    if any(item.get("text") == nearest["text"] for item in existing):
        return table
    return {**table, "context": [*existing, {"text": nearest["text"], "bbox": nearest["bbox"],
                                            "position": "unit_heading_above", "source": "same_page_native_line"}]}


def _row_evidence(table: dict, index: int, header: bool) -> dict:
    citation = dict(table.get("citation", {}))
    boxes = [cell["bbox"] for cell in table.get("cells", [])
             if cell.get("row") == index and cell.get("bbox")]
    if boxes:
        citation["bbox"] = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                            max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return {"kind": "row", "evidence_id": f"{table['table_id']}:row:{index}",
            "table_id": table["table_id"], "row_index": index,
            "row_text": " | ".join("" if cell is None else str(cell) for cell in table["rows"][index]),
            "role": "header" if header else "body", "citation": citation}


def _table_chunks(table: dict, metadata: dict, config: ChunkConfig, tokenizer):
    table_id = table["table_id"]
    prefix, prefix_evidence = _table_prefix(table)
    warnings = list(table.get("warnings", []))
    rows = table.get("rows", [])
    if table.get("structured_reliable") is not True or not rows:
        source_text = table.get("source_text")
        if source_text is None:
            source_text = table.get("markdown", "")
            warnings.append("missing_native_table_text: retained complete table placeholder")
        evidence = prefix_evidence + [{"kind": "table_source", "evidence_id": table_id + ":source",
                                      "table_id": table_id, "char_start": 0, "char_end": len(source_text),
                                      "text": source_text,
                                      "citation": dict(table.get("citation", {}))}]
        chunk = _base(metadata, config, "table_fallback") | {
            "text": (prefix + "\n\n" if prefix else "") + source_text,
            "table_id": table_id, "rows": [], "row_indices": [], "header_row_indices": [],
            "structured_reliable": False, "evidence": evidence, "overlap_tokens": 0,
            "source_quality_flags": table.get("_source_quality_flags", []),
            "warnings": warnings + ["unverified_table_atomic: row boundaries are not trusted"],
            "oversized_reason": "unverified_table_kept_whole", "repeated_header_tokens": 0}
        yield _finish(chunk, config, tokenizer)
        return
    header_indices = header_row_indices(table)
    header_set = set(header_indices)
    body_indices = [index for index in range(len(rows)) if index not in header_set]
    columns = max(map(len, rows))
    scaffold = ["| " + " | ".join(f"Column {i + 1}" for i in range(columns)) + " |",
                "| " + " | ".join("---" for _ in range(columns)) + " |"]
    rendered = [_row_markdown(row, columns) for row in rows]
    fixed = ([prefix] if prefix else []) + scaffold + [rendered[i] for i in header_indices]
    fixed_text = "\n".join(fixed)
    fixed_tokens = token_count(fixed_text, tokenizer)
    if not header_indices:
        warnings.append("source_column_headers_not_identified: neutral Column N labels are positional only")

    def emit(indices: list[int], sequence: int):
        indices = header_indices + indices
        text = "\n".join([fixed_text] + [rendered[i] for i in indices if i not in header_set])
        chunk = _base(metadata, config, "table") | {
            "text": text, "table_id": table_id, "row_indices": indices,
            "rows": [rows[i] for i in indices], "header_row_indices": header_indices,
            "structured_reliable": True,
            "source_quality_flags": table.get("_source_quality_flags", []),
            "evidence": prefix_evidence + [_row_evidence(table, i, i in header_set) for i in indices],
            "overlap_tokens": 0, "warnings": warnings,
            "repeated_header_tokens": fixed_tokens if sequence else 0,
            "oversized_reason": "table_prefix_and_atomic_row" if fixed_tokens >= config.chunk_size else "atomic_table_row"}
        return _finish(chunk, config, tokenizer)

    group, sequence = [], 0
    for index in body_indices:
        candidate = "\n".join([fixed_text] + [rendered[i] for i in group + [index]])
        if group and token_count(candidate, tokenizer) > config.chunk_size:
            yield emit(group, sequence)
            sequence += 1
            group = []
        group.append(index)
    if group or not body_indices:
        yield emit(group, sequence)


def chunk_document(metadata: dict, pages: Iterable[dict], tables: Iterable[dict] | Mapping[str, dict],
                   config: ChunkConfig | None = None, tokenizer=None) -> Iterator[dict[str, Any]]:
    """Yield reusable chunks from the parser's pages.jsonl and tables.jsonl.

    Every prose evidence range is relative to the original element string, and
    every table-row index is zero-based in the original reliable rows array.
    Table overlaps repeat source headers/context, not body rows; requested
    overlap applies only to prose and is measured using the same BPE encoding.
    """
    config = config or ChunkConfig()
    tokenizer = tokenizer or get_tokenizer(config.tokenizer_name)
    if not metadata.get("document_id"):
        raise ValueError("document_id is required for stable citations")
    if isinstance(tables, Mapping):
        table_map = dict(tables)
    else:
        table_map = {}
        for table in tables:
            if table["table_id"] in table_map:
                raise ValueError(f"Duplicate table_id: {table['table_id']}")
            table_map[table["table_id"]] = table
    pending, seen_tables = [], set()
    sequence = 0

    def numbered(chunks):
        nonlocal sequence
        for chunk in chunks:
            sequence += 1
            digest = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()[:12]
            chunk["chunk_id"] = f"{metadata['document_id']}:{config.strategy_id}:c{sequence}:{digest}"
            chunk["chunk_index"] = sequence - 1
            yield chunk

    for page in pages:
        quality_flags = [warning for warning in page.get("warnings", [])
                         if any(marker in warning for marker in ("OCR/review", "unusual_text_distribution", "table_extraction_error"))]
        if page.get("table_scan") == "error":
            yield from numbered(_prose_chunks(pending, metadata, config, tokenizer))
            pending = []
            if any(element.get("kind") == "table" for element in page.get("elements", [])):
                raise ValueError("Failed table-scan page unexpectedly contains structured table placeholders")
            annotated = [{**element, "_source_quality_flags": quality_flags} for element in page.get("elements", [])]
            yield from numbered(_prose_chunks(annotated, metadata, config, tokenizer, atomic=True))
            continue
        for element in page.get("elements", []):
            if element.get("kind") != "table":
                if element.get("text"):
                    pending.append({**element, "_source_quality_flags": quality_flags})
                continue
            yield from numbered(_prose_chunks(pending, metadata, config, tokenizer))
            pending = []
            table_id = element.get("table_id", element.get("element_id"))
            if not table_id:
                raise ValueError("Table element lacks both table_id and element_id")
            if table_id in seen_tables:
                raise ValueError(f"Table placeholder occurs twice: {table_id}")
            seen_tables.add(table_id)
            table = table_map.get(table_id)
            if table is None:
                table = {"table_id": table_id, "source_text": element.get("text", ""),
                         "structured_reliable": False, "citation": element.get("citation", {}),
                         "warnings": ["missing_table_record: complete placeholder retained atomically"]}
            table = {**_with_nearby_unit_context(table, page, table_map), "_source_quality_flags": quality_flags}
            yield from numbered(_table_chunks(table, metadata, config, tokenizer))
    yield from numbered(_prose_chunks(pending, metadata, config, tokenizer))
    missing = set(table_map) - seen_tables
    if missing:
        raise ValueError(f"{len(missing)} table records have no page element placeholder")
