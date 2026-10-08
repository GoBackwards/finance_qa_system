"""Independent source-conservation checks for one document's chunk output.

This validator does not call the chunker's windowing or table-rendering code.
It checks source characters and original row arrays directly, including the
actual text indexed for retrieval, and never substitutes candidate table rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Iterable, Mapping


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha(citation: dict):
    return citation.get("source_sha256", citation.get("sha256"))


def _box(value, label: str) -> list:
    _require(isinstance(value, (list, tuple)) and len(value) == 4, f"{label}: invalid bbox")
    _require(all(isinstance(number, (int, float)) and not isinstance(number, bool)
                 and math.isfinite(number) for number in value), f"{label}: nonfinite bbox")
    _require(value[0] <= value[2] and value[1] <= value[3], f"{label}: reversed bbox")
    return list(value)


def _same_box(actual, expected, label: str) -> None:
    left, right = _box(actual, label), _box(expected, label + " source")
    _require(all(abs(a - b) <= 1e-6 for a, b in zip(left, right)), f"{label}: source bbox changed")


def _citation(value: dict, metadata: dict, source_pages: set[int], label: str,
              expected: dict | None = None, expected_bbox=None) -> None:
    _require(isinstance(value, dict), f"{label}: missing citation")
    _require(value.get("document_id") == metadata["document_id"], f"{label}: wrong citation document")
    page = value.get("pdf_page")
    _require(_integer(page) and page in source_pages, f"{label}: unknown citation page")
    _require(isinstance(_sha(value), str) and bool(_sha(value)), f"{label}: missing source hash")
    expected_hash = _sha(metadata)
    if expected_hash is not None:
        _require(_sha(value) == expected_hash, f"{label}: wrong source hash")
    _box(value.get("bbox"), label)
    if expected is not None:
        for key in ("document_id", "pdf_page", "source_file", "ticker", "fiscal_year"):
            if expected.get(key) is not None:
                _require(value.get(key) == expected[key], f"{label}: citation {key} changed")
        if _sha(expected) is not None:
            _require(_sha(value) == _sha(expected), f"{label}: citation source hash changed")
        if expected_bbox is None:
            expected_bbox = expected.get("bbox")
    if expected_bbox is not None:
        _same_box(value.get("bbox"), expected_bbox, label)


def _covered(intervals: list[tuple[int, int]], length: int) -> int:
    cursor, total = 0, 0
    for start, end in sorted(intervals):
        if end <= cursor:
            continue
        total += end - max(start, cursor)
        cursor = end
    return total


def _row_text(row: list) -> str:
    return " | ".join("" if value is None else str(value) for value in row)


def _rendered_row(row: list, width: int) -> str:
    escaped = ["" if value is None else str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
               for value in row]
    escaped.extend([""] * (width - len(row)))
    return "| " + " | ".join(escaped) + " |"


def _row_bbox(table: dict, index: int):
    boxes = [cell["bbox"] for cell in table.get("cells", []) if cell.get("row") == index and cell.get("bbox")]
    if not boxes:
        return table.get("citation", {}).get("bbox")
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def validate_document_chunks(metadata: dict, pages: Iterable[dict],
                             tables: Iterable[dict] | Mapping[str, dict],
                             chunks: Iterable[dict]) -> dict:
    """Return counts if all source invariants hold, otherwise raise ValueError.

    Header-row inference is a chunking policy, not re-inferred here. Declared
    headers must be consistent across every chunk of that table; authoritative
    source header indices, when present, must match. Every other row occurs
    exactly once, and every header remains an unchanged, complete source row.
    """
    _require(bool(metadata.get("document_id")), "Missing source document_id")
    pages, chunks = list(pages), list(chunks)
    page_numbers = {page["page_number"] for page in pages}
    _require(len(page_numbers) == len(pages), "Duplicate source page_number")
    _require(all(_integer(number) and number > 0 for number in page_numbers), "Invalid source page_number")
    table_records = list(tables.values()) if isinstance(tables, Mapping) else list(tables)
    table_map = {table["table_id"]: table for table in table_records}
    _require(len(table_map) == len(table_records), "Duplicate source table_id")
    elements, placeholders = {}, {}
    error_pages = set()
    for page in pages:
        if page.get("table_scan") == "error" and any(element.get("text") for element in page.get("elements", [])):
            error_pages.add(page["page_number"])
        for element in page.get("elements", []):
            if element.get("kind") == "table":
                identifier = element.get("table_id", element.get("element_id"))
                _require(identifier not in placeholders, "Duplicate source table placeholder")
                placeholders[identifier] = element
            else:
                identifier = element.get("element_id")
                _require(bool(identifier) and identifier not in elements, "Missing or duplicate source element_id")
                _require(isinstance(element.get("text", ""), str), f"{identifier}: non-string source text")
                elements[identifier] = element
    _require(set(table_map).issubset(placeholders), "Table record lacks source placeholder")
    # The engine also supports conservatively preserving a missing table record.
    for identifier, placeholder in placeholders.items():
        if identifier not in table_map:
            table_map[identifier] = {"table_id": identifier, "structured_reliable": False,
                                     "source_text": placeholder.get("text", ""),
                                     "citation": placeholder.get("citation", {})}
    ids, spans, table_chunks = set(), defaultdict(list), defaultdict(list)
    evidence_count = 0
    for chunk in chunks:
        identifier = chunk.get("chunk_id")
        _require(isinstance(identifier, str) and identifier not in ids, "Missing or duplicate chunk_id")
        ids.add(identifier)
        _require(chunk.get("document_id") == metadata["document_id"], f"{identifier}: wrong document_id")
        for field in ("source_file", "ticker", "fiscal_year", "company"):
            if metadata.get(field) is not None:
                _require(chunk.get(field) == metadata[field], f"{identifier}: changed {field}")
        text = chunk.get("text")
        _require(isinstance(text, str), f"{identifier}: non-string chunk text")
        _require(chunk.get("character_count") == len(text), f"{identifier}: wrong character_count")
        evidence = chunk.get("evidence")
        _require(isinstance(evidence, list) and bool(evidence), f"{identifier}: missing source evidence")
        cited_pages = set()
        for atom in evidence:
            evidence_count += 1
            citation = atom.get("citation")
            _citation(citation, metadata, page_numbers, identifier)
            cited_pages.add(citation["pdf_page"])
            kind = atom.get("kind")
            if kind == "prose":
                element_id = atom.get("element_id")
                _require(element_id in elements, f"{identifier}: unknown prose element")
                source = elements[element_id]
                start, end = atom.get("char_start"), atom.get("char_end")
                _require(_integer(start) and _integer(end) and 0 <= start < end <= len(source["text"]),
                         f"{identifier}: invalid source character offsets")
                expected = source["text"][start:end]
                _require(atom.get("text") == expected, f"{identifier}: source substring changed")
                chunk_start, chunk_end = atom.get("chunk_char_start"), atom.get("chunk_char_end")
                _require(_integer(chunk_start) and _integer(chunk_end)
                         and 0 <= chunk_start < chunk_end <= len(text), f"{identifier}: invalid chunk character offsets")
                _require(text[chunk_start:chunk_end] == expected, f"{identifier}: mapped source text missing in chunk")
                _citation(citation, metadata, page_numbers, identifier, source.get("citation", {}))
                spans[element_id].append((start, end))
            elif kind in {"row", "table_source", "table_context"}:
                table_id = atom.get("table_id")
                _require(table_id in table_map, f"{identifier}: unknown source table")
                _require(table_id == chunk.get("table_id"), f"{identifier}: table evidence mismatches its chunk")
                # Context may have a more precise nearby-line bbox than the table.
                expected = table_map[table_id].get("citation", {})
                identity_only = {key: value for key, value in expected.items() if key != "bbox"}
                _citation(citation, metadata, page_numbers, identifier, identity_only)
            else:
                raise ValueError(f"{identifier}: unknown evidence kind {kind!r}")
        _require(chunk.get("source_pages") == sorted(cited_pages), f"{identifier}: source_pages mismatch")
        if chunk.get("kind") in {"table", "table_fallback"}:
            _require(chunk.get("table_id") in table_map, f"{identifier}: unknown table chunk")
            table_chunks[chunk["table_id"]].append(chunk)
        else:
            _require(chunk.get("kind") in {"prose", "page_fallback"}, f"{identifier}: unknown chunk kind")
            _require(all(atom.get("kind") == "prose" for atom in evidence), f"{identifier}: unexpected non-prose evidence")
    total_characters = 0
    for element_id, source in elements.items():
        expected_length = len(source.get("text", ""))
        _require(_covered(spans[element_id], expected_length) == expected_length,
                 f"{element_id}: source text characters missing from chunks")
        total_characters += expected_length
    body_rows = header_rows = fallback_tables = 0
    for table_id, source in table_map.items():
        emitted = table_chunks[table_id]
        _require(bool(emitted), f"{table_id}: missing table chunks")
        rows = source.get("rows", [])
        if source.get("structured_reliable") is not True or not rows:
            fallback_tables += 1
            _require(len(emitted) == 1 and emitted[0].get("kind") == "table_fallback",
                     f"{table_id}: unverified table must remain one atomic fallback")
            chunk = emitted[0]
            _require(chunk.get("rows") == [] and chunk.get("row_indices") == []
                     and not chunk.get("structured_reliable"), f"{table_id}: untrusted rows used as structured data")
            _require(not any(key in chunk for key in ("candidate_rows", "candidate_cells", "raw_rows")),
                     f"{table_id}: untrusted candidate structures copied")
            atoms = [atom for atom in chunk["evidence"] if atom.get("kind") == "table_source"]
            _require(len(atoms) == 1 and not any(atom.get("kind") == "row" for atom in chunk["evidence"]),
                     f"{table_id}: invalid fallback evidence")
            source_text = source.get("source_text")
            if source_text is None:
                source_text = source.get("markdown", "")
            atom = atoms[0]
            _require(atom.get("text") == source_text and atom.get("char_start") == 0
                     and atom.get("char_end") == len(source_text), f"{table_id}: fallback source text changed")
            _require(source_text in chunk["text"], f"{table_id}: fallback text absent from chunk")
            _citation(atom["citation"], metadata, page_numbers, table_id, source.get("citation", {}))
            continue
        headers = emitted[0].get("header_row_indices")
        _require(isinstance(headers, list)
                 and all(_integer(index) and 0 <= index < len(rows) for index in headers)
                 and len(headers) == len(set(headers)),
                 f"{table_id}: invalid header indices")
        if source.get("header_row_indices") is not None:
            _require(headers == sorted(set(source["header_row_indices"])), f"{table_id}: changed source header indices")
        counts = Counter()
        for chunk in emitted:
            _require(chunk.get("kind") == "table" and chunk.get("structured_reliable") is True,
                     f"{table_id}: reliable table unexpectedly replaced")
            _require(chunk.get("header_row_indices") == headers, f"{table_id}: inconsistent repeated headers")
            indices, chunk_rows = chunk.get("row_indices"), chunk.get("rows")
            _require(isinstance(indices, list) and isinstance(chunk_rows, list)
                     and all(_integer(index) for index in indices)
                     and len(indices) == len(chunk_rows) and len(indices) == len(set(indices)),
                     f"{table_id}: malformed row arrays")
            _require(set(headers).issubset(indices), f"{table_id}: repeated header row missing")
            row_atoms = [atom for atom in chunk["evidence"] if atom.get("kind") == "row"]
            _require(len(row_atoms) == len(indices), f"{table_id}: row evidence count mismatch")
            _require(all(_integer(atom.get("row_index")) for atom in row_atoms), f"{table_id}: invalid row evidence index")
            indexed_atoms = {atom.get("row_index"): atom for atom in row_atoms}
            _require(len(indexed_atoms) == len(row_atoms), f"{table_id}: duplicate row evidence")
            rendered_lines = set(chunk["text"].splitlines())
            for index, row in zip(indices, chunk_rows):
                _require(_integer(index) and 0 <= index < len(rows), f"{table_id}: invalid row index")
                _require(row == rows[index], f"{table_id}: source row cells changed or split")
                _require(_rendered_row(row, max(map(len, rows))) in rendered_lines,
                         f"{table_id}: complete rendered row missing from retrieval text")
                atom = indexed_atoms.get(index, {})
                _require(atom.get("row_text") == _row_text(rows[index]), f"{table_id}: row evidence text changed")
                _require(atom.get("role") == ("header" if index in headers else "body"),
                         f"{table_id}: wrong row role")
                _citation(atom.get("citation"), metadata, page_numbers, table_id,
                          source.get("citation", {}), _row_bbox(source, index))
                counts[index] += 1
        for index in range(len(rows)):
            required = len(emitted) if index in headers else 1
            _require(counts[index] == required, f"{table_id}: source row {index} missing or duplicated")
        body_rows += len(rows) - len(headers)
        header_rows += len(headers)
    for page_number in error_pages:
        containing = [chunk for chunk in chunks if any(atom.get("kind") == "prose"
                      and atom["citation"]["pdf_page"] == page_number for atom in chunk["evidence"])]
        _require(len(containing) == 1 and containing[0].get("kind") == "page_fallback",
                 f"Page {page_number}: failed table-scan page was split")
        _require(containing[0]["source_pages"] == [page_number], f"Page {page_number}: atomic fallback mixed with other pages")
    return {"status": "passed", "documents": 1, "chunks": len(chunks),
            "prose_elements": len(elements), "source_prose_characters": total_characters,
            "structured_tables": len(table_map) - fallback_tables, "table_body_rows": body_rows,
            "table_header_rows": header_rows, "atomic_fallback_tables": fallback_tables,
            "atomic_fallback_pages": len(error_pages), "evidence_atoms": evidence_count,
            "missing_source_characters": 0, "missing_or_split_rows": 0}
