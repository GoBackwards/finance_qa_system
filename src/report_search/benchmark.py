"""Validate source-grounded retrieval labels without loading the full corpus.

This checks schema, source identity, exact evidence, and split isolation. It
does not establish that an answer's financial interpretation is correct or
that every valid source location has been labeled. Human review remains a
separate benchmark property.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


CATEGORIES = frozenset({"factual", "comparative", "summary"})
SPLITS = frozenset({"dev", "test"})


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _evidence_kind(unit: dict) -> str:
    kind = unit.get("kind")
    if kind in {"table", "row"}:
        return "table"
    if kind in {"prose", "table_source"}:
        return kind
    raise ValueError(f"Unsupported evidence kind {kind!r}")


def _source_key(unit: dict) -> tuple:
    """Source identity excludes freely chosen labels and question IDs."""
    kind = _evidence_kind(unit)
    document = _text(unit.get("document_id"), "evidence document_id")
    page = _integer(unit.get("pdf_page"), "evidence pdf_page", 1)
    sha = _text(unit.get("source_sha256", unit.get("sha256")), "evidence source_sha256")
    if kind == "table":
        table = _text(unit.get("table_id"), "evidence table_id")
        row_index = _integer(unit.get("row_index"), "evidence row_index")
        _text(unit.get("row_text"), "evidence row_text")
        if not isinstance(unit.get("row_cells"), list) or not unit["row_cells"]:
            raise ValueError("Structured evidence requires the complete row_cells")
        return document, sha, page, kind, table, row_index
    identifier = _text(unit.get("table_id" if kind == "table_source" else "element_id"),
                       "evidence source element/table ID")
    start = _integer(unit.get("char_start"), "evidence char_start")
    end = _integer(unit.get("char_end"), "evidence char_end", start + 1)
    quote = _text(unit.get("quote"), "evidence quote")
    if end - start != len(quote):
        raise ValueError("Evidence character range must have exactly the quote's length")
    return document, sha, page, kind, identifier, start, end


def _question_schema(benchmark: dict) -> tuple[list[dict], dict[str, list[tuple[dict, dict]]]]:
    questions = benchmark.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Benchmark must have a nonempty questions list")
    ids, queries = set(), set()
    needed = defaultdict(list)
    source_splits = {}
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object")
        question_id = _text(question.get("question_id"), "question_id")
        if question_id in ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        ids.add(question_id)
        query = _text(question.get("query"), "query")
        if question.get("question") != query:
            raise ValueError(f"query and question must match exactly: {question_id}")
        query_key = " ".join(query.casefold().split())
        if query_key in queries:
            raise ValueError(f"Duplicate query text: {question_id}")
        queries.add(query_key)
        category, split = question.get("category"), question.get("split")
        if category not in CATEGORIES or question.get("type") != category:
            raise ValueError(f"category/type must agree and be factual, comparative, or summary: {question_id}")
        if split not in SPLITS:
            raise ValueError(f"split must be dev or test: {question_id}")
        _text(question.get("answer"), "answer")
        units = question.get("evidence")
        if not isinstance(units, list) or not units:
            raise ValueError(f"Question requires source evidence: {question_id}")
        unit_ids, canonical, document_ids = set(), set(), set()
        for unit in units:
            if not isinstance(unit, dict):
                raise ValueError("Each evidence unit must be an object")
            evidence_id = _text(unit.get("evidence_id"), "evidence_id")
            if evidence_id in unit_ids:
                raise ValueError(f"Duplicate evidence_id within question: {question_id}")
            unit_ids.add(evidence_id)
            key = _source_key(unit)
            if key in canonical:
                raise ValueError(f"Duplicate canonical evidence within question: {question_id}")
            canonical.add(key)
            alternatives = unit.get("alternatives", [])
            if not isinstance(alternatives, list) or any(not isinstance(a, dict) for a in alternatives):
                raise ValueError("Evidence alternatives must be a list of objects")
            locations = [unit, *alternatives]
            for location in locations:
                location_key = _source_key(location)
                previous = source_splits.setdefault(location_key, split)
                if previous != split:
                    raise ValueError("Canonical evidence overlaps development and test splits")
                document_ids.add(location["document_id"])
                needed[location["document_id"]].append((question, location))
        truth = question.get("ground_truth_doc_ids")
        if (not isinstance(truth, list) or any(not isinstance(value, str) or not value for value in truth)
                or len(set(truth)) != len(truth) or set(truth) != document_ids):
            raise ValueError(f"ground_truth_doc_ids must exactly match evidence documents: {question_id}")
        if category == "comparative" and len({unit["document_id"] for unit in units}) < 2:
            raise ValueError(f"Comparative questions require at least two canonical source documents: {question_id}")
        points = question.get("answer_points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"answer_points must contain source-linked claims: {question_id}")
        linked, claims = set(), set()
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("Each answer point must be an object")
            claim = _text(point.get("claim"), "answer point claim")
            claim_key = " ".join(claim.casefold().split())
            if claim_key in claims:
                raise ValueError(f"Duplicate answer point claim: {question_id}")
            claims.add(claim_key)
            references = point.get("evidence_ids")
            if (not isinstance(references, list) or not references
                    or any(not isinstance(value, str) for value in references)
                    or len(set(references)) != len(references)
                    or not set(references) <= unit_ids):
                raise ValueError(f"Answer point references missing or invalid evidence IDs: {question_id}")
            linked.update(references)
        if linked != unit_ids:
            raise ValueError(f"Every required evidence unit must support an answer point: {question_id}")
        if category == "summary" and (not 2 <= len(canonical) <= 4 or len(points) < 3):
            raise ValueError(f"Summary questions need 2-4 distinct evidence units and at least 3 answer points: {question_id}")
    return questions, dict(needed)


def _selected_sources(folder: Path, metadata: dict, locations: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    wanted_elements = {unit["element_id"] for _, unit in locations if _evidence_kind(unit) == "prose"}
    wanted_tables = {unit["table_id"] for _, unit in locations if _evidence_kind(unit) != "prose"}
    elements, tables = {}, {}
    for name in ("pages.jsonl", "tables.jsonl"):
        expected_hash = metadata.get("artifacts", {}).get(name, {}).get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(f"Missing parsed artifact checksum: {folder / name}")
        digest = hashlib.sha256()
        with (folder / name).open("rb") as source:
            for raw in source:
                digest.update(raw)
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if name == "pages.jsonl":
                    for element in record.get("elements", []):
                        element_id = element.get("element_id")
                        if element_id in wanted_elements:
                            if element_id in elements:
                                raise ValueError("Duplicate selected source element ID")
                            # Retain only source evidence fields, not page layouts,
                            # source-line arrays, or other reports' payloads.
                            elements[element_id] = {"text": element.get("text"),
                                                    "citation": element.get("citation"),
                                                    "page_number": record.get("page_number")}
                elif record.get("table_id") in wanted_tables:
                    table_id = record["table_id"]
                    if table_id in tables:
                        raise ValueError("Duplicate selected source table ID")
                    tables[table_id] = {key: record.get(key) for key in
                                       ("structured_reliable", "rows", "source_text", "citation", "pdf_page")}
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Parsed artifact checksum mismatch: {folder / name}")
    if elements.keys() != wanted_elements or tables.keys() != wanted_tables:
        missing = (wanted_elements - elements.keys()) | (wanted_tables - tables.keys())
        raise ValueError(f"Gold source element/table IDs are missing: {sorted(missing)}")
    return elements, tables


def _validate_location(unit: dict, metadata: dict, elements: dict, tables: dict) -> None:
    document_id = unit["document_id"]
    if metadata.get("document_id") != document_id:
        raise ValueError("Source document ID differs from gold document ID")
    expected_sha = unit.get("source_sha256", unit.get("sha256"))
    if metadata.get("sha256") != expected_sha:
        raise ValueError("Gold source SHA256 differs from original parsed document")
    if unit.get("sha256", expected_sha) != expected_sha:
        raise ValueError("Gold source hash aliases disagree")
    for key in ("ticker", "fiscal_year", "source_file"):
        if key in unit and unit[key] != metadata.get(key):
            raise ValueError(f"Gold {key} differs from original parsed document")
    kind = _evidence_kind(unit)
    if kind == "prose":
        source = elements[unit["element_id"]]
        text = source.get("text")
        page = source["page_number"]
    else:
        source = tables[unit["table_id"]]
        page = source["pdf_page"]
        if kind == "table":
            if source.get("structured_reliable") is not True:
                raise ValueError("Structured gold evidence cannot use unverified table cells")
            rows, row_index = source.get("rows"), unit["row_index"]
            if not isinstance(rows, list) or row_index >= len(rows):
                raise ValueError("Gold table row index is outside source rows")
            row = rows[row_index]
            row_text = " | ".join("" if value is None else str(value) for value in row)
            if unit["row_cells"] != row or unit["row_text"] != row_text:
                raise ValueError("Gold table row/cells differ from the complete source row")
            text = None
        else:
            if source.get("structured_reliable") is not False:
                raise ValueError("table_source evidence is only valid for an unstructured source fallback")
            text = source.get("source_text")
    if kind != "table":
        if (not isinstance(text, str) or unit["char_end"] > len(text)
                or text[unit["char_start"]:unit["char_end"]] != unit["quote"]):
            raise ValueError("Gold quote/character range differs from original source text")
    citation = source.get("citation")
    if (not isinstance(citation, dict) or citation.get("document_id") != document_id
            or citation.get("sha256") != expected_sha
            or citation.get("pdf_page") != unit["pdf_page"] or page != unit["pdf_page"]):
        raise ValueError("Original source citation document/hash/PDF page differs from gold")


def validate_benchmark(benchmark_dict: dict, processed_dir: str | Path) -> dict:
    """Validate all labels, loading only selected evidence from one report at a time."""
    questions, needed = _question_schema(benchmark_dict)
    processed_dir = Path(processed_dir)
    folders = {}
    for metadata_path in sorted(processed_dir.glob("*/*/document.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        document_id = metadata.get("document_id")
        if document_id in needed:
            if document_id in folders:
                raise ValueError(f"Duplicate source document ID: {document_id}")
            folders[document_id] = metadata_path.parent
    if folders.keys() != needed.keys():
        raise ValueError(f"Missing benchmark source documents: {sorted(needed.keys() - folders.keys())}")
    companies_by_split = defaultdict(set)
    companies_by_question, years_by_question = defaultdict(set), defaultdict(set)
    for document_id in sorted(needed):
        folder = folders[document_id]
        metadata = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        if metadata.get("complete") is not True or metadata.get("partial"):
            raise ValueError(f"Benchmark source parse is incomplete: {document_id}")
        ticker = _text(metadata.get("ticker"), "source ticker")
        fiscal_year = _integer(metadata.get("fiscal_year"), "source fiscal_year", 1)
        locations = needed[document_id]
        elements, tables = _selected_sources(folder, metadata, locations)
        for question, unit in locations:
            _validate_location(unit, metadata, elements, tables)
            companies_by_split[question["split"]].add(ticker)
            companies_by_question[question["question_id"]].add(ticker)
            years_by_question[question["question_id"]].add(fiscal_year)
        del elements, tables
    overlap = companies_by_split["dev"] & companies_by_split["test"]
    if overlap:
        raise ValueError(f"Development/test companies overlap: {sorted(overlap)}")
    companies = set().union(*companies_by_split.values())
    return {
        "valid": True, "questions": len(questions),
        "required_evidence_units": sum(len(q["evidence"]) for q in questions),
        "source_locations_including_alternatives": sum(len(locations) for locations in needed.values()),
        "source_documents": len(needed), "company_count": len(companies), "companies": sorted(companies),
        "by_category": dict(sorted(Counter(q["category"] for q in questions).items())),
        "by_split": dict(sorted(Counter(q["split"] for q in questions).items())),
        "by_fiscal_year": dict(sorted(Counter(str(year) for years in years_by_question.values() for year in years).items())),
        "questions_by_company": dict(sorted(Counter(ticker for tickers in companies_by_question.values() for ticker in tickers).items())),
        "companies_by_split": {split: sorted(companies_by_split[split]) for split in sorted(SPLITS)},
        "split_company_overlap": [], "cross_split_duplicate_source_units": 0,
        "source_artifact_checksums_verified": True,
        "validation_scope": "Exact parsed source spans/full rows, claim links, source identity, and company-disjoint splits; not semantic answer accuracy or exhaustive relevance.",
        "coverage_counting": "Year/company counts include each distinct source year/company once per question; comparative questions can contribute to several companies.",
        "memory_policy": "Stream each source document; retain only gold-selected element/table fields, then release them.",
    }


def merge_benchmarks(paths: list[str | Path]) -> dict:
    """Merge curated parts deterministically, preserving provenance and hashes.

    Source validation is deliberately separate: callers must validate the
    merged result against processed documents before tuning or evaluation.
    """
    if not paths:
        raise ValueError("At least one benchmark part is required")
    questions, parts = [], []
    for path in map(Path, paths):
        raw = path.read_bytes()
        part = json.loads(raw)
        if not isinstance(part, dict) or not isinstance(part.get("questions"), list):
            raise ValueError(f"Invalid benchmark part: {path}")
        questions.extend(part["questions"])
        parts.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                      "benchmark_id": part.get("benchmark_id"), "provenance": part.get("provenance")})
    merged = {"schema_version": "1.0", "benchmark_id": "financial_retrieval_v1",
              "provenance": {"source_parts": parts, "independent_human_review": False,
                             "status": "source-curated development/test benchmark; human review not established by merge"},
              "questions": sorted(questions, key=lambda question: question.get("question_id", ""))}
    _question_schema(merged)
    return merged
