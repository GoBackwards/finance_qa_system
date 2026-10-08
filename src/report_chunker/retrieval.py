"""Reproducible lexical retrieval and source-evidence Recall@k evaluation.

SQLite FTS5 supplies BM25 (k1=1.2, b=0.75; lower scores are better) and
the Porter stemmer over unicode61 tokenization. The query is a disjunction
of quoted, deduplicated words from the question, with a fixed stopword list.
The indexed metadata is equally weighted with body text in every experiment.
See https://sqlite.org/fts5.html#the_bm25_function and #porter_tokenizer.

Gold evidence is never supplied to search. Recall counts canonical source
units, not chunk IDs, so overlap cannot create additional relevant units.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable


INDEX_VERSION = 1
STOPWORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had has "
    "have how i if in into is it its me my of on or our s should that the their "
    "them these they this those to us was we were what when where which who why "
    "will with would you your".split()
)
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
RETRIEVER = {
    "name": "sqlite_fts5_bm25",
    "index_version": INDEX_VERSION,
    "sqlite_version": sqlite3.sqlite_version,
    "tokenizer": "porter unicode61",
    "query_mode": "OR of quoted distinct question words; fixed English stopwords removed",
    "stopwords": sorted(STOPWORDS),
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "body_weight": 1.0,
    "metadata_weight": 1.0,
    "score_order": "ascending; lower is better",
    "tie_breaker": "chunk_id ascending",
    "metadata_fields": ["company", "ticker", "fiscal_year", "FY-prefixed fiscal_year"],
    "documentation": "https://sqlite.org/fts5.html",
}


def query_terms(question: str) -> list[str]:
    """Discard FTS syntax, then deduplicate question words in original order."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    return list(dict.fromkeys(
        word for word in WORD_RE.findall(question.casefold())
        if word not in STOPWORDS
    ))


def query_expression(question: str) -> str:
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in query_terms(question))


def _first_citation(chunk: dict) -> dict:
    if chunk.get("citation"):
        return chunk["citation"]
    if chunk.get("citations"):
        return chunk["citations"][0]
    for atom in chunk.get("evidence", []):
        if atom.get("citation"):
            return atom["citation"]
    return {}


def _metadata(chunk: dict) -> tuple[str, str, str, str]:
    citation = _first_citation(chunk)
    ticker = str(chunk.get("ticker") or citation.get("ticker") or "")
    year = str(chunk.get("fiscal_year") or citation.get("fiscal_year") or "")
    company = str(chunk.get("company") or citation.get("company") or "")
    document_id = str(chunk.get("document_id") or citation.get("document_id") or "")
    terms = [company, ticker, year, f"FY{year}" if year else ""]
    return ticker, year, document_id, " ".join(value for value in terms if value)


def build_index(chunks_jsonl: str | Path, index_path: str | Path) -> dict:
    """Stream a complete chunks JSONL into an atomically replaced FTS5 index.

    The source text is stored once in a regular table; its FTS index is an
    external-content table. Payloads retain all evidence mappings. An existing
    finished index survives malformed input or a failed rebuild.
    """
    chunks_path, target = Path(chunks_jsonl), Path(index_path)
    if chunks_path.resolve() == target.resolve():
        raise ValueError("Index path must differ from chunks JSONL")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".building")
    if temporary.exists():
        temporary.unlink()
    hasher = hashlib.sha256()
    count = token_sum = character_sum = 0
    kinds: Counter[str] = Counter()
    documents: set[str] = set()
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA temp_store=FILE")
            connection.executescript("""
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    document_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    fiscal_year TEXT NOT NULL,
                    body TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    body, metadata, content='chunks', content_rowid='id',
                    tokenize='porter unicode61'
                );
            """)
            batch = []
            with chunks_path.open("rb") as source:
                for line_number, raw in enumerate(source, 1):
                    hasher.update(raw)
                    if not raw.strip():
                        continue
                    chunk = json.loads(raw)
                    if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                        raise ValueError(f"Invalid chunk text at JSONL line {line_number}")
                    if not chunk.get("chunk_id"):
                        raise ValueError(f"Missing chunk_id at JSONL line {line_number}")
                    ticker, year, document_id, metadata = _metadata(chunk)
                    if not document_id:
                        raise ValueError(f"Missing document_id at JSONL line {line_number}")
                    body = chunk["text"]
                    payload = {key: value for key, value in chunk.items() if key != "text"}
                    batch.append((chunk["chunk_id"], document_id, ticker, year, body,
                                  metadata, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
                    count += 1
                    token_sum += int(chunk.get("token_count", 0))
                    character_sum += len(body)
                    kinds[str(chunk.get("kind", "unknown"))] += 1
                    documents.add(document_id)
                    if len(batch) >= 500:
                        connection.executemany(
                            "INSERT INTO chunks(chunk_id,document_id,ticker,fiscal_year,body,metadata,payload) "
                            "VALUES (?,?,?,?,?,?,?)", batch)
                        batch.clear()
                if batch:
                    connection.executemany(
                        "INSERT INTO chunks(chunk_id,document_id,ticker,fiscal_year,body,metadata,payload) "
                        "VALUES (?,?,?,?,?,?,?)", batch)
            if count == 0:
                raise ValueError("Cannot build a retrieval index from an empty corpus")
            connection.execute("CREATE INDEX chunks_company_year ON chunks(ticker,fiscal_year)")
            connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
            connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')")
            connection.execute("INSERT INTO chunk_fts(chunk_fts,rank) VALUES('integrity-check',1)")
            stats = {
                "index_version": INDEX_VERSION,
                "chunks_sha256": hasher.hexdigest(),
                "chunks_path": str(chunks_path.resolve()),
                "index_path": str(target.resolve()),
                "chunk_count": count,
                "document_count": len(documents),
                "total_tokens": token_sum,
                "average_tokens": token_sum / count,
                "total_characters": character_sum,
                "average_characters": character_sum / count,
                "chunk_kinds": dict(sorted(kinds.items())),
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "retriever": RETRIEVER,
            }
            connection.execute("INSERT INTO index_metadata VALUES ('stats',?)", (json.dumps(stats),))
            connection.commit()
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite quick_check failed")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return stats


def _open_index(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def index_stats(index_path: str | Path) -> dict:
    with closing(_open_index(index_path)) as connection:
        row = connection.execute("SELECT value FROM index_metadata WHERE key='stats'").fetchone()
        if row is None:
            raise ValueError("Missing completed index metadata")
        return json.loads(row[0])


def _search(connection: sqlite3.Connection, question: str, top_k: int,
            ticker: str | None, fiscal_year: int | str | None) -> list[dict]:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    expression = query_expression(question)
    if not expression:
        return []
    where, values = ["chunk_fts MATCH ?"], [expression]
    if ticker is not None:
        where.append("c.ticker = ?")
        values.append(str(ticker))
    if fiscal_year is not None:
        where.append("c.fiscal_year = ?")
        values.append(str(fiscal_year))
    values.append(top_k)
    sql = (
        "SELECT c.payload,c.body,bm25(chunk_fts,1.0,1.0) AS score "
        "FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.rowid WHERE "
        + " AND ".join(where)
        + " ORDER BY score ASC,c.chunk_id ASC LIMIT ?"
    )
    results = []
    for rank, (payload, body, score) in enumerate(connection.execute(sql, values), 1):
        chunk = json.loads(payload)
        chunk.update(text=body, rank=rank, score=score)
        results.append(chunk)
    return results


def search(index_path: str | Path, question: str, top_k: int = 5, *,
           ticker: str | None = None, fiscal_year: int | str | None = None) -> list[dict]:
    """Retrieve from question text only; optional explicit metadata filters."""
    with closing(_open_index(index_path)) as connection:
        return _search(connection, question, top_k, ticker, fiscal_year)


def _field(atom: dict, chunk: dict, key: str) -> Any:
    aliases = ("source_sha256", "sha256") if key == "source_sha256" else (key,)
    for location in (atom, atom.get("citation", {}), chunk, _first_citation(chunk)):
        for alias in aliases:
            if location.get(alias) is not None:
                return location[alias]
    return None


def _source_matches(gold: dict, atom: dict, chunk: dict) -> bool:
    for key in ("document_id", "pdf_page", "source_sha256"):
        expected = _field(gold, {}, key)
        actual = _field(atom, chunk, key)
        if expected is not None and str(expected) != str(actual):
            return False
    return True


def _validate_gold(gold: dict) -> None:
    if not _field(gold, {}, "document_id"):
        raise ValueError("Every gold evidence location needs document_id")
    if gold.get("table_id") is not None:
        if not isinstance(gold.get("row_index"), int) or isinstance(gold.get("row_index"), bool):
            raise ValueError("Table evidence needs an integer row_index")
        if gold["row_index"] < 0:
            raise ValueError("Table row_index must be nonnegative")
    else:
        if not gold.get("element_id"):
            raise ValueError("Prose evidence needs element_id")
        start, end = gold.get("char_start"), gold.get("char_end")
        if (not isinstance(start, int) or not isinstance(end, int)
                or isinstance(start, bool) or isinstance(end, bool) or start < 0 or end <= start):
            raise ValueError("Prose evidence needs a nonempty source character interval")
        if gold.get("quote") is not None and len(gold["quote"]) != end - start:
            raise ValueError("Gold quote length does not match its source character interval")


def _canonical_unit(gold: dict) -> str:
    fields = {key: _field(gold, {}, key) for key in ("document_id", "pdf_page", "source_sha256")}
    for key in ("element_id", "char_start", "char_end", "table_id", "row_index"):
        if key in gold:
            fields[key] = gold[key]
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _match_location(gold: dict, results: list[dict]) -> dict:
    _validate_gold(gold)
    contributors: set[str] = set()
    if gold.get("table_id") is not None:
        for chunk in results:
            for atom in chunk.get("evidence", []):
                if not _source_matches(gold, atom, chunk):
                    continue
                if atom.get("table_id") != gold["table_id"] or atom.get("row_index") != gold["row_index"]:
                    continue
                actual_text = atom.get("row_text", atom.get("text"))
                if gold.get("row_text") is not None and actual_text != gold["row_text"]:
                    continue
                contributors.add(chunk["chunk_id"])
        return {"recovered": bool(contributors), "chunk_ids": sorted(contributors),
                "reason": "exact source table row recovered" if contributors else "source table row absent"}
    start, end = gold["char_start"], gold["char_end"]
    intervals: list[tuple[int, int]] = []
    rejected_text = 0
    for chunk in results:
        for atom in chunk.get("evidence", []):
            if not _source_matches(gold, atom, chunk) or atom.get("element_id") != gold["element_id"]:
                continue
            atom_start, atom_end = atom.get("char_start"), atom.get("char_end")
            if not isinstance(atom_start, int) or not isinstance(atom_end, int):
                continue
            left, right = max(start, atom_start), min(end, atom_end)
            if right <= left:
                continue
            if gold.get("quote") is not None and atom.get("text") is not None:
                actual = atom["text"][left - atom_start:right - atom_start]
                expected = gold["quote"][left - start:right - start]
                if actual != expected:
                    rejected_text += 1
                    continue
            intervals.append((left, right))
            contributors.add(chunk["chunk_id"])
    merged: list[list[int]] = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    covered = sum(right - left for left, right in merged)
    recovered = covered == end - start
    reason = "full source quote covered by union of retrieved spans" if recovered else "source quote incomplete"
    if rejected_text:
        reason += f"; {rejected_text} mismatched text spans rejected"
    return {"recovered": recovered, "chunk_ids": sorted(contributors), "reason": reason,
            "covered_characters": covered, "required_characters": end - start,
            "covered_intervals": merged}


def judge_retrieval(question: dict, results: list[dict], top_k: int = 5) -> dict:
    """Score canonical evidence against the union of the first k results.

    Alternative source locations satisfy the same evidence unit rather than
    enlarging its denominator. Exact duplicate gold source units are collapsed.
    A repeated retrieved chunk or overlapping character span never adds recall.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    gold_units = question.get("evidence", [])
    if not gold_units:
        raise ValueError(f"Question {question.get('question_id')} has no gold evidence")
    judgments, seen, ids = [], set(), set()
    for number, gold in enumerate(gold_units, 1):
        _validate_gold(gold)
        key = _canonical_unit(gold)
        if key in seen:
            continue
        seen.add(key)
        evidence_id = gold.get("evidence_id", f"evidence_{number}")
        if evidence_id in ids:
            raise ValueError("Distinct gold source units cannot share an evidence_id")
        ids.add(evidence_id)
        locations = [gold] + gold.get("alternatives", [])
        attempts = [_match_location(location, results[:top_k]) for location in locations]
        selected = next((i for i, attempt in enumerate(attempts) if attempt["recovered"]), None)
        best = attempts[selected] if selected is not None else max(
            attempts, key=lambda value: value.get("covered_characters", 0))
        judgments.append({"evidence_id": evidence_id, **best,
                          "matched_location": selected, "location_count": len(locations)})
    recovered = sum(judgment["recovered"] for judgment in judgments)
    total = len(judgments)
    return {"gold_units": total, "recovered_units": recovered,
            f"recall_at_{top_k}": recovered / total,
            f"hit_at_{top_k}": int(recovered > 0),
            f"all_evidence_hit_at_{top_k}": int(recovered == total),
            "evidence_judgments": judgments}


def _aggregate(per_query: Iterable[dict], top_k: int) -> dict:
    rows = list(per_query)
    if not rows:
        return {"question_count": 0, f"recall_at_{top_k}": None,
                f"hit_at_{top_k}": None, f"all_evidence_hit_at_{top_k}": None}
    result = {"question_count": len(rows)}
    for key in (f"recall_at_{top_k}", f"hit_at_{top_k}", f"all_evidence_hit_at_{top_k}"):
        result[key] = sum(row[key] for row in rows) / len(rows)
    result["gold_units"] = sum(row["gold_units"] for row in rows)
    result["recovered_units"] = sum(row["recovered_units"] for row in rows)
    return result


def evaluate(index_path: str | Path, questions: dict | list[dict], *,
             filtered: bool = False, top_k: int = 5) -> dict:
    """Evaluate fixed benchmark questions against a persisted retrieval index.

    By default searches the complete index. ``filtered=True`` is a separately
    labeled diagnostic using explicit benchmark ticker/year metadata; it must
    not be substituted for the unfiltered retrieval score.
    """
    rows = questions["questions"] if isinstance(questions, dict) else questions
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation needs a nonempty list of benchmark questions")
    question_ids = [row["question_id"] for row in rows]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Duplicate benchmark question_id")
    per_query = []
    with closing(_open_index(index_path)) as connection:
        for question in rows:
            ticker = question.get("ticker") if filtered else None
            year = question.get("fiscal_year") if filtered else None
            if filtered and (ticker is None or year is None):
                raise ValueError("Filtered diagnostic requires explicit ticker and fiscal_year")
            results = _search(connection, question["question"], top_k, ticker, year)
            judged = judge_retrieval(question, results, top_k)
            compact_results = []
            for chunk in results:
                citations = chunk.get("citations") or [
                    atom["citation"] for atom in chunk.get("evidence", []) if atom.get("citation")]
                unique_citations = {json.dumps(citation, sort_keys=True): citation for citation in citations}
                recovered_units = [item["evidence_id"] for item in judged["evidence_judgments"]
                                   if item["recovered"] and chunk["chunk_id"] in item["chunk_ids"]]
                compact_results.append({
                    "rank": chunk["rank"], "chunk_id": chunk["chunk_id"], "score": chunk["score"],
                    "document_id": chunk.get("document_id"), "kind": chunk.get("kind"),
                    "token_count": chunk.get("token_count"),
                    "citations": list(unique_citations.values()),
                    "contributes_to_recovered_units": recovered_units,
                    "text_excerpt": chunk["text"][:1200],
                })
            per_query.append({"question_id": question["question_id"], "question": question["question"],
                              "type": question.get("type", "unspecified"),
                              "query_terms": query_terms(question["question"]),
                              "filters": {"ticker": ticker, "fiscal_year": year} if filtered else None,
                              **judged, "retrieved": compact_results})
    types = sorted({row["type"] for row in per_query})
    return {
        "retriever": RETRIEVER,
        "scope": "explicit_company_fiscal_year_filtered_diagnostic" if filtered else "full_corpus_unfiltered",
        "top_k": top_k,
        "recall_definition": "macro mean of canonical gold evidence units completely covered by top-k results",
        "relevance_policy": "exact source identity and table row, or union of full source quote character intervals",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate": _aggregate(per_query, top_k),
        "by_type": {kind: _aggregate((row for row in per_query if row["type"] == kind), top_k) for kind in types},
        "per_query": per_query,
    }
