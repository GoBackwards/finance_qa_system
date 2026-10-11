"""Scope-aware candidate pools and balanced Top-5 selection for live questions.

The live bad-case audit found three retrieval failures that a single global
vector Top-20 cannot avoid: one company of a comparison crowding out the other,
near-identical paragraphs from several report years filling every slot, and
undated or relative questions answered from stale reports. Scoping uses only
the user's question (resolved companies and periods), never benchmark labels.
After the balanced Top-5, ``context_extras`` appends the passages the reranker
tends to drop: statement tables, summary themes and table continuations.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from pathlib import Path
import re

from report_chunker.retrieval import query_terms


SELECTION_VERSION = "scoped-balanced-top5-v2"
POOL_DEPTH = 20
MAX_POOL = 30
LATEST_POOL = 4
LATEST_RESERVED = 2
DUPLICATE_JACCARD = 0.8
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_FIGURE = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?")

# Passages appended after the reranked Top-5 (see context_extras).
CONTEXT_VERSION = "context-extras-v3"
TABLE_SLOT_DEPTH = 20
MAX_EXTRAS = 8
MAX_SUMMARY_EXTRAS = 10
SUMMARY_NEIGHBOURS = 3
HITS_PER_THEME = 2
_TABLE_KINDS = frozenset({"table", "table_fallback"})
_SUMMARY = re.compile(r"\bsummar(?:y|ies|i[sz]e|i[sz]ing)\b", re.I)
_NOT_TERMS = frozenset(
    "report reported reports annual based according using use fy higher lower larger smaller compare compared "
    "company companies fiscal year years which million millions billion billions amount value consolidated total "
    "between".split())
# Row-label families of primary statements; a table naming three or more is statement-like.
_STATEMENT_ROWS = tuple(re.compile(pattern) for pattern in (
    r"\b(?:net )?(?:revenues?|sales)\b", r"\bcost of\b", r"\bgross (?:profit|margin)\b",
    r"\boperating (?:income|profit|earnings|expenses)\b", r"\bincome (?:taxes|before)\b|\bprovision for income\b",
    r"\bnet (?:income|earnings|loss)\b", r"\bper (?:common )?share\b", r"\btotal assets\b", r"\btotal liabilities\b",
    r"\b(?:operating|investing|financing) activities\b"))
_SEGMENT = re.compile(r"(?i)\bsegments?\b")
_THEME_CUE = re.compile(r"(?i)\b(?:concerning|related to|surrounding|associated with|about|from|of|to|on)\b\s*:?\s*")


def document_index(inventory: Iterable[Mapping]) -> dict[str, dict[int, str]]:
    """Map ticker -> fiscal year -> indexed document_id for validated reports."""
    index: dict[str, dict[int, str]] = {}
    for row in inventory:
        if row.get("status", "validated") != "validated":
            continue
        document_id = Path(row["relative_path"]).stem + "-" + row["sha256"][:12]
        index.setdefault(row["ticker"], {})[int(row["fiscal_year"])] = document_id
    return index


def plan_pools(scope: Mapping, documents: Mapping[str, Mapping[int, str]], depth: int = POOL_DEPTH) -> list[dict]:
    """One exact scoped vector search per resolved company; [] means global search.

    Each company gets an equal share of the usual 20-candidate depth (at least
    six), so a comparison cannot be decided by which name dominates the query
    embedding. A company's search is limited to the resolved report years when
    the question names or implies them; otherwise all its reports are searched.
    """
    tickers = [ticker for ticker in scope.get("tickers", ()) if ticker in documents]
    if not tickers:
        return []
    share = max(6, math.ceil(depth / len(tickers)))
    if share * len(tickers) > MAX_POOL:
        share = max(1, MAX_POOL // len(tickers))
    pools = []
    undated = scope.get("period_mode") == "unspecified"
    for ticker in tickers:
        available = documents[ticker]
        years = sorted(year for year in (scope.get("report_years", {}).get(ticker) or ()) if year in available)
        years = years or sorted(available)
        pools.append({"ticker": ticker, "fiscal_years": years,
                      "document_ids": [available[year] for year in years], "top_k": share, "purpose": "scope"})
        latest = max(available)
        if undated and years != [latest]:
            # The all-years pool may hold no latest-report passage at all; this
            # small pool makes the latest-year reservation possible.
            pools.append({"ticker": ticker, "fiscal_years": [latest], "document_ids": [available[latest]],
                          "top_k": LATEST_POOL, "purpose": "latest"})
    return pools


def merge_pools(results: Iterable[list[dict]]) -> list[dict]:
    """Union pool results by chunk ID and assign one global vector rank.

    Every pool is scored against the same query embedding, so cosine scores are
    comparable; per-pool ranks restart at 1 and would collide as tie-breakers.
    """
    merged: dict[str, dict] = {}
    for rows in results:
        for row in rows:
            merged.setdefault(row["chunk_id"], row)
    ordered = sorted(merged.values(), key=lambda row: (-row["score"], row["chunk_id"]))
    for rank, row in enumerate(ordered, 1):
        row["rank"] = rank
    return ordered


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall((text or "").lower()))


def near_duplicate(first: Mapping, second: Mapping) -> bool:
    """Same company's passage repeated in another report with the same figures.

    Boilerplate recurs across a company's reports, but a paragraph that differs
    in its numbers or years is a different fact (FY2021 versus FY2022 margin
    drivers) and must stay selectable.
    """
    if first.get("document_id") == second.get("document_id") or first.get("ticker") != second.get("ticker"):
        return False
    if set(_FIGURE.findall(first.get("text", ""))) != set(_FIGURE.findall(second.get("text", ""))):
        return False
    a, b = _words(first.get("text", "")), _words(second.get("text", ""))
    if len(a) < 20 or len(b) < 20:
        return False
    return len(a & b) / len(a | b) >= DUPLICATE_JACCARD


def select_candidates(reranked: list[dict], scope: Mapping, top_k: int = 5) -> tuple[list[dict], dict]:
    """Pick Top-K from reranked candidates (best first) with three guarantees.

    1. Comparisons reserve floor(K / companies) slots per resolved company.
    2. A passage that near-duplicates an already selected passage from another
       report is skipped while non-duplicates remain.
    3. An undated single-company question reserves two slots for the latest
       available report, so a stale year cannot silently stand in for "now".
    Selected candidates are re-ranked 1..K by reranker score; payloads unchanged.
    """
    present = {candidate.get("ticker") for candidate in reranked}
    tickers = [ticker for ticker in scope.get("tickers", ()) if ticker in present]
    trace = {"selection_version": SELECTION_VERSION, "pool_size": len(reranked), "quota": {},
             "reserved_latest": [], "suppressed_duplicates": []}
    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    def take(candidate: dict) -> bool:
        if candidate["chunk_id"] in chosen_ids or len(chosen) >= top_k:
            return False
        if any(near_duplicate(candidate, other) for other in chosen):
            if candidate["chunk_id"] not in trace["suppressed_duplicates"]:
                trace["suppressed_duplicates"].append(candidate["chunk_id"])
            return False
        chosen.append(candidate)
        chosen_ids.add(candidate["chunk_id"])
        return True

    undated = scope.get("period_mode") == "unspecified"
    latest_years = scope.get("latest_year", {})
    if len(tickers) >= 2:
        quota = max(1, top_k // len(tickers))
        for ticker in tickers:
            taken = 0
            if undated:
                # An undated comparison is answered for the latest reports.
                for candidate in reranked:
                    if (candidate.get("ticker") == ticker and candidate.get("fiscal_year") == latest_years.get(ticker)
                            and take(candidate)):
                        trace["reserved_latest"].append(candidate["chunk_id"])
                        taken += 1
                        break
            for candidate in reranked:
                if taken < quota and candidate.get("ticker") == ticker and take(candidate):
                    taken += 1
            trace["quota"][ticker] = taken
        trace["uncovered"] = [ticker for ticker in tickers if not trace["quota"].get(ticker)]
    elif len(tickers) == 1 and undated:
        latest = latest_years.get(tickers[0])
        for candidate in reranked:
            if len(trace["reserved_latest"]) >= LATEST_RESERVED:
                break
            if (candidate.get("ticker") == tickers[0] and candidate.get("fiscal_year") == latest
                    and take(candidate)):
                trace["reserved_latest"].append(candidate["chunk_id"])
    for candidate in reranked:
        take(candidate)
    # Duplicate suppression must not shrink the context below K when only repeats remain.
    for candidate in reranked:
        if len(chosen) >= top_k:
            break
        if candidate["chunk_id"] not in chosen_ids:
            chosen.append(candidate)
            chosen_ids.add(candidate["chunk_id"])
    chosen.sort(key=lambda c: (-c["reranker_score"], c.get("vector_rank", 0), c["chunk_id"]))
    selected = []
    for rank, candidate in enumerate(chosen, 1):
        item = dict(candidate)
        item["rank"] = rank
        selected.append(item)
    return selected, trace


def _singular(word: str) -> str:
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    return word[:-3] + "y" if word.endswith("ies") else word[:-1] if word.endswith("s") else word


def _label_terms(question: str, names: Iterable[str]) -> list[str]:
    """Question words that could name a table row: no company names, years or comparison words."""
    drop = set(_NOT_TERMS) | {word for name in names for word in query_terms(str(name))}
    return list(dict.fromkeys(_singular(word) for word in query_terms(question)
                              if word not in drop and not re.fullmatch(r"(?:fy)?\d{2,4}", word)))


def _table_score(chunk: Mapping, terms: list[str]) -> tuple | None:
    """Statement-like and non-segment first, then best single-row match, overall match, vector rank."""
    rows = [{_singular(word) for word in query_terms(str(row[0]))} for row in chunk.get("rows") or () if row and row[0]]
    overall = sum(term in set().union(*rows) for term in terms) / len(terms) if terms and rows else 0.0
    if not overall:
        return None
    labels = " ".join(str(row[0]).lower() for row in chunk.get("rows") or () if row and row[0])
    statement = (sum(bool(pattern.search(labels)) for pattern in _STATEMENT_ROWS) >= 3) - 0.5 * bool(
        _SEGMENT.search(chunk.get("text") or ""))
    return statement, max(sum(term in row for term in terms) for row in rows) / len(terms), overall, -chunk["rank"]


def summary_themes(question: str) -> list[str]:
    """Themes a summary question enumerates, e.g. "risks from A, B and C" -> [A, B, C]."""
    sentences = [part for part in re.split(r"(?<=[.?!])\s+", question.strip()) if part]
    sentence = max(sentences, key=lambda part: part.count(","), default="").strip().rstrip(".?")
    if ":" in sentence:
        parts = re.split(r",\s*(?:and\s+|or\s+)?|\s+and\s+", sentence.rsplit(":", 1)[1])
    else:
        segments = re.split(r",\s*", sentence)
        last = segments[-1]
        parts = [last[4:]] if last.lower().startswith("and ") else re.split(r"\s+and\s+", last, maxsplit=1)
        for segment in reversed(segments[:-1]):
            if len(parts) >= 3:
                break
            parts.insert(0, segment)
        cues = list(_THEME_CUE.finditer(parts[0]))
        if cues:
            parts[0] = parts[0][cues[-1].end():]
    return [part.strip() for part in parts if part.strip()]


def context_extras(question: str, scope: Mapping, pools: list[Mapping], selected: list[Mapping],
                   engine, query_vector) -> list[dict]:
    """Passages appended after the reranked Top-5 that the reranker tends to drop.

    The diagnosis of the 100-question benchmark found most lost evidence was
    retrieved but outranked: statement tables lose to prose that repeats the
    question's words, one blended summary query loses whole themes, and a
    table's continuation or a statement title sits in the next chunk. From
    the question, the selected passages and scoped searches of the same
    indexes (no lexicon or gold labels), this adds:
    - non-summary questions: per company, the table among its vector Top-20
      whose row labels best match the question (statement-like tables first,
      half a step lower when they mention segments), and for one company the
      question's best BM25 hit;
    - questions asking for a summary: the selected prose's neighbours most
      similar to the question (risk paragraphs run across chunks), and for one
      company two BM25 hits per enumerated theme;
    - the continuation of a selected table, and the table that follows a
      selected non-table passage on the same or next page.
    The Top-5 is unchanged; at most MAX_EXTRAS (MAX_SUMMARY_EXTRAS) are added.
    """
    chosen = {chunk["chunk_id"] for chunk in selected}
    extras = []
    summary = bool(_SUMMARY.search(question))
    limit = MAX_SUMMARY_EXTRAS if summary else MAX_EXTRAS

    def add(chunk, reason):
        if chunk is not None and chunk["chunk_id"] not in chosen and len(extras) < limit:
            chosen.add(chunk["chunk_id"])
            extras.append(dict(chunk, context_reason=reason))

    scoped = [pool for pool in pools if pool.get("purpose") == "scope"]

    def lexical(text, pool, top_k):
        hits = [hit for year in pool["fiscal_years"]
                for hit in engine.bm25_report(text, top_k, ticker=pool["ticker"], fiscal_year=year)]
        return sorted(hits, key=lambda hit: (hit["score"], hit["chunk_id"]))

    if summary:
        if len(scoped) == 1:
            for theme in summary_themes(question):
                for hit in [hit for hit in lexical(theme, scoped[0], 5) if hit["chunk_id"] not in chosen][:HITS_PER_THEME]:
                    add(hit, "summary_theme")
        # Risk paragraphs run across chunks: the prose neighbours of the selected prose most similar to the question.
        around = {neighbour["chunk_id"]: neighbour for chunk in selected if chunk.get("kind") == "prose"
                  for neighbour in (engine.neighbour(chunk["chunk_id"], -1), engine.neighbour(chunk["chunk_id"], 1))
                  if neighbour is not None and neighbour.get("kind") == "prose" and neighbour["chunk_id"] not in chosen}
        nearby = list(around.values())
        for _, neighbour in sorted(zip(engine.vector_scores(query_vector, nearby), nearby),
                                   key=lambda item: (-item[0], item[1]["chunk_id"]))[:SUMMARY_NEIGHBOURS]:
            add(neighbour, "summary_neighbour")
    else:
        ranked = {pool["ticker"]: engine.vector(query_vector, top_k=TABLE_SLOT_DEPTH, document_ids=pool["document_ids"])
                  for pool in scoped}
        names = {*ranked, *(chunk.get("company", "") for rows in ranked.values() for chunk in rows)}
        terms = _label_terms(question, names)
        for rows in ranked.values():
            scored = [(score, chunk) for chunk in rows if chunk.get("kind") in _TABLE_KINDS
                      and chunk["chunk_id"] not in chosen and (score := _table_score(chunk, terms)) is not None]
            if scored:
                add(max(scored, key=lambda item: item[0])[1], "row_label_table")
        if len(scoped) == 1:
            add(next((hit for hit in lexical(question, scoped[0], 3) if hit["chunk_id"] not in chosen), None),
                "question_bm25")
    for chunk in selected:
        if chunk.get("table_id"):
            for offset in (-1, 1):
                neighbour = engine.neighbour(chunk["chunk_id"], offset)
                if neighbour is not None and neighbour.get("table_id") == chunk["table_id"]:
                    add(neighbour, "same_table")
        elif chunk.get("kind") not in _TABLE_KINDS:
            following = engine.neighbour(chunk["chunk_id"], 1)
            pages = set(chunk.get("source_pages") or ())
            if (following is not None and following.get("kind") in _TABLE_KINDS
                    and any(page in pages or page - 1 in pages for page in following.get("source_pages") or ())):
                add(following, "title_table")
    return extras
