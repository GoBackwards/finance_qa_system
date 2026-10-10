"""Scope-aware candidate pools and balanced Top-5 selection for live questions.

The live bad-case audit found three retrieval failures that a single global
vector Top-20 cannot avoid: one company of a comparison crowding out the other,
near-identical paragraphs from several report years filling every slot, and
undated or relative questions answered from stale reports. Scoping uses only
the user's question (resolved companies and periods), never benchmark labels.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from pathlib import Path
import re


SELECTION_VERSION = "scoped-balanced-top5-v2"
POOL_DEPTH = 20
MAX_POOL = 30
LATEST_POOL = 4
LATEST_RESERVED = 2
DUPLICATE_JACCARD = 0.8
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_FIGURE = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?")


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
