"""Fail-closed reference checks and answer rendering for grounded generation.

An exact source quote proves a reference exists; it does not prove that a
claim follows from that quote. Callers must obtain the semantic verdict in a
separate model request. This module validates that verdict's complete schema
and coverage, but cannot turn a model judgment into a guarantee of truth.

Around that core, three deterministic layers come from a live bad-case audit:
``canonicalize_citations`` rewrites retyped quotes (NBSP typed as a space,
ligatures, a table row quoted as label plus one column, a quote cited to the
neighbouring unit) to exact source spans *before* validation; arithmetic is
done only by ``verify_calculation`` on cited operands; and
``check_claim_grounding`` rejects claims whose years, numbers or companies are
absent from their own citations, or which admit that support is missing.
"""

from __future__ import annotations

import ast
import copy
import difflib
from decimal import Decimal, DecimalException, localcontext
import json
import keyword
import re
import string
import unicodedata
from typing import AbstractSet, Any, Callable


ABSTENTION = "I'm not sure"
MAX_JSON_CHARS = 100_000
MAX_CLAIMS = 20
MAX_CLAIM_CHARS = 2_000
MAX_CITATIONS_PER_CLAIM = 8
MAX_QUOTE_CHARS = 8_000
MAX_REASON_CHARS = 2_000
MAX_EXPRESSION_CHARS = 200
MAX_OPERANDS = 8
MAX_OPERAND_LABEL_CHARS = 120
MAX_OPERAND_VALUE_CHARS = 64
MAX_UNIT_CHARS = 40
_EVIDENCE_ID = re.compile(r"E[1-9][0-9]*\Z")
_INLINE_REFERENCE = re.compile(r"\[(?:\^?[0-9]+|E[0-9]+)\]|【[^】]*】")
_KINDS = frozenset({"prose", "row", "table_source", "table_context"})
_CLAIM_KEYS = (frozenset({"text", "citations"}), frozenset({"text", "citations", "calculation"}))
_CALCULATION_KEYS = frozenset({"expression", "operands", "result", "unit"})
_OPERAND_KEYS = frozenset({"name", "evidence_id", "value", "label"})
_OPERAND_NAME = re.compile(r"[a-z][a-z0-9_]{0,15}\Z")
_PLAIN_DECIMAL = re.compile(r"-?[0-9]{1,30}(?:\.[0-9]{1,30})?\Z")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _object(value: Any) -> dict | None:
    """Accept only a bounded JSON object, without fence stripping or repair."""
    try:
        if isinstance(value, str):
            if len(value) > MAX_JSON_CHARS:
                return None
            value = json.loads(value, object_pairs_hook=_unique_object,
                               parse_constant=_reject_constant)
        elif type(value) is dict:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if len(encoded) > MAX_JSON_CHARS:
                return None
        else:
            return None
        return value if type(value) is dict else None
    except (ValueError, TypeError, OverflowError, RecursionError):
        return None


def _text(value: Any, maximum: int) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= maximum
            and not any(unicodedata.category(char).startswith("C")
                        and char not in "\n\r\t" for char in value))


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _source_entry(identifier: str, entry: Any) -> bool:
    """Check the trusted binding's shape; source revalidation happens upstream."""
    if (type(entry) is not dict or entry.get("evidence_id") != identifier
            or not isinstance(entry.get("kind"), str) or entry["kind"] not in _KINDS
            or not isinstance(entry.get("text"), str) or not entry["text"].strip()):
        return False
    citation, locator = entry.get("citation"), entry.get("source_locator")
    if (type(citation) is not dict or type(locator) is not dict
            or not _text(citation.get("document_id"), 1_000)
            or not _integer(citation.get("pdf_page"), 1)
            or not _text(citation.get("source_file"), 4_000)
            or not _text(citation.get("sha256"), 256)):
        return False
    chunk_ids = entry.get("chunk_ids")
    if (type(chunk_ids) is not list or not chunk_ids
            or not all(_text(item, 1_000) for item in chunk_ids)
            or len(set(chunk_ids)) != len(chunk_ids)):
        return False
    kind = entry["kind"]
    if kind in ("prose", "table_source"):
        key = "element_id" if kind == "prose" else "table_id"
        return (_text(locator.get(key), 1_000)
                and _integer(locator.get("char_start"))
                and _integer(locator.get("char_end"), 1)
                and locator["char_end"] > locator["char_start"])
    if kind == "row":
        return (_text(locator.get("table_id"), 1_000)
                and _integer(locator.get("row_index")))
    return (_text(locator.get("table_id"), 1_000)
            and _text(locator.get("role"), 1_000)
            and ("context_index" not in locator or _integer(locator["context_index"])))


# --- Matching normalization -------------------------------------------------

_SPACE_CATEGORIES = frozenset({"Zs", "Zl", "Zp", "Cc", "Cf"})
_TYPOGRAPHY = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
               "\u2013": "-", "\u2014": "-", "\u2212": "-",
               "\u2010": "-", "\u2011": "-", "\u2012": "-"}


_LEADER = re.compile(r"\.(?:[ \u00a0\t]*\.){2,}")  # dot leaders, incl. no-break spaces


def _normalized_map(text: str, *, keep_pipes: bool = False) -> tuple[str, list[int]]:
    """Normalize for matching only, mapping each output char to its source index.

    Generators retype NBSP/NUL as spaces, PDF ligatures as letters and curly
    quotes as straight ones, and quote table cells without their '|'
    scaffolding. The index map lets a repair copy the *original* source span;
    normalized text is never shown or cited. Pipes become spaces (not deleted)
    so that unspaced cells such as '560|431' cannot fuse into one number.
    Dot leaders ('Membership fees . . . . $') are layout filler whose length
    models miscount, so a run of three or more dots counts as one space.
    """
    leaders = {index for match in _LEADER.finditer(text) for index in range(match.start(), match.end())}
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if index in leaders:
            if chars and chars[-1] != " ":
                chars.append(" ")
                positions.append(index)
            continue
        if " " < char < "\x7f" and char != "|":
            chars.append(char)
            positions.append(index)
            continue
        for out in unicodedata.normalize("NFKC", char):
            out = _TYPOGRAPHY.get(out, out)
            if (out == "|" and not keep_pipes) or unicodedata.category(out) in _SPACE_CATEGORIES:
                if not chars or chars[-1] == " ":
                    continue
                out = " "
            chars.append(out)
            positions.append(index)
    if chars and chars[-1] == " ":
        chars.pop()
        positions.pop()
    return "".join(chars), positions


def _normalized(text: str, *, keep_pipes: bool = False) -> str:
    return _normalized_map(text, keep_pipes=keep_pipes)[0]


def _ascii_word(char: str) -> bool:
    return char.isascii() and char.isalnum()


def _bounded_finds(haystack: str, needle: str):
    """Occurrences that do not cut through an ASCII word or a number.

    '431' must not match inside '1,431' or '431.5', and 'debt' not inside
    'debts'. CJK text has no word spaces, so no boundary is required there.
    """
    start = 0
    while needle:
        found = haystack.find(needle, start)
        if found < 0:
            return
        end = found + len(needle)
        before_ok = (not _ascii_word(needle[0]) or found == 0
                     or (not _ascii_word(haystack[found - 1])
                         and not (needle[0].isdigit() and haystack[found - 1] in ".,"
                                  and found >= 2 and haystack[found - 2].isdigit())))
        after_ok = (not _ascii_word(needle[-1]) or end == len(haystack)
                    or (not _ascii_word(haystack[end])
                        and not (needle[-1].isdigit() and haystack[end] in ".,"
                                 and end + 1 < len(haystack) and haystack[end + 1].isdigit())))
        if before_ok and after_ok:
            yield found
        start = found + 1


def _bounded_find(haystack: str, needle: str) -> int:
    return next(_bounded_finds(haystack, needle), -1)


def _trim(span: str) -> str:
    start, end = 0, len(span)
    while start < end and (span[start] == "|" or span[start].isspace()
                           or unicodedata.category(span[start]) in _SPACE_CATEGORIES):
        start += 1
    while end > start and (span[end - 1] == "|" or span[end - 1].isspace()
                           or unicodedata.category(span[end - 1]) in _SPACE_CATEGORIES):
        end -= 1
    return span[start:end]


# --- Citation repair ---------------------------------------------------------

def _locate(needle: str, source: str, mapped: tuple[str, list[int]]) -> str | None:
    """Return the exact original span whose normalization contains ``needle``."""
    normalized, positions = mapped
    found = _bounded_find(normalized, needle)
    if found < 0:
        return None
    span = _trim(source[positions[found]:positions[found + len(needle) - 1] + 1])
    return span if span and span in source else None


def _cells(text: str) -> list[str]:
    cells = []
    for cell in text.split("|"):
        cell = _normalized(cell)
        if cell.startswith("$"):
            cell = cell[1:].strip()
        if cell:
            cells.append(cell)
    return cells


def _row_cells_match(quote: str, row: str) -> bool:
    """Label plus an in-order subset of the row's cells, e.g. 'Current debt | 431'."""
    wanted, cells = _cells(quote), _cells(row)
    if len(wanted) < 2 or not cells or wanted[0] != cells[0]:
        return False
    remaining = iter(cells[1:])
    return all(any(cell == want for cell in remaining) for want in wanted[1:])


_ESCAPED_QUOTE = re.compile("\x00(22|27)")
_ELLIPSIS = re.compile(r"\s*(?:\.\s?\.\s?\.|…)\s*")


def _ellipsis_span(quote: str, source: str, mapped: tuple[str, list[int]]) -> str | None:
    """'A ... B' quoting one unit becomes the exact source span from A through B.

    The released quote then shows everything between the two parts, so no
    text is silently omitted. Every part must be specific and occur in order.
    """
    parts = [_normalized(part) for part in _ELLIPSIS.split(quote)]
    parts = [part for part in parts if part]
    if len(parts) < 2 or not all(_rebindable(part, bundle_wide=False) for part in parts):
        return None
    normalized, positions = mapped
    start = cursor = None
    for part in parts:
        found = next((hit for hit in _bounded_finds(normalized, part) if cursor is None or hit >= cursor), -1)
        if found < 0:
            return None
        start = found if start is None else start
        cursor = found + len(part)
    span = _trim(source[positions[start]:positions[cursor - 1] + 1])
    return span if span and span in source and len(span) <= MAX_QUOTE_CHARS else None


def _chunk_sequences(bundle: dict) -> list[list[str]]:
    sequences = []
    for chunk in bundle.get("chunks") or []:
        identifiers = chunk.get("evidence_ids") if type(chunk) is dict else None
        if type(identifiers) is list and all(isinstance(item, str) for item in identifiers):
            sequences.append(identifiers)
    return sequences


def _split_across_units(quote: str, identifier: str, evidence: dict, bundle: dict) -> list[tuple[str, str]] | None:
    """Split a quote that runs across consecutive units into exact per-unit quotes.

    The parser stores one visual line or paragraph as several adjacent units
    ('Membership fees . . . $' | '4,828'), and models quote across the
    boundary. Each returned piece is an exact substring of its own unit, so
    the strict validator still decides. The match must be unique: first in
    the chunks containing the cited unit, otherwise across the whole bundle.
    """
    needle = _normalized(quote)
    if not needle or not _rebindable(needle, bundle_wide=True):
        return None
    sequences = _chunk_sequences(bundle)
    preferred = [sequence for sequence in sequences if identifier in sequence]
    for scope in (preferred, sequences):
        hits = []
        for sequence in scope:
            joined, owners = [], []
            for key in sequence:
                entry = evidence.get(key)
                if type(entry) is not dict or not isinstance(entry.get("text"), str):
                    break
                text, positions = _normalized_map(entry["text"])
                if joined and text:
                    joined.append(" ")
                    owners.append(None)
                for char, position in zip(text, positions):
                    joined.append(char)
                    owners.append((key, position))
            else:
                haystack = "".join(joined)
                for found in _bounded_finds(haystack, needle):
                    covered = [owner for owner in owners[found:found + len(needle)] if owner is not None]
                    keys = list(dict.fromkeys(key for key, _ in covered))
                    if len(keys) < 2:
                        continue
                    pieces = []
                    for key in keys:
                        indexes = [position for owner_key, position in covered if owner_key == key]
                        source = evidence[key]["text"]
                        piece = _trim(source[min(indexes):max(indexes) + 1])
                        if piece and piece in source:
                            pieces.append((key, piece))
                    signature = tuple(pieces)
                    if pieces and signature not in hits:
                        hits.append(signature)
        if len(hits) == 1:
            return list(hits[0])
        if len(hits) > 1:
            return None
    return None


def _rebindable(needle: str, *, bundle_wide: bool) -> bool:
    """A quote specific enough that a unique match elsewhere identifies its unit.

    Inside the cited chunk, three digits suffice: PDF tables split into one
    atom per fragment put '4,224' next to the '$' it was cited to (D08).
    Across the whole bundle -- other companies and years -- the contract's
    stricter rule applies (6+ characters with a letter or 3+ digits), so a
    bare '391' is never moved to some other company's table.
    """
    digits = sum(char.isdigit() for char in needle)
    letter = any(char.isalpha() for char in needle)
    if bundle_wide:
        return len(needle) >= 6 and (letter or digits >= 3)
    return digits >= 3 or (len(needle) >= 6 and letter)


def _repair_citation(citation: Any, evidence: dict, cache: dict) -> tuple[str, str, str] | None:
    if type(citation) is not dict:
        return None
    identifier, quote = citation.get("evidence_id"), citation.get("quote")
    if (not isinstance(identifier, str) or not _EVIDENCE_ID.fullmatch(identifier)
            or identifier not in evidence or not isinstance(quote, str)
            or not quote.strip() or len(quote) > MAX_QUOTE_CHARS):
        return None
    entry = evidence[identifier]
    if type(entry) is not dict or not isinstance(entry.get("text"), str):
        return None
    source = entry["text"]
    if quote in source:
        return None
    # gpt-4.1-mini sometimes writes a curly quote as NUL + its ASCII hex ('\x0022').
    quote = _ESCAPED_QUOTE.sub(lambda match: '"' if match.group(1) == "22" else "'", quote)

    def mapped(key: str, keep_pipes: bool = False) -> tuple[str, list[int]]:
        if (key, keep_pipes) not in cache:
            cache[key, keep_pipes] = _normalized_map(evidence[key]["text"], keep_pipes=keep_pipes)
        return cache[key, keep_pipes]

    needle = _normalized(quote)
    if not needle:
        return None
    is_row = entry.get("kind") == "row"
    # Rows keep their pipes here so that a cell-structure repair is labelled as one.
    span = (_locate(_normalized(quote, keep_pipes=True), source, mapped(identifier, True)) if is_row
            else _locate(needle, source, mapped(identifier)))
    if span:
        return identifier, span, "typography"
    if is_row:
        span = _locate(needle, source, mapped(identifier))
        if not span and _row_cells_match(quote, source):
            span = _trim(source)
        if span:
            return identifier, span, "row_cells"
    span = _ellipsis_span(quote, source, mapped(identifier))
    if span:
        return identifier, span, "ellipsis_span"
    trimmed = quote.rstrip().rstrip(".;:,")
    if trimmed != quote.rstrip() and _rebindable(_normalized(trimmed), bundle_wide=False):
        span = _locate(_normalized(trimmed), source, mapped(identifier))
        if span:
            return identifier, span, "terminal_punctuation"
    if "-" in quote or "-" in source:
        # '110-basis-point' for '110 basis-point': same length, so positions map back.
        span = _locate(_normalized(quote.replace("-", " ")), source,
                       _normalized_map(source.replace("-", " ")))
        if span:
            return identifier, span, "hyphenation"
    if not _rebindable(needle, bundle_wide=False):
        return None

    def chunk_ids(unit: dict) -> set[str]:
        items = unit.get("chunk_ids")
        return {item for item in items if isinstance(item, str)} if type(items) is list else set()

    cited_chunks = chunk_ids(entry)
    others = [key for key, other in evidence.items()
              if key != identifier and isinstance(key, str) and _EVIDENCE_ID.fullmatch(key)
              and type(other) is dict and isinstance(other.get("text"), str)]
    same_chunk = [key for key in others if cited_chunks & chunk_ids(evidence[key])]
    document = (entry.get("citation") or {}).get("document_id") if type(entry.get("citation")) is dict else None
    if not any(char.isalpha() for char in needle):
        # A bare figure ('54,228') may only move within its own report: the
        # same digits in another company's table are a different fact.
        others = [key for key in others if document is not None and type(evidence[key].get("citation")) is dict
                  and evidence[key]["citation"].get("document_id") == document]
    for scope, kind in ((same_chunk, "rebound_same_chunk"), (others, "rebound_bundle")):
        if kind == "rebound_bundle" and not _rebindable(needle, bundle_wide=True):
            return None
        hits = []
        for key in scope:
            span = _locate(needle, evidence[key]["text"], mapped(key))
            if span:
                hits.append((key, span))
                if len(hits) > 1:
                    return None  # ambiguous; a wider scope cannot be less ambiguous
        if len(hits) == 1:
            return hits[0][0], hits[0][1], kind
    return None


def _trim_to_longest_block(claim_index: int, claim: dict, evidence: dict, repairs: list[dict]) -> None:
    """Last repair, after all others: keep the longest verbatim block of a near-exact quote.

    Live drafts added or changed a few words inside an otherwise verbatim
    quote. The block must cover at least 70% of the quote (and 60 characters);
    it is a verbatim substring of the cited unit, and the grounding checks and
    the verifier still read the whole unit, not just the quote.
    """
    for citation_index, citation in enumerate(claim["citations"]):
        if type(citation) is not dict or not isinstance(citation.get("quote"), str):
            continue
        entry = evidence.get(citation.get("evidence_id")) if isinstance(citation.get("evidence_id"), str) else None
        source, quote = (entry or {}).get("text"), citation["quote"]
        if not isinstance(source, str) or quote in source or len(quote) > MAX_QUOTE_CHARS:
            continue
        block = difflib.SequenceMatcher(None, quote, source, autojunk=False).find_longest_match(
            0, len(quote), 0, len(source))
        span = _trim(source[block.b:block.b + block.size]) if block.size >= max(60, 0.7 * len(quote)) else ""
        if span and span in source:
            citation["quote"] = span
            repairs.append({"claim_index": claim_index, "citation_index": citation_index, "kind": "longest_block",
                            "from_evidence_id": citation["evidence_id"], "to_evidence_id": citation["evidence_id"]})


def _repair_claim(claim_index: int, citations: list, evidence: dict, cache: dict,
                  repairs: list[dict]) -> list:
    proposals = [_repair_citation(citation, evidence, cache) for citation in citations]

    def final_pair(index: int) -> tuple[Any, Any] | None:
        citation, proposal = citations[index], proposals[index]
        if type(citation) is not dict:
            return None
        if proposal is None:
            return citation.get("evidence_id"), citation.get("quote")
        return proposal[0], proposal[1]

    output, kept = [], []
    for index, (citation, proposal) in enumerate(zip(citations, proposals)):
        if proposal is None:
            output.append(citation)
            kept.append(final_pair(index))
            continue
        identifier, quote, kind = proposal
        log = {"claim_index": claim_index, "citation_index": index, "kind": kind,
               "from_evidence_id": citation["evidence_id"], "to_evidence_id": identifier}
        if identifier != citation["evidence_id"]:
            # Later citations that keep their own ID are fixed; later rebinds
            # are checked against this one when they are processed.
            later = [final_pair(other) for other in range(index + 1, len(citations))
                     if proposals[other] is None or proposals[other][0] == citations[other].get("evidence_id")]
            clashes = [pair for pair in kept + later if pair is not None and pair[0] == identifier]
            if clashes and all(pair[1] == quote for pair in clashes):
                repairs.append(dict(log, kind="duplicate_dropped"))
                continue
            if clashes:
                output.append(citation)  # a different quote under that ID: leave unrepaired
                kept.append((citation.get("evidence_id"), citation.get("quote")))
                continue
        output.append(dict(citation, evidence_id=identifier, quote=quote))
        kept.append((identifier, quote))
        repairs.append(log)
    return output


def _rebind_operands(claim_index: int, claim: dict, evidence: dict, repairs: list[dict]) -> None:
    """Follow a rebound citation with the calculation operands that named it.

    Otherwise a growth claim whose '4,224' citation moved from the '$' atom
    E75 to E76 would fail as 'operand not cited'. Only an operand whose old
    unit is no longer cited, whose citation moved to exactly one unit, and
    whose value occurs in that unit (and not in the old one) is rewritten;
    ``verify_calculation`` still checks the result.
    """
    calculation = claim.get("calculation")
    operands = calculation.get("operands") if type(calculation) is dict else None
    if type(operands) is not list:
        return
    cited = {citation.get("evidence_id") for citation in claim["citations"] if type(citation) is dict}
    moves: dict[str, list[dict]] = {}
    for log in repairs:
        if (log["claim_index"] == claim_index and log["from_evidence_id"] != log["to_evidence_id"]
                and log["kind"] in ("rebound_same_chunk", "rebound_bundle", "duplicate_dropped")):
            moves.setdefault(log["from_evidence_id"], []).append(log)
    for operand_index, operand in enumerate(operands):
        if type(operand) is not dict or not isinstance(operand.get("value"), str):
            continue
        old = operand.get("evidence_id")
        logs = moves.get(old) if isinstance(old, str) and old not in cited else None
        if not logs or len({log["to_evidence_id"] for log in logs}) != 1:
            continue
        new, value = logs[0]["to_evidence_id"], _normalized(operand["value"])
        if (new in cited and value and _bounded_find(_normalized(evidence[new]["text"]), value) >= 0
                and _bounded_find(_normalized(evidence[old]["text"]), value) < 0):
            operand["evidence_id"] = new
            repairs.append({"claim_index": claim_index, "citation_index": logs[0]["citation_index"],
                            "kind": "operand_rebound", "from_evidence_id": old, "to_evidence_id": new,
                            "operand_index": operand_index})


def _merge_same_unit(claim_index: int, claim: dict, evidence: dict, repairs: list[dict]) -> None:
    """Join exact quotes of one unit into the smallest exact span covering them.

    Validation rejects a repeated evidence ID, and the model splits a passage
    into two citations of the same unit when told to cite split passages
    separately. The merged quote shows everything between the two parts.
    """
    output, positions = [], {}
    for index, citation in enumerate(claim["citations"]):
        identifier = citation.get("evidence_id") if type(citation) is dict else None
        entry = evidence.get(identifier) if isinstance(identifier, str) else None
        quote = citation.get("quote") if type(citation) is dict else None
        if type(entry) is not dict or not isinstance(quote, str) or quote not in entry["text"]:
            output.append(citation)
            continue
        if identifier not in positions:
            positions[identifier] = len(output)
            output.append(citation)
            continue
        text = entry["text"]
        first = output[positions[identifier]]["quote"]
        starts = [text.index(first), text.index(quote)]
        ends = [starts[0] + len(first), starts[1] + len(quote)]
        span = text[min(starts):max(ends)]
        if len(span) > MAX_QUOTE_CHARS:
            output.append(citation)
            continue
        output[positions[identifier]] = dict(output[positions[identifier]], quote=span)
        repairs.append({"claim_index": claim_index, "citation_index": index, "kind": "merged_same_unit",
                        "from_evidence_id": identifier, "to_evidence_id": identifier})
    claim["citations"] = output


def _split_claim(claim_index: int, claim: dict, evidence: dict, bundle: dict, repairs: list[dict]) -> None:
    """Replace each still-inexact citation by exact per-unit pieces when possible."""
    citations, output = claim["citations"], []
    cited = {c.get("evidence_id") for c in citations if type(c) is dict}
    for index, citation in enumerate(citations):
        identifier = citation.get("evidence_id") if type(citation) is dict else None
        quote = citation.get("quote") if type(citation) is dict else None
        entry = evidence.get(identifier) if isinstance(identifier, str) else None
        if (type(entry) is not dict or not isinstance(quote, str) or not isinstance(entry.get("text"), str)
                or quote in entry["text"] or len(quote) > MAX_QUOTE_CHARS):
            output.append(citation)
            continue
        pieces = _split_across_units(quote, identifier, evidence, bundle)
        pieces = [(key, piece) for key, piece in pieces or [] if key == identifier or key not in cited]
        remaining = len(citations) - index - 1
        if len(pieces) < 2 or len(output) + len(pieces) + remaining > MAX_CITATIONS_PER_CLAIM:
            output.append(citation)
            continue
        for key, piece in pieces:
            output.append(dict(citation, evidence_id=key, quote=piece))
            cited.add(key)
        repairs.append({"claim_index": claim_index, "citation_index": index, "kind": "split_across_units",
                        "from_evidence_id": identifier, "to_evidence_id": pieces[0][0],
                        "to_evidence_ids": [key for key, _ in pieces]})
        calculation = claim.get("calculation")
        for operand in (calculation.get("operands") if type(calculation) is dict else None) or []:
            if type(operand) is not dict or operand.get("evidence_id") != identifier:
                continue
            value = _normalized(operand.get("value")) if isinstance(operand.get("value"), str) else ""
            holders = [key for key, _ in pieces if value and _bounded_find(_normalized(evidence[key]["text"]), value) >= 0]
            if value and _bounded_find(_normalized(entry["text"]), value) < 0 and len(holders) == 1:
                operand["evidence_id"] = holders[0]
    claim["citations"] = output


def canonicalize_citations(draft: Any, bundle: dict) -> tuple[Any, list[dict]]:
    """Rewrite repairable citations so each quote is an exact source substring.

    Runs before ``validate_draft`` and never raises. Repairs only re-point a
    citation at text that already exists verbatim in the verified bundle; the
    claim text is never touched, and every repaired quote is still checked by
    ``validate_draft`` and the semantic verifier. Ambiguous or short quotes
    are left alone so that validation rejects them as before. Returns the
    deep-copied draft (or the unparseable input unchanged) and a repair log.
    A rebind also moves calculation operands that named the old unit (kind
    'operand_rebound', with an extra 'operand_index'). Drafts beyond the
    validation limits are copied unrepaired, which bounds the matching work.
    """
    try:
        parsed = _object(draft)
        if parsed is None or type(parsed.get("claims")) is not list:
            return draft, []
        repaired = copy.deepcopy(parsed)
        evidence = bundle.get("evidence") if type(bundle) is dict else None
        if type(evidence) is not dict or not evidence or len(repaired["claims"]) > MAX_CLAIMS:
            return repaired, []
        repairs: list[dict] = []
        cache: dict = {}
        for claim_index, claim in enumerate(repaired["claims"]):
            if (type(claim) is dict and type(claim.get("citations")) is list
                    and len(claim["citations"]) <= MAX_CITATIONS_PER_CLAIM):
                claim["citations"] = _repair_claim(claim_index, claim["citations"], evidence,
                                                   cache, repairs)
                _rebind_operands(claim_index, claim, evidence, repairs)
                _split_claim(claim_index, claim, evidence, bundle, repairs)
                _merge_same_unit(claim_index, claim, evidence, repairs)
                _trim_to_longest_block(claim_index, claim, evidence, repairs)
        return repaired, repairs
    except Exception:  # noqa: BLE001 - repair is best effort; validation still decides
        return draft, []


# --- Application-computed calculations --------------------------------------

def _calculation_failure(reason: str) -> dict:
    return {"ok": False, "reason": reason, "computed": None, "expression_display": None}


def _decimal_value(value: str) -> Decimal | None:
    """Parse a source figure: '$ 1,234.5', '25.37%', '(1,234)' or '−5'."""
    text = _normalized(value)
    for char in "$, %":
        text = text.replace(char, "")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    if not re.fullmatch(r"[-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", text):
        return None
    number = Decimal(text)
    return -number if negative else number


def _result_text(result: Any) -> str | None:
    """'-24,796', '+5.0' or '−31.4' -> plain decimal text; anything else -> None."""
    if not isinstance(result, str):
        return None
    text = result.strip().replace(",", "").replace("−", "-")
    text = text[1:] if text.startswith("+") else text
    return text if _PLAIN_DECIMAL.fullmatch(text) else None


def _calculation_schema(calculation: Any) -> bool:
    if type(calculation) is not dict or set(calculation) != _CALCULATION_KEYS:
        return False
    operands, unit = calculation["operands"], calculation["unit"]
    if (not _text(calculation["expression"], MAX_EXPRESSION_CHARS)
            or _result_text(calculation["result"]) is None
            or not isinstance(unit, str) or len(unit) > MAX_UNIT_CHARS
            or any(unicodedata.category(char).startswith("C") for char in unit)
            or type(operands) is not list or not 1 <= len(operands) <= MAX_OPERANDS):
        return False
    names = set()
    for operand in operands:
        if (type(operand) is not dict or set(operand) != _OPERAND_KEYS
                or not isinstance(operand["name"], str) or not _OPERAND_NAME.fullmatch(operand["name"])
                or keyword.iskeyword(operand["name"]) or operand["name"] in names
                or not isinstance(operand["evidence_id"], str)
                or not _text(operand["value"], MAX_OPERAND_VALUE_CHARS)
                or _decimal_value(operand["value"]) is None
                or not _text(operand["label"], MAX_OPERAND_LABEL_CHARS)):
            return False
        names.add(operand["name"])
    return True


_ALLOWED_CONSTANTS = frozenset(Decimal(value) for value in (1, 100, 1000, 10**6, 10**9))
_DECREASE = re.compile(r"(?i)\b(?:decreas|declin|fell|fall|falling|drop|lower|down|shr[ai]nk|shrunk|reduc|contract|"
                       r"less|loss|negative|deficit|dip|slid|slump|plung|weak)|下降|减少|下滑|降低|亏损|下跌|缩减")
_BINARY = {ast.Add: (1, "+"), ast.Sub: (1, "\u2212"), ast.Mult: (2, "\u00d7"), ast.Div: (2, "\u00f7")}


def _evaluate(node: ast.AST, values: dict[str, Decimal], used: set[str]) -> Decimal:
    """Evaluate a whitelisted arithmetic AST with Decimal; anything else raises."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values, used)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _evaluate(node.left, values, used)
        right = _evaluate(node.right, values, used)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return left / right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _evaluate(node.operand, values, used)
        return -operand if isinstance(node.op, ast.USub) else +operand
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        number = Decimal(repr(node.value)) if type(node.value) is float else Decimal(node.value)
        # Only percentage/scale factors: any other constant would let a model
        # inject an arbitrary figure as an "application-computed" value.
        if not number.is_finite() or number not in _ALLOWED_CONSTANTS:
            raise ValueError("constant not allowed")
        return number
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in values:
        used.add(node.id)
        return values[node.id]
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _display(node: ast.AST, shown: dict[str, str]) -> tuple[str, int]:
    """Readable infix form with source values; parentheses only where needed."""
    if isinstance(node, ast.Expression):
        return _display(node.body, shown)
    if isinstance(node, ast.BinOp):
        precedence, symbol = _BINARY[type(node.op)]
        left, left_precedence = _display(node.left, shown)
        right, right_precedence = _display(node.right, shown)
        if left_precedence < precedence:
            left = f"({left})"
        if right_precedence < precedence or (right_precedence == precedence
                                             and isinstance(node.op, (ast.Sub, ast.Div))):
            right = f"({right})"
        return f"{left} {symbol} {right}", precedence
    if isinstance(node, ast.UnaryOp):
        operand, precedence = _display(node.operand, shown)
        if precedence < 3:
            operand = f"({operand})"
        return ("\u2212" if isinstance(node.op, ast.USub) else "+") + operand, 3
    if isinstance(node, ast.Name):
        return shown[node.id], 4
    return repr(node.value), 4


def _negative_context(text: str, start: int, end: int, *, parentheses: bool) -> bool:
    """True if the figure at text[start:end] carries a sign: '-5', '-$5', '$ (1,234)'.

    A spaced '- 5' is not a sign: in rows a lone '-' cell means nil, and
    '2022-2023' is a range.
    """
    before = start - 1 if start > 0 and text[start - 1] == "$" else start
    if (before > 0 and text[before - 1] == "-"
            and (before == 1 or not (text[before - 2].isalnum() or text[before - 2] == "-"))):
        return True
    if not parentheses:
        return False
    while before > 0 and text[before - 1] in " $":
        before -= 1
    after = end
    while after < len(text) and text[after] in " %":
        after += 1
    return before > 0 and text[before - 1] == "(" and after < len(text) and text[after] == ")"


def _operand_in_source(value: str, entry: dict) -> bool:
    """The operand occurs in its unit, and keeps the sign it has there.

    A substring check alone accepts '1,234' from a table cell '(1,234)' and
    turns a loss into a profit. In a row's value cells parentheses mean
    negative; in prose and row labels they usually do not ('increased $1,398
    million (14.4%)', 'Notes due 2027 (4.5%)').
    """
    row = entry.get("kind") == "row"
    source, needle = _normalized(entry["text"], keep_pipes=row), _normalized(value)
    if needle.lstrip("$ ").startswith(("-", "(")):
        return _bounded_find(source, needle) >= 0
    first_cell_end = source.find("|") if row else -1
    return any(not _negative_context(source, found, found + len(needle),
                                     parentheses=0 <= first_cell_end < found)
               for found in _bounded_finds(source, needle))


def _contains_number(text: str, digits: str) -> bool:
    """True if ``digits`` (e.g. '1234.5') appears as a whole number, commas allowed."""
    integer, _, fraction = digits.partition(".")
    forms = {integer}
    if integer.isdigit() and not (len(integer) > 1 and integer.startswith("0")):
        forms.add(f"{int(integer):,}")
    suffix = re.escape("." + fraction) if fraction else ""
    pattern = (r"(?<![0-9])(?<![0-9][.,])(?:" + "|".join(re.escape(form) for form in forms)
               + r")" + suffix + r"(?![0-9])(?![.,][0-9])")
    return re.search(pattern, _normalized(text)) is not None


def verify_calculation(claim: dict, evidence: dict) -> dict:
    """Recompute a claim's derived value from cited source operands.

    Models are unreliable at arithmetic, and a margin or growth rate cannot be
    quoted from a source. The model therefore only names operands (each an
    exact figure from a cited unit) and an expression; the application parses
    the expression with a strict arithmetic whitelist, evaluates it in Decimal,
    and requires the stated result, rounded as written, to match and to appear
    in the claim text. Returns ``{"ok", "reason", "computed", "expression_display"}``.
    """
    if (type(claim) is not dict or not isinstance(claim.get("text"), str)
            or not _calculation_schema(claim.get("calculation"))):
        return _calculation_failure("invalid_calculation_schema")
    calculation = claim["calculation"]
    citations = claim.get("citations") if type(claim.get("citations")) is list else []
    cited = {citation.get("evidence_id") for citation in citations
             if type(citation) is dict and isinstance(citation.get("evidence_id"), str)}
    operands = calculation["operands"]
    if any(operand["evidence_id"] not in cited for operand in operands):
        return _calculation_failure("calculation_operand_not_cited")
    for operand in operands:
        entry = evidence.get(operand["evidence_id"]) if type(evidence) is dict else None
        if (type(entry) is not dict or not isinstance(entry.get("text"), str)
                or not _operand_in_source(operand["value"], entry)):
            return _calculation_failure("calculation_operand_not_in_source")
    values = {operand["name"]: _decimal_value(operand["value"]) for operand in operands}
    used: set[str] = set()
    try:
        tree = ast.parse(calculation["expression"], mode="eval")
        with localcontext() as context:
            context.prec = 28
            computed = _evaluate(tree, values, used)
        display = _display(tree, {operand["name"]: operand["value"] for operand in operands})[0]
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, DecimalException,
            RecursionError, MemoryError, OverflowError):
        return _calculation_failure("invalid_calculation_expression")
    if used != set(values):
        return _calculation_failure("invalid_calculation_expression")
    result = _result_text(calculation["result"])
    decimals = len(result.partition(".")[2])
    if abs(computed - Decimal(result)) > Decimal("0.5") * Decimal(10) ** -decimals + Decimal("1e-12"):
        return {"ok": False, "reason": "calculation_result_mismatch", "computed": str(computed),
                "expression_display": display}
    if not _contains_number(claim["text"], result.lstrip("-")):
        return {"ok": False, "reason": "calculation_result_not_in_claim", "computed": str(computed),
                "expression_display": display}
    # A negative result must read as negative ('-31.4%' or 'fell 31.4%'); a
    # positive result must not be written with a minus sign.
    text = _normalized(claim["text"])
    signed = re.search(r"[-−]\s?\$?\s?" + re.escape(result.lstrip("-")), text) is not None
    if (computed < 0 and not signed and not _DECREASE.search(text)) or (computed > 0 and signed):
        return {"ok": False, "reason": "calculation_direction_mismatch", "computed": str(computed),
                "expression_display": display}
    return {"ok": True, "reason": None, "computed": str(computed), "expression_display": display}


# --- Draft validation ---------------------------------------------------------

def _draft_failure(reason: str) -> dict:
    return {"ok": False, "status": "abstain", "reason": reason,
            "reference_verified": False, "claims": []}


def validate_draft(draft: Any, bundle: dict) -> dict:
    """Validate the generator schema and bind every quote to a known source.

    ``draft`` is either strict JSON text or a JSON object with exactly
    ``status`` and ``claims``. Every claim has exactly ``text`` and
    ``citations``, plus an optional ``calculation`` that is recomputed by
    ``verify_calculation``; every citation has exactly ``evidence_id`` and
    ``quote``. Quotation matching is case-, whitespace-, and
    punctuation-sensitive. A failed result never returns any unverified text.
    """
    parsed = _object(draft)
    if parsed is None:
        return _draft_failure("invalid_draft_json")
    if set(parsed) != {"status", "claims"} or parsed.get("status") not in ("answer", "abstain"):
        return _draft_failure("invalid_draft_schema")
    claims = parsed.get("claims")
    if type(claims) is not list:
        return _draft_failure("invalid_draft_schema")
    if parsed["status"] == "abstain":
        return _draft_failure("generator_abstained" if not claims else "invalid_draft_schema")
    if not 1 <= len(claims) <= MAX_CLAIMS:
        return _draft_failure("invalid_claim_count")
    if (type(bundle) is not dict or bundle.get("source_verified") is not True
            or type(bundle.get("evidence")) is not dict or not bundle["evidence"]):
        return _draft_failure("unverified_evidence_bundle")
    evidence = bundle["evidence"]
    for claim in claims:
        if type(claim) is not dict or set(claim) not in _CLAIM_KEYS:
            return _draft_failure("invalid_claim_schema")
        if not _text(claim["text"], MAX_CLAIM_CHARS):
            return _draft_failure("invalid_claim_text")
        # Only this renderer may insert citation markers into the response.
        if _INLINE_REFERENCE.search(claim["text"]):
            return _draft_failure("inline_citation_in_claim")
        citations = claim["citations"]
        if type(citations) is not list or not 1 <= len(citations) <= MAX_CITATIONS_PER_CLAIM:
            return _draft_failure("missing_or_excess_citations")
        seen = set()
        for citation in citations:
            if type(citation) is not dict or set(citation) != {"evidence_id", "quote"}:
                return _draft_failure("invalid_citation_schema")
            identifier = citation["evidence_id"]
            if not isinstance(identifier, str) or not _EVIDENCE_ID.fullmatch(identifier) or identifier not in evidence:
                return _draft_failure("unknown_evidence_id")
            if identifier in seen:
                return _draft_failure("duplicate_citation")
            seen.add(identifier)
            if not _source_entry(identifier, evidence[identifier]):
                return _draft_failure("invalid_source_binding")
            quote = citation["quote"]
            if not _text(quote, MAX_QUOTE_CHARS):
                return _draft_failure("invalid_quote")
            if quote not in evidence[identifier]["text"]:
                return _draft_failure("quote_not_in_source")
        if "calculation" in claim:
            calculation = verify_calculation(claim, evidence)
            if not calculation["ok"]:
                return _draft_failure(calculation["reason"])
    return {"ok": True, "status": "answer", "reason": None,
            "reference_verified": True, "claims": copy.deepcopy(claims)}


# --- Deterministic claim grounding -------------------------------------------

_EVIDENCE_NOUN = r"(?:evidence|sources?|excerpts?|documents?|information|context|passages?|materials?|data)"
_EVIDENCE_ADJECTIVE = r"(?:provided|supplied|given|available|cited|retrieved|above|accompanying)"
# Bare nouns that only ever mean the model's own inputs, unlike 'data center' or 'sources of cash'.
_INPUT_NOUN = r"(?:evidence|excerpts?|passages?|sources?(?!\s+of\b))"
_INPUTS = r"(?:" + _EVIDENCE_ADJECTIVE + r"\s+" + _EVIDENCE_NOUN + r"|" + _INPUT_NOUN + r")\b"
_DO_NOT = r"(?:does|do|did)(?:\s+not\b|n't)"
_QUALIFIER = r"(?:separate\s+|specific\s+|further\s+|additional\s+|explicit\s+|direct\s+|detailed\s+)?"
_ABSENT_THING = r"(?:mention|information|data|details?|disclosure|breakdown)s?"
_MISSING_VERB = (r"(?:provided|included|available|disclosed|detailed|specified|mentioned|found|given|stated|"
                 r"reported|addressed|shown|broken\s+out|listed|presented)")
_CJK_INPUTS = r"(?:证据|资料|(?<!原)材料|信息|文本|来源)"
_ADMITS_MISSING = re.compile("|".join((
    # "... are not fully detailed in the provided evidence", "isn't mentioned in the sources"
    r"(?:\bnot|n't)\s+(?:explicitly\s+|specifically\s+|separately\s+|directly\s+|fully\s+|clearly\s+)?"
    + _MISSING_VERB + r"\s+(?:in|within|by)\s+(?:any\s+of\s+)?the\s+" + _INPUTS,
    # "the provided evidence does not ...", "the sources provided don't ...", "the excerpts do not ..."
    r"\b" + _EVIDENCE_ADJECTIVE + r"\s+" + _EVIDENCE_NOUN + r"\s+" + _DO_NOT,
    r"\b" + _EVIDENCE_NOUN + r"\s+(?:provided|supplied|given|cited|retrieved)\s+" + _DO_NOT,
    r"\bthe\s+" + _INPUT_NOUN + r"\s+" + _DO_NOT,
    # "no separate mention of X is provided" (C01): the noun is followed by the verb or by a
    # prepositional phrase, so "no data breaches were disclosed" is not an admission.
    r"\bno\s+" + _QUALIFIER + _ABSENT_THING
    + r"(?:\s+(?:of|about|on|regarding|for|by|in|concerning)\b.{0,80}?)?\s+(?:is|are|was|were)\s+"
    r"(?:provided|available|given|included|disclosed|specified|mentioned|stated|reported|found|presented|"
    r"shown|listed|made)\b",
    # "there is no information about X in the provided evidence"
    r"\bno\s+" + _QUALIFIER + r"(?:" + _ABSENT_THING + r"|figures?)\b.{0,120}?\b(?:in|within|from)\s+the\s+"
    + _INPUTS,
    r"\bnot\s+fully\s+(?:detailed|disclosed|provided|specified|available|broken\s+out)",
    r"\bcannot\s+be\s+(?:determined|confirmed|verified|calculated|derived)\s+from\s+the\s+"
    r"(?:provided|supplied|available|given|cited|" + _INPUT_NOUN + r")",
    r"\b(?:insufficient|not\s+enough)\s+(?:evidence|information)\b",
    r"\bthe\s+(?:annual\s+)?(?:report|filing)\s+" + _DO_NOT
    + r"\s+(?:specify|mention|break\s+out|detail|quantify|separately\s+disclose)",
    r"(?:所?提供的|现有的?|给定的?|引用的)" + _CJK_INPUTS + r"中?(?:并)?(?:未|没有)",
    _CJK_INPUTS + r"中(?:并)?(?:未|没有)(?:提及|提供|披露|说明|包含)",
    r"(?:未|没有)在?(?:所?提供的|现有的?|给定的?|引用的)" + _CJK_INPUTS + r"中",
    r"无法(?:从|根据)(?:所?提供的|现有的?|给定的?|引用的)?" + _CJK_INPUTS
    + r"中?(?:确定|得出|判断|计算|确认)",
    r"(?:证据|资料|信息)不足",
)), re.IGNORECASE | re.DOTALL)

_YEAR = re.compile(r"(?<![0-9])(?<![0-9][.,])(?:(?<![A-Za-z])FY\s?'?(?P<short>[0-9]{2})(?![0-9])"
                   r"|(?P<full>199[0-9]|20[0-9]{2})(?![0-9])(?![.,][0-9])"
                   r"(?:\s?[-/]\s?(?P<tail>[0-9]{2})(?![0-9]))?)", re.IGNORECASE)
# A four-digit figure with a currency or a scale is an amount, not a year: '$2024 million'.
_AMOUNT_BEFORE = re.compile(r"(?:[$€£¥]|\b(?:USD|RMB|CNY|EUR|GBP|JPY|HKD|CAD|AUD|CHF|INR|KRW|TWD|SGD))\s?\Z",
                            re.IGNORECASE)
_AMOUNT_AFTER = re.compile(r"\s?(?:%|percent\b|per\s?cent\b|thousand\b|millions?\b|billions?\b|trillions?\b"
                           r"|万|亿|百万|千|美元|元|个百分点)|(?:mn|bn|[kmb])(?![A-Za-z0-9])", re.IGNORECASE)
_MONTH_NAMES = ("Jan(?:uary)?", "Feb(?:ruary)?", "Mar(?:ch)?", "Apr(?:il)?", "May", "June?", "July?",
                "Aug(?:ust)?", "Sep(?:tember|t)?", "Oct(?:ober)?", "Nov(?:ember)?", "Dec(?:ember)?")
# Case-sensitive (Title or UPPER case), so the verb in '25 may rise' is not a date.
_MONTH = "(?P<month>" + "|".join(f"{name}|{name.upper()}" for name in _MONTH_NAMES) + ")"
# Calendar dates are periods, not quantities: 'December 31', '31 December', '12月31日'.
_DATES = (
    re.compile(r"(?<![A-Za-z])" + _MONTH + r"\.?\s?(?P<day>[0-9]{1,2})(?:st|nd|rd|th)?(?![0-9A-Za-z])"),
    re.compile(r"(?<![0-9.,])(?P<day>[0-9]{1,2})(?:st|nd|rd|th)?\s(?:of\s)?" + _MONTH + r"(?![A-Za-z])"),
    re.compile(r"(?<![0-9])(?P<month>[0-9]{1,2})\s?月\s?(?P<day>[0-9]{1,2})\s?[日号]"),
)
_CJK_MONTH = re.compile(r"(?<![0-9])[0-9]{1,2}\s?月")
_MONTH_NUMBERS = {name: number for number, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_SCALE_WORD = (r"(?:\s?(?P<scale>万亿|千万|百万|十亿|万|亿|千|thousand|millions?|billions?|trillions?"
               r"|mn|bn|[kmbx])(?![A-Za-z0-9]))?")
_NUMBER = re.compile(r"(?<![0-9])(?<![0-9][.,])(?P<number>[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?"
                     r"|[0-9]+(?:\.[0-9]+)?|(?<![A-Za-z])\.[0-9]+)(?![0-9])" + _SCALE_WORD, re.IGNORECASE)
# Percentages, points and multiples are dimensionless: a scale word never applies to them.
_DIMENSIONLESS_AFTER = re.compile(r"\s?(?:%|percent\b|per\s?cent\b|percentage\s+points?\b|pp\b|ppts?\b"
                                  r"|bps?\b|basis\s+points?\b|个百分点)", re.IGNORECASE)
_NOT_A_QUANTITY = re.compile(r"(?<![A-Za-z0-9])(?:10-K|10-Q|8-K|20-F|11-K|401\(k\)|401k|COVID-19"
                             r"|S&P\s?500|Fortune\s?500)(?![A-Za-z0-9])", re.IGNORECASE)
# Letters glued to digits usually make an identifier (H100, Q4, MI300X, 5G, 10b5) -- unless
# they are a currency code or unit ('USD9,700', '16.9pp', '50bps') or the figure has a
# separator or decimals, which identifiers do not.
_CURRENCY_CODES = frozenset({"usd", "us", "rmb", "cny", "eur", "gbp", "jpy", "hkd", "cad", "aud", "chf",
                             "inr", "krw", "twd", "sgd", "nzd", "rs"})
_UNIT_SUFFIXES = _CURRENCY_CODES | {"pp", "ppt", "ppts", "bp", "bps", "pt", "pts", "pct", "percent",
                                    "yen", "yuan", "dollars", "dollar"}
_PERCENT_AFTER = re.compile(r"\s?(?:%|percent\b|per\s?cent\b|percentage\s+points?\b|pp\b)", re.I)
_ORDINALS = frozenset({"st", "nd", "rd", "th"})
_SCALES = {"thousand": 10**3, "k": 10**3, "千": 10**3, "万": 10**4, "million": 10**6,
           "millions": 10**6, "mn": 10**6, "m": 10**6, "百万": 10**6, "千万": 10**7, "亿": 10**8,
           "billion": 10**9, "billions": 10**9, "bn": 10**9, "b": 10**9, "十亿": 10**9,
           "trillion": 10**12, "trillions": 10**12, "万亿": 10**12, "x": 1}
_SOURCE_SCALES = (1, 10**3, 10**6, 10**9)


def _grounding_result(reason: str | None = None, claim_index: int | None = None,
                      detail: str | None = None) -> dict:
    return {"ok": reason is None, "reason": reason, "claim_index": claim_index, "detail": detail}


def _years(text: str) -> list[tuple[int, tuple[int, int]]]:
    """Year tokens with spans: '2025', 'FY2025', 'FY24', '2025财年', both ends of '2023-24'.

    A four-digit figure with a currency sign or scale ('$2024 million') is an
    amount and is left to the number check instead of passing as a period.
    """
    found = []
    for match in _YEAR.finditer(text):
        if match["short"]:
            found.append((2000 + int(match["short"]), match.span()))
            continue
        start, end = match.span("full")
        if _AMOUNT_BEFORE.search(text, 0, start) or _AMOUNT_AFTER.match(text, end):
            continue
        year = int(match["full"])
        found.append((year, (start, end)))
        if match["tail"] and int(match["tail"]) == (year + 1) % 100:
            found.append((year + 1, match.span("tail")))
    return found


def _dates(text: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """(month, day) pairs with their spans, so 'Dec. 31' matches 'December 31' or '12月31日'."""
    found = []
    for pattern in _DATES:
        for match in pattern.finditer(text):
            month = match["month"]
            month = int(month) if month.isdigit() else _MONTH_NUMBERS[month[:3].lower()]
            day = int(match["day"])
            if 1 <= month <= 12 and 1 <= day <= 31:
                found.append(((month, day), match.span()))
    return found


def _letters_around(text: str, start: int, end: int) -> tuple[str, str]:
    """The ASCII letter runs glued to text[start:end] on each side, lower-cased."""
    before = start
    while before > 0 and text[before - 1].isascii() and text[before - 1].isalpha():
        before -= 1
    after = end
    while after < len(text) and text[after].isascii() and text[after].isalpha():
        after += 1
    return text[before:start].lower(), text[end:after].lower()


def _claim_numbers(text: str) -> list[tuple[str, Decimal, int, int]]:
    """Numbers the claim asserts: (token, |value|, decimals, scale).

    Years and calendar dates are periods (checked separately), and form names
    such as '10-K' or '401(k)' are not quantities.
    """
    skipped = ([span for _, span in _years(text)] + [span for _, span in _dates(text)]
               + [match.span() for pattern in (_NOT_A_QUANTITY, _CJK_MONTH) for match in pattern.finditer(text)])
    numbers = []
    for match in _NUMBER.finditer(text):
        start, end = match.span("number")
        if any(start < stop and end > begin for begin, stop in skipped):
            continue
        raw = match["number"]
        separated = "," in raw or "." in raw
        prefix, suffix = _letters_around(text, start, match.end())
        if suffix in _ORDINALS or (prefix and prefix not in _CURRENCY_CODES and not separated):
            continue  # ordinals and identifiers: 21st, H100, Q4, MI300X
        if suffix and suffix not in _UNIT_SUFFIXES and not separated:
            continue  # identifiers: 5G, 10b5, 3D
        scale = _SCALES[match["scale"].lower()] if match["scale"] else 1
        percent = bool(_PERCENT_AFTER.match(text, match.end()))
        if match["scale"] is None and "." not in raw and len(raw) == 1 and not percent:
            continue  # single digits: list markers, footnotes, "Scope 1" (but "grew 5%" is checked)
        decimals = len(raw.partition(".")[2])
        numbers.append((match.group(0).strip(), abs(Decimal(raw.replace(",", ""))), decimals, scale))
    return numbers


def _source_numbers(text: str) -> list[tuple[Decimal, int | None, bool]]:
    """Figures in cited text: (|value|, explicit scale or None, dimensionless)."""
    numbers = []
    for match in _NUMBER.finditer(text):
        scale = _SCALES[match["scale"].lower()] if match["scale"] else None
        dimensionless = match["scale"] is None and _DIMENSIONLESS_AFTER.match(text, match.end()) is not None
        numbers.append((abs(Decimal(match["number"].replace(",", ""))), scale, dimensionless))
    return numbers


def _dimensionless_unit(unit: str) -> bool:
    unit = unit.strip().lower()
    return ("%" in unit or "percent" in unit or "point" in unit
            or unit in {"x", "times", "pp", "ppt", "pts", "bp", "bps"})


def _number_grounded(value: Decimal, decimals: int, scale: int,
                     candidates: list[tuple[Decimal, int | None, bool]]) -> bool:
    """A claim figure may restate a source figure at another scale, rounded as written.

    '$6.690 billion' restates '6,690' (in millions); '$4.8 billion' rounds
    '4,828'. A percentage is never rescaled, so '9.7%' cannot ground '$9,700
    million'. Units columns are not parsed, so magnitude stays a semantic check.
    """
    target = value * scale
    tolerance = Decimal("0.5") * Decimal(10) ** -decimals * scale
    for number, own, dimensionless in candidates:
        factors = (1,) if dimensionless else _SOURCE_SCALES + ((own,) if own else ())
        for factor in factors:
            actual = number * factor
            if abs(target - actual) <= tolerance + Decimal("1e-9") * max(Decimal(1), actual):
                return True
    return False


def _claim_sources(claim: dict, bundle: dict) -> dict:
    """Texts, fiscal years and tickers of exactly what this claim cites."""
    evidence = bundle["evidence"]
    cited = [citation["evidence_id"] for citation in claim["citations"]]
    entries = [evidence[identifier] for identifier in cited]
    chunk_ids = {item for entry in entries for item in entry["chunk_ids"]}
    chunks = []
    raw_chunks = bundle.get("chunks") if type(bundle.get("chunks")) is list else []
    for chunk in raw_chunks:
        if type(chunk) is not dict:
            continue
        identifiers = chunk.get("evidence_ids") if type(chunk.get("evidence_ids")) is list else []
        if ((isinstance(chunk.get("chunk_id"), str) and chunk["chunk_id"] in chunk_ids)
                or any(isinstance(item, str) and item in cited for item in identifiers)):
            chunks.append(chunk)
    unit_texts = [entry["text"] for entry in entries] + [citation["quote"] for citation in claim["citations"]]
    passage_texts = unit_texts + [chunk["text"] for chunk in chunks if isinstance(chunk.get("text"), str)]
    metadata = [entry["citation"] for entry in entries] + chunks
    return {"unit_texts": unit_texts, "passage_texts": passage_texts,
            "fiscal_years": {item["fiscal_year"] for item in metadata
                             if type(item.get("fiscal_year")) is int},
            "tickers": {item["ticker"].upper() for item in metadata if isinstance(item.get("ticker"), str)}}



_MONTH_DATE = re.compile(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},\s*(?:19|20)\d\d")
_FOOTNOTE_MARK = re.compile(r"\s*(?:\(\d\)|\*+)\s*$")


def _cell_value(text) -> Decimal | None:
    digits = re.sub(r"[\s$,%]", "", text if isinstance(text, str) else "").strip("()")
    return Decimal(digits) if re.fullmatch(r"\d+(?:\.\d+)?", digits) else None


def _column_year_mismatch(claim: dict, bundle: dict) -> str | None:
    """A claimed table figure that sits in another fiscal year's column of its cited row.

    Row units carry each cell's column year, bound by x position in the
    evidence store. Accepted years for the claim: the one year it names outside
    dates; else the year of its one date, plus the year before for a January or
    February date (a fiscal year 'ended January 1, 2022' is FY2021, while
    'December 31, 2024' is only 2024); else the cited report's year. A claim
    naming several years is not checked. Years are also compared as offsets
    from the latest column, since some reports label a fiscal year by its
    calendar end. Calculation operands use their label's year.
    """
    evidence = bundle["evidence"]
    stripped = {year for year, _ in _years(_MONTH_DATE.sub(" ", claim["text"]))}
    dates = {(match.group()[:3].lower(), int(match.group()[-4:])) for match in _MONTH_DATE.finditer(claim["text"])}
    if len(stripped) > 1 or (not stripped and len({year for _, year in dates}) > 1):
        accepted = None
    else:
        accepted = stripped or {year - back for month, year in dates
                                for back in ((0, 1) if month in ("jan", "feb") else (0,))}
    figures = [(value, scale) for _, value, _, scale in _claim_numbers(claim["text"])]
    targets = [(citation["evidence_id"], figures, accepted) for citation in claim["citations"]] if accepted is not None else []
    for operand in (claim.get("calculation") or {}).get("operands") or ():
        label_years = {year for year, _ in _years(operand.get("label") or "")}
        value = _cell_value(operand.get("value"))
        if value is not None and len(label_years) == 1:
            targets.append((operand["evidence_id"], [(value, 1)], label_years))
    for identifier, values, years in targets:
        entry = evidence.get(identifier) or {}
        cells, report_year = entry.get("column_years"), (entry.get("citation") or {}).get("fiscal_year")
        if not cells or not isinstance(report_year, int):
            continue
        years = years or {report_year}
        latest = max(year for _, year in cells)
        for position, (value, scale) in enumerate(values, 1):
            wanted = {value, value * scale / 10**6, value * scale / 10**3}
            hits = {year for text, year in cells if _cell_value(text) in wanted}
            if hits and not hits & years and not {report_year - year for year in years} & {latest - year for year in hits}:
                label = "/".join(map(str, sorted(years)))
                return (f"figure {position} is in the {'/'.join(map(str, sorted(hits)))} column of cited row "
                        f"{identifier}, not the {label} column; use that column's value or name the year the "
                        "figure belongs to")
    return None


def _footnoted_row(claim: dict, bundle: dict) -> str | None:
    """A figure cited from a footnoted row label while the same report's unmarked row is in context."""
    evidence = bundle["evidence"]
    for citation in claim["citations"]:
        entry = evidence.get(citation["evidence_id"]) or {}
        label = entry.get("text", "").split("|")[0].strip()
        if entry.get("kind") != "row" or not _FOOTNOTE_MARK.search(label):
            continue
        plain, origin = _FOOTNOTE_MARK.sub("", label).strip().lower(), entry.get("citation") or {}
        for identifier, other in evidence.items():
            source = other.get("citation") or {}
            if (other.get("kind") == "row" and identifier != citation["evidence_id"]
                    and other["text"].split("|")[0].strip().lower() == plain
                    and (source.get("ticker"), source.get("fiscal_year")) == (origin.get("ticker"), origin.get("fiscal_year"))):
                return (f"cited row {citation['evidence_id']} carries a footnote marker; {identifier} is the same "
                        "report's unmarked row with the same label. Cite that row unless the question asks for "
                        "the footnoted measure")
    return None

_QUALIFIERS = {
    # One family: the live audit restated "Organic Net Revenue, a non-GAAP
    # financial measure" as plain "net revenues".
    "non-GAAP/organic/adjusted": re.compile(r"(?i)\bnon[- ]?gaap\b|\badjusted\b|\borganic\b"),
    "noncontrolling interests": re.compile(r"(?i)\bnon[- ]?controlling\b|\bminority interests?\b"),
    "constant currency": re.compile(r"(?i)\bconstant[- ]currency\b|\bcurrency[- ]neutral\b"),
    "pro forma": re.compile(r"(?i)\bpro[- ]forma\b"),
}
# A claim may narrow its scope only with words its sources use.
_CLAIM_ONLY_QUALIFIERS = dict(_QUALIFIERS, excluding=re.compile(r"(?i)\bexclud(?:ing|es|ed)\b|\bex-[a-z]"))
_FIGURE_TOKEN = re.compile(r"(?<![0-9.,])[0-9][0-9,]*(?:\.[0-9]+)?")
_CLAUSE_BREAK = re.compile(r"[.;!?](?:\s|$)|\n")
_CONNECTOR_WORDS = frozenset({"from", "with", "million", "millions", "billion", "billions", "thousand", "percent",
                              "points", "basis", "year", "years", "fiscal", "were", "was", "and", "respectively"})


def _figure_contexts(text: str) -> dict[str, set[str]]:
    """Qualifier families that govern each figure in ``text``.

    A figure's context is its own clause up to the previous figure. When only
    connector words separate two figures ("increased 14.7% to $35.6 billion"),
    the second inherits the first's context. In a table row the label cell is
    the context of every value in the row.
    """
    contexts: dict[str, set[str]] = {}
    label = text.split("|", 1)[0] if "|" in text else None
    previous_end, inherited = 0, set()
    for match in _FIGURE_TOKEN.finditer(text):
        if label is not None:
            window = label
        else:
            window = text[previous_end:match.start()]
            breaks = list(_CLAUSE_BREAK.finditer(window))
            if breaks:
                window, inherited = window[breaks[-1].end():], set()
        found = {name for name, pattern in _QUALIFIERS.items() if pattern.search(window)}
        words = [word for word in re.findall(r"[A-Za-z]{4,}", window) if word.lower() not in _CONNECTOR_WORDS]
        families = found | (inherited if not words else set())
        contexts.setdefault(match.group().replace(",", ""), set()).update(families)
        inherited, previous_end = families, match.end()
    return contexts


_YEAR_LIKE = re.compile(r"(?:19|20)[0-9]{2}")


def _claim_clause_contexts(text: str) -> dict[str, set[str]]:
    """Qualifier families named in the sentence before each claim figure.

    The claim side is deliberately lenient: "net earnings including
    noncontrolling interests in fiscal year 2025 amounted to $1,856.7 million"
    keeps its qualifier although a year sits between them (live D07), as does
    "Organic Net Revenue, a non-GAAP measure, increased ... to $35.6 billion".
    Only text before the figure counts, so "... to $35.6 billion in 2023,
    representing organic growth ..." does not lend 'organic' to $35.6 billion.
    """
    breaks = [match.end() for match in _CLAUSE_BREAK.finditer(text)]
    contexts: dict[str, set[str]] = {}
    for match in _FIGURE_TOKEN.finditer(text):
        start = max((end for end in breaks if end <= match.start()), default=0)
        clause = text[start:match.start()]
        contexts.setdefault(match.group().replace(",", ""), set()).update(
            name for name, pattern in _QUALIFIERS.items() if pattern.search(clause))
    return contexts


def _qualifier_mismatch(claim_text: str, quotes: list[str], cited_texts: list[str]) -> str | None:
    """Name a qualifier that a claim drops from, or adds to, its cited figures.

    The live audit released 'net revenues' for a quote about 'Organic Net
    Revenue, a non-GAAP financial measure', 'net earnings' for a row labelled
    'including noncontrolling interests', and 'excluding financing' for a
    consolidated total. Each figure the claim shares with a quote must carry
    the quote's qualifier family for that figure, and any qualifier the claim
    adds must appear somewhere in its cited text.
    """
    cited = " ".join(cited_texts)
    for name, pattern in _CLAIM_ONLY_QUALIFIERS.items():
        if pattern.search(claim_text) and not pattern.search(cited):
            return f"claim adds '{name}' that its sources do not state"
    claimed = _claim_clause_contexts(claim_text)
    for quote in quotes:
        for figure, families in _figure_contexts(quote).items():
            # Years and dates are periods, not qualified amounts ("On May 16, 2022" in live C01).
            if len(figure.replace(".", "")) < 3 or figure not in claimed or _YEAR_LIKE.fullmatch(figure):
                continue
            missing = families - claimed[figure]
            if missing:
                return f"claim drops '{sorted(missing)[0]}' that qualifies a figure it states"
    return None


def _calculation_words(claim: dict) -> list[str]:
    calculation = claim.get("calculation")
    if type(calculation) is not dict:
        return []
    return [operand["label"] for operand in calculation["operands"]] + [calculation["unit"]]


def _resolved(resolver: Callable[[str], AbstractSet[str]], text: str) -> set[str]:
    found = resolver(text) or ()
    found = (found,) if isinstance(found, str) else found
    return {item.upper() for item in found if isinstance(item, str)}


def check_claim_grounding(claims: list[dict], bundle: dict, *,
                          company_resolver: Callable[[str], AbstractSet[str]] | None = None) -> dict:
    """Reject claims whose own citations cannot contain what the claim states.

    Runs after ``validate_draft`` and before the semantic verifier, which in
    the live audit approved a historical value restated as a 2027 projection
    and a competitor's figures relabelled as another company's. These checks
    are deliberately literal: every year, calendar date, number and recognized
    company in a claim must occur in the claim's own cited sources (or be the
    result of a verified calculation), and a claim may not concede that
    support is missing. Rules run in order across all claims; the first
    failure wins.
    Returns ``{"ok", "reason", "claim_index", "detail"}``.
    """
    checked = validate_draft({"status": "answer", "claims": claims}, bundle)
    if not checked["ok"]:
        return _grounding_result(checked["reason"], None, "claims failed reference validation")
    claims = checked["claims"]
    texts = [_normalized(claim["text"]) for claim in claims]
    # Operand labels and units are model-written text shown in the answer, so
    # their years, companies and admissions are checked like the claim itself.
    described = [_normalized(" ".join([claim["text"], *_calculation_words(claim)])) for claim in claims]
    sources = [_claim_sources(claim, bundle) for claim in claims]

    # Details name positions only, never claim text: a rejected draft must not
    # leak into the audit trail that accompanies the public abstention.
    for index, text in enumerate(described):
        if _ADMITS_MISSING.search(text):
            return _grounding_result("claim_admits_missing_evidence", index,
                                     "claim states that its support is missing from the sources")

    for index, (text, source) in enumerate(zip(described, sources)):
        passages = [_normalized(passage) for passage in source["passage_texts"]]
        years = _years(text)
        for position, (year, _) in enumerate(years, 1):
            if year not in source["fiscal_years"] and not any(str(year) in passage for passage in passages):
                return _grounding_result("claim_period_not_in_evidence", index,
                                         f"year {position} of {len(years)} in the claim is neither a cited "
                                         "report's fiscal year nor present in the cited passages")
        dates = _dates(text)
        if dates:
            cited_dates = {date for passage in passages for date, _ in _dates(passage)}
            for position, (date, _) in enumerate(dates, 1):
                if date not in cited_dates:
                    return _grounding_result("claim_period_not_in_evidence", index,
                                             f"date {position} of {len(dates)} in the claim does not occur "
                                             "in the cited passages")

    # A figure may restate another claim's application-verified calculation result.
    computed = [(abs(Decimal(_result_text(claim["calculation"]["result"]))), None,
                 _dimensionless_unit(claim["calculation"]["unit"])) for claim in claims if "calculation" in claim]
    for index, (claim, text, source) in enumerate(zip(claims, texts, sources)):
        candidates = [number for unit in source["unit_texts"] for number in _source_numbers(_normalized(unit))]
        candidates += computed
        calculation = claim.get("calculation")
        if calculation is not None:
            candidates.append((abs(Decimal(_result_text(calculation["result"]))), None,
                               _dimensionless_unit(calculation["unit"])))
            candidates.extend((abs(_decimal_value(operand["value"])), None, "%" in operand["value"])
                              for operand in calculation["operands"])
        numbers = _claim_numbers(text)
        for position, (_, value, decimals, scale) in enumerate(numbers, 1):
            if not _number_grounded(value, decimals, scale, candidates):
                return _grounding_result("claim_number_not_in_evidence", index,
                                         f"number {position} of {len(numbers)} in the claim does not occur "
                                         "in the cited evidence at any scale or rounding")

    for index, (claim, source) in enumerate(zip(claims, sources)):
        quotes = [citation["quote"] for citation in claim["citations"]]
        mismatch = _qualifier_mismatch(" ".join([claim["text"], *_calculation_words(claim)]), quotes,
                                       source["unit_texts"])
        if mismatch:
            return _grounding_result("claim_qualifier_mismatch", index, mismatch)

    for index, claim in enumerate(claims):
        mismatch = _column_year_mismatch(claim, bundle)
        if mismatch:
            return _grounding_result("claim_column_year_mismatch", index, mismatch)
        footnoted = _footnoted_row(claim, bundle)
        if footnoted:
            return _grounding_result("claim_footnoted_row", index, footnoted)

    if company_resolver is not None:
        for index, (claim, source) in enumerate(zip(claims, sources)):
            try:
                named = _resolved(company_resolver, " ".join([claim["text"], *_calculation_words(claim)]))
                # A competitor or customer named inside the cited text itself is supported.
                mentioned = set().union(*(_resolved(company_resolver, unit) for unit in source["unit_texts"]))
            except Exception:  # noqa: BLE001 - an unusable resolver must not release the claim
                return _grounding_result("claim_entity_not_in_evidence", index, "company resolver failed")
            if named - source["tickers"] - mentioned:
                cited = ", ".join(sorted(source["tickers"])) or "none"
                return _grounding_result("claim_entity_not_in_evidence", index,
                                         f"claim names a company outside the cited sources (cited: {cited})")
    return _grounding_result()



_COMPARATIVE = r"(?:higher|larger|more|bigger|greater|lower|less|smaller)"
_ASKS_WHICH = re.compile(r"(?i)\b(?:which|whose|who)\b.*\b" + _COMPARATIVE + r"\b|\b(?:did|does|was|is|were)\b[^?]*\bor\b"
                         r"[^?]*\b" + _COMPARATIVE + r"\b")
_ASKS_LOWER = re.compile(r"(?i)\b(?:lower|less|smaller)\b")
_STATES_COMPARISON = re.compile(r"(?i)\b(?:higher|more than|larger|greater|exceed\w*|lower than|less than|outpac\w*|"
                                r"surpass\w*|bigger|smaller)\b")


def comparison_verdict(question: str, tickers: list[str], claims: list[dict], bundle: dict) -> dict | None:
    """A 'which is higher' result claim built from verified claims, or None.

    Generators often state both companies' amounts without saying which is
    higher (all 12 partially correct comparative benchmark answers). This
    fires only for a two-company question asking for the higher or lower one,
    when no claim states a comparison and each company has exactly one
    currency or scaled amount, attributed by its cited unit's ticker. The
    sentence follows the question's direction and cites the compared claims.
    """
    if (len(tickers) != 2 or not _ASKS_WHICH.search(question)
            or any(_STATES_COMPARISON.search(claim["text"]) for claim in claims)):
        return None
    evidence, found = bundle["evidence"], {ticker: {} for ticker in tickers}
    for index, claim in enumerate(claims):
        text = _normalized(claim["text"])
        cited = [evidence[citation["evidence_id"]] for citation in claim["citations"]]
        owners = {entry["citation"].get("ticker") for entry in cited}
        for raw, value, _, scale in _claim_numbers(text):
            at = text.find(raw)
            dollar = "$" in text[max(0, at - 2):at] or "usd" in text[at + len(raw):at + len(raw) + 12].lower()
            if at < 0 or _PERCENT_AFTER.match(text, at + len(raw)) or (not dollar and scale == 1):
                continue
            digits = re.search(r"[0-9][0-9,.]*[0-9]|[0-9]", raw).group()
            owner = owners if len(owners) == 1 else {entry["citation"].get("ticker") for entry in cited
                                                     if digits in entry["text"]}
            ticker = next(iter(owner)) if len(owner) == 1 else None
            if ticker in found:
                found[ticker][value * scale] = (index, ("$" if dollar else "") + raw.strip(), (dollar, scale > 1))
    if any(len(amounts) != 1 for amounts in found.values()):
        return None
    (high, high_ticker, high_index, high_text, high_mark), (low, low_ticker, low_index, low_text, low_mark) = sorted(
        ((amount, ticker, *entry) for ticker, amounts in found.items() for amount, entry in amounts.items()),
        key=lambda item: item[0], reverse=True)
    if high == low or high_mark != low_mark:
        return None
    names = {chunk.get("ticker"): chunk.get("company") for chunk in bundle.get("chunks") or ()}
    high_name, low_name = names.get(high_ticker) or high_ticker, names.get(low_ticker) or low_ticker
    text = (f"{low_name} reported the lower amount: {low_text}, against {high_text} for {high_name}."
            if _ASKS_LOWER.search(question) else
            f"{high_name} reported the higher amount: {high_text}, against {low_text} for {low_name}.")
    citations = list({(c["evidence_id"], c["quote"]): c for index in (high_index, low_index)
                      for c in claims[index]["citations"]}.values())
    return {"text": text, "citations": copy.deepcopy(citations)}

# --- Semantic verdict and rendering ------------------------------------------

def _semantic_failure(reason: str) -> dict:
    return {"ok": False, "reason": reason, "semantic_supported": False,
            "answers_question": False, "claim_verdicts": []}


# The live verifier wrote "not in the cited evidence but implied" and "a
# projection for 2025, not for 2027; however ... reasonably interpreted" while
# still returning supported=true. Its own doubt overrides the boolean. Words
# such as "assumed" are excluded: "including assumed debt" is ordinary finance.
_VERDICT_DOUBT = re.compile(
    r"(?i)\bnot (?:explicitly |directly )?(?:stated|mentioned|provided|shown|given|disclosed|supported)\b"
    r"|\bnot (?:in|present in|found in) the (?:cited |provided |supplied )?(?:evidence|sources?|excerpts?)\b"
    r"|\bimplied\b|\bimplicitly\b|\breasonably (?:be )?interpreted\b|\bcan be inferred\b"
    r"|\bdoes not (?:specify|state|mention|support)\b")


def validate_semantic_verdict(verdict: Any, claims: list[dict]) -> dict:
    """Require an independent semantic judgment for every claim and the query.

    The exact verifier schema is ``{'answers_question': bool,
    'claim_verdicts': [{'claim_index': int, 'supported': bool, 'reason': str}]}``.
    Indices are zero based, unique, and exhaustive. A true result reports a
    model judgment, not a deterministic entailment or numerical proof.
    """
    if type(claims) is not list or not 1 <= len(claims) <= MAX_CLAIMS:
        return _semantic_failure("invalid_claim_count")
    parsed = _object(verdict)
    if parsed is None:
        return _semantic_failure("invalid_verifier_json")
    if set(parsed) != {"answers_question", "claim_verdicts"} or type(parsed["answers_question"]) is not bool:
        return _semantic_failure("invalid_verifier_schema")
    judgments = parsed["claim_verdicts"]
    if type(judgments) is not list or len(judgments) != len(claims):
        return _semantic_failure("incomplete_verifier_coverage")
    by_index = {}
    for judgment in judgments:
        if (type(judgment) is not dict
                or set(judgment) != {"claim_index", "supported", "reason"}
                or not _integer(judgment["claim_index"])
                or type(judgment["supported"]) is not bool
                or not _text(judgment["reason"], MAX_REASON_CHARS)):
            return _semantic_failure("invalid_verifier_schema")
        index = judgment["claim_index"]
        if index >= len(claims) or index in by_index:
            return _semantic_failure("incomplete_verifier_coverage")
        by_index[index] = judgment
    if set(by_index) != set(range(len(claims))):
        return _semantic_failure("incomplete_verifier_coverage")
    if any(judgment["supported"] and _VERDICT_DOUBT.search(judgment["reason"]) for judgment in judgments):
        return _semantic_failure("verifier_reason_contradicts_verdict")
    if not parsed["answers_question"]:
        return _semantic_failure("question_not_answered")
    if not all(judgment["supported"] for judgment in judgments):
        return _semantic_failure("unsupported_claim")
    return {"ok": True, "reason": None, "semantic_supported": True,
            "answers_question": True,
            "claim_verdicts": [copy.deepcopy(by_index[index]) for index in range(len(claims))]}


def _escape(text: Any) -> str:
    """Render untrusted values as text, without Markdown, HTML, or autolinks.

    Entity-encoded punctuation is decoded *after* CommonMark structure is
    recognized. In contrast, backslash escapes alone do not block bare URL
    autolinks in all Markdown renderers. Control characters are not displayed.
    """
    output = []
    for char in str(text):
        if char in string.punctuation:
            output.append(f"&#{ord(char)};")
        elif char in "\r\n\t":
            output.append(" ")
        elif unicodedata.category(char).startswith("C"):
            output.append("\ufffd")
        else:
            output.append(char)
    return "".join(output)


def _locator_label(entry: dict) -> str:
    locator, kind = entry["source_locator"], entry["kind"]
    if kind == "prose":
        return (f"paragraph {locator['element_id']}, characters "
                f"{locator['char_start']}–{locator['char_end']} (end exclusive)")
    if kind == "table_source":
        return (f"table {locator['table_id']} source text, characters "
                f"{locator['char_start']}–{locator['char_end']} (end exclusive)")
    if kind == "row":
        return f"table {locator['table_id']}, source row index {locator['row_index']} (zero based)"
    extra = f", context index {locator['context_index']} (zero based)" if "context_index" in locator else ""
    return f"table {locator['table_id']}, {locator['role']}{extra}"


def _result_label(calculation: dict) -> str:
    result, unit = calculation["result"].replace("-", "\u2212"), calculation["unit"].strip()
    if not unit:
        return result
    return f"{result}{unit}" if unit in ("%", "x") else f"{result} {unit}"


def _calculation_line(calculation: dict, display: str, markers: dict[str, str]) -> str:
    """One escaped line that shows the application-computed arithmetic and operands."""
    parts = [_escape(f"Calculated by the application: {display} = {_result_label(calculation)} (operands: ")]
    for position, operand in enumerate(calculation["operands"]):
        separator = "; " if position else ""
        parts.append(_escape(f"{separator}{operand['name']} = {operand['label']}, {operand['value']}")
                     + f" {markers[operand['evidence_id']]}")
    parts.append(_escape(")"))
    return "".join(parts)


def render_verified_answer(claims: list[dict], bundle: dict, semantic_verdict: Any) -> dict:
    """Render only when deterministic and independent semantic gates both pass.

    Page numbers and paragraph/table locators always come from the validated
    evidence bundle, never generator-provided metadata. Structured citations
    retain the full source unit and exact quote offsets for a citation UI.
    Calculated claims are followed by the recomputed expression and operands.
    """
    checked = validate_draft({"status": "answer", "claims": claims}, bundle)
    semantic = (validate_semantic_verdict(semantic_verdict, checked["claims"])
                if checked["ok"] else _semantic_failure("reference_verification_failed"))
    if not checked["ok"] or not semantic["ok"]:
        return {"status": "abstain", "answer": ABSTENTION, "citations": [], "calculations": [],
                "reference_verified": checked["reference_verified"],
                "semantic_supported": False, "reason": checked["reason"] or semantic["reason"]}
    references, numbering, paragraphs, calculations = [], {}, [], []
    for claim_index, claim in enumerate(checked["claims"]):
        markers, by_evidence = [], {}
        for citation in claim["citations"]:
            identifier, quote = citation["evidence_id"], citation["quote"]
            key = (identifier, quote)
            if key not in numbering:
                number = len(references) + 1
                numbering[key] = number
                entry = bundle["evidence"][identifier]
                start = entry["text"].index(quote)
                references.append({
                    "reference_number": number, "evidence_id": identifier,
                    "quote": quote, "quote_char_start": start,
                    "quote_char_end": start + len(quote), "source_text": entry["text"],
                    "kind": entry["kind"], "citation": copy.deepcopy(entry["citation"]),
                    "source_locator": copy.deepcopy(entry["source_locator"]),
                    "chunk_ids": list(entry["chunk_ids"]),
                })
            markers.append(f"[^{numbering[key]}]")
            by_evidence[identifier] = markers[-1]
        paragraphs.append(f"{_escape(claim['text'])} {''.join(markers)}")
        if "calculation" in claim:
            calculation = claim["calculation"]
            display = verify_calculation(claim, bundle["evidence"])["expression_display"]
            paragraphs.append(_calculation_line(calculation, display, by_evidence))
            calculations.append({"claim_index": claim_index, "expression": calculation["expression"],
                                 "expression_display": display,
                                 "operands": copy.deepcopy(calculation["operands"]),
                                 "result": calculation["result"], "unit": calculation["unit"]})
    footnotes = []
    for reference in references:
        entry = bundle["evidence"][reference["evidence_id"]]
        citation = entry["citation"]
        label = (f"{citation['document_id']}, PDF page {citation['pdf_page']}; "
                 f"{_locator_label(entry)}. Evidence {reference['evidence_id']}.")
        footnotes.append(f"[^{reference['reference_number']}]: {_escape(label)}\n"
                         f"    > {_escape(reference['quote'])}")
    return {"status": "answer", "answer": "\n\n".join(paragraphs + footnotes),
            "citations": references, "calculations": calculations, "reference_verified": True,
            "semantic_supported": True, "reason": None}
