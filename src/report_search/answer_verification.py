"""Fail-closed reference checks and answer rendering for grounded generation.

An exact source quote proves a reference exists; it does not prove that a
claim follows from that quote. Callers must obtain the semantic verdict in a
separate model request. This module validates that verdict's complete schema
and coverage, but cannot turn a model judgment into a guarantee of truth.
"""

from __future__ import annotations

import copy
import json
import re
import string
import unicodedata
from typing import Any


ABSTENTION = "I'm not sure"
MAX_JSON_CHARS = 100_000
MAX_CLAIMS = 20
MAX_CLAIM_CHARS = 2_000
MAX_CITATIONS_PER_CLAIM = 8
MAX_QUOTE_CHARS = 8_000
MAX_REASON_CHARS = 2_000
_EVIDENCE_ID = re.compile(r"E[1-9][0-9]*\Z")
_INLINE_REFERENCE = re.compile(r"\[(?:\^?[0-9]+|E[0-9]+)\]|【[^】]*】")
_KINDS = frozenset({"prose", "row", "table_source", "table_context"})


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


def _draft_failure(reason: str) -> dict:
    return {"ok": False, "status": "abstain", "reason": reason,
            "reference_verified": False, "claims": []}


def validate_draft(draft: Any, bundle: dict) -> dict:
    """Validate the generator schema and bind every quote to a known source.

    ``draft`` is either strict JSON text or a JSON object with exactly
    ``status`` and ``claims``. Every claim has exactly ``text`` and
    ``citations``; every citation has exactly ``evidence_id`` and ``quote``.
    Quotation matching is case-, whitespace-, and punctuation-sensitive.
    A failed result never returns any unverified draft text.
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
        if type(claim) is not dict or set(claim) != {"text", "citations"}:
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
    return {"ok": True, "status": "answer", "reason": None,
            "reference_verified": True, "claims": copy.deepcopy(claims)}


def _semantic_failure(reason: str) -> dict:
    return {"ok": False, "reason": reason, "semantic_supported": False,
            "answers_question": False, "claim_verdicts": []}


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


def render_verified_answer(claims: list[dict], bundle: dict, semantic_verdict: Any) -> dict:
    """Render only when deterministic and independent semantic gates both pass.

    Page numbers and paragraph/table locators always come from the validated
    evidence bundle, never generator-provided metadata. Structured citations
    retain the full source unit and exact quote offsets for a citation UI.
    """
    checked = validate_draft({"status": "answer", "claims": claims}, bundle)
    semantic = (validate_semantic_verdict(semantic_verdict, checked["claims"])
                if checked["ok"] else _semantic_failure("reference_verification_failed"))
    if not checked["ok"] or not semantic["ok"]:
        return {"status": "abstain", "answer": ABSTENTION, "citations": [],
                "reference_verified": checked["reference_verified"],
                "semantic_supported": False, "reason": checked["reason"] or semantic["reason"]}
    references, numbering, paragraphs = [], {}, []
    for claim in checked["claims"]:
        markers = []
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
        paragraphs.append(f"{_escape(claim['text'])} {''.join(markers)}")
    footnotes = []
    for reference in references:
        entry = bundle["evidence"][reference["evidence_id"]]
        citation = entry["citation"]
        label = (f"{citation['document_id']}, PDF page {citation['pdf_page']}; "
                 f"{_locator_label(entry)}. Evidence {reference['evidence_id']}.")
        footnotes.append(f"[^{reference['reference_number']}]: {_escape(label)}\n"
                         f"    > {_escape(reference['quote'])}")
    return {"status": "answer", "answer": "\n\n".join(paragraphs + footnotes),
            "citations": references, "reference_verified": True,
            "semantic_supported": True, "reason": None}
