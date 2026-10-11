"""Wording and layout rules for the Streamlit demo, kept free of Streamlit.

Reason texts, the refusal-stage map, the verification checklist, the answer
layout, the scope summary and the corpus overview are plain functions here, so
they are unit-tested without the UI and streamlit_app.py stays a layout file.
Every value that came from a model or a user is escaped before it is shown as
Markdown; passage and quote text is shown with st.text by the app instead.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from report_search.answer_verification import escape_markdown, result_label

REPO_URL = "https://github.com/GoBackwards/finance_qa_system"

CORPUS_LABELS = {"v1": "Corpus v1 · original parse (Tables A1–A13)",
                 "v2": "Corpus v2 · table headers read in column order (Table A14)"}

# Written for the demo; none reuses a benchmark or live-pool question or its answer.
EXAMPLES = (
    ("Look up a figure", ":material/search:", (
        "How much did Walmart spend on capital expenditures in fiscal 2024?",
        "how much did coke pay in dividends last year?",
        "英伟达2024财年的研发费用是多少？",
    )),
    ("Compare or calculate", ":material/compare_arrows:", (
        "Which had higher net sales in fiscal 2024, Target or Lowe's?",
        "Which company reported higher revenue in fiscal 2024, Adobe or Intuit?",
        "By what percentage did Kimberly-Clark's net sales change from 2023 to 2024?",
    )),
    ("See it refuse", ":material/block:", (
        "What were Netflix's paid memberships at the end of 2024?",
        "What will Costco's net sales be in fiscal 2026?",
    )),
)

STEPS = ("Evidence strong enough",
         "Sources re-read from the parsed reports",
         "Draft written from those sources",
         "Quotes and calculations checked by code",
         "Figures, years and companies traced to their quotes",
         "Independent review by a separate model call")

REASONS = {
    "model_not_configured": "Connect an answer model in the sidebar to generate a cited answer.",
    "invalid_model_configuration": "Check the answer model settings in the sidebar.",
    "unconfigured_threshold": "Set an explicit minimum evidence score in the sidebar before generating an answer.",
    "invalid_threshold": "The minimum evidence score must be a finite number.",
    "invalid_query": "Enter a valid question of up to 10,000 characters.",
    "below_threshold": "The retrieved evidence did not meet the configured confidence threshold.",
    "deadline_exceeded": "Answer preparation exceeded the configured time budget.",
    "unsupported_claim": "The proposed answer was not fully supported by the retrieved evidence.",
    "question_not_answered": "The evidence did not support a complete answer to this question.",
    "generator_abstained": "The answer model could not fully answer from the supplied evidence.",
    "source_verification_failed": "The source integrity check failed. Review the local source artifacts.",
    "context_budget_exceeded": "The selected evidence exceeded the answer context budget even after dropping the passages that could be removed.",
    "quote_not_in_source": "A supporting quote could not be matched to its source passage.",
    "claim_period_not_in_evidence": "The proposed answer named a period that its sources do not cover.",
    "claim_number_not_in_evidence": "The proposed answer contained a number that its sources do not state.",
    "claim_entity_not_in_evidence": "The proposed answer named a company that its sources do not cover.",
    "claim_admits_missing_evidence": "The retrieved evidence did not cover every part of the question.",
    "calculation_result_mismatch": "A calculated value did not match the application's recomputation.",
    "calculation_operand_not_in_source": "A calculation used a value that its source does not state.",
    "calculation_operand_not_cited": "A calculation used a value without citing its source.",
    "calculation_result_not_in_claim": "A calculated value was not stated in the answer it supports.",
    "calculation_direction_mismatch": "A calculated change was described in the wrong direction.",
    "invalid_calculation_expression": "A calculation could not be checked by the application.",
    "invalid_calculation_schema": "A calculation could not be checked by the application.",
    "claim_qualifier_mismatch": "The proposed answer narrowed or widened a figure's scope beyond its source.",
    "verifier_reason_contradicts_verdict": "The independent review doubted a claim it marked as supported.",
    "claim_column_year_mismatch": "The proposed answer took a figure from another fiscal year's column.",
    "claim_footnoted_row": "The proposed answer used a footnoted figure where the plain figure was available.",
    "too_many_passages": "More passages were selected than one answer may use.",
    "answer_unavailable": "The answer service did not return a result. No unverified answer was released.",
}
GENERIC_REASON = "A required answer check failed. No unverified answer was released."

STAGE_DEFAULTS = {
    "setup": "Answer settings are incomplete.",
    "evidence": "The retrieved passages did not score high enough to answer from.",
    "sources": "The passages could not be matched back to the parsed reports.",
    "draft": "The answer model did not return a usable draft.",
    "quotes": "A quote, citation or calculation in the draft did not check out.",
    "grounding": "A figure, year or company in the draft is not backed by its quote.",
    "review": "The independent review did not confirm every claim.",
    "deadline": REASONS["deadline_exceeded"],
}
HEADLINES = {
    "setup": "Setup needed: no answer was attempted",
    "evidence": "The evidence was too weak",
    "sources": "The source check failed",
    "draft": "The answer model did not produce a usable answer",
    "quotes": "A quote or calculation did not check out",
    "grounding": "A figure, year or company was not backed by its quote",
    "review": "The independent review did not confirm the answer",
    "deadline": "The answer ran out of time",
    "service": "The answer service was unavailable",
}
# Refusals where a check worked as intended get a headline that says so, and their step is
# drawn as a deliberate stop rather than a failure.
CODE_HEADLINES = {
    "generator_abstained": "The sources do not answer this, so the model declined",
    "claim_admits_missing_evidence": "The evidence does not cover every part of the question",
    "question_not_answered": "The review found the question only partly answered",
}
STOP_DETAILS = {
    "generator_abstained": "the model declined: the sources do not answer this",
    "claim_admits_missing_evidence": "a claim said the evidence does not cover part of the question",
    "question_not_answered": "the review found the question only partly answered",
}
GATE_OPEN_BELOW = -100  # a minimum below any re-ranker score: the gate passes everything (evaluation setting)
NEXT_STEPS = {
    "setup": "Open the sidebar: connect a model and enter a minimum evidence score. Nothing is pre-filled on purpose.",
    "sources": "The local source files do not match the index; rebuild the corpus folders.",
    "model": "Check the endpoint, model name, API key and token parameter in the sidebar.",
    "draft": "Ask for the figure as the report states it, name the fiscal year, or split a multi-part question "
             "into one company, one metric, one year.",
    "deadline": "Raise the answer time limit in the sidebar and try again.",
    "service": "Try again. If it keeps failing, check the terminal running the app.",
}
for _stage in ("quotes", "grounding", "review"):
    NEXT_STEPS[_stage] = NEXT_STEPS["draft"]

_STAGE_CODES = {
    "setup": {"unconfigured_threshold", "invalid_threshold", "model_not_configured",
              "invalid_model_configuration", "invalid_query"},
    "evidence": {"below_threshold", "low_confidence", "empty_candidates", "missing_reranker_score",
                 "invalid_reranker_score", "invalid_candidates", "too_many_passages"},
    "sources": {"source_verification_failed", "context_budget_exceeded"},
    "draft": {"generator_abstained", "generation_failed", "invalid_draft_json", "invalid_draft_schema"},
    "quotes": {"quote_not_in_source", "invalid_quote", "unknown_evidence_id", "duplicate_citation",
               "invalid_citation_schema", "invalid_claim_schema", "invalid_claim_text", "invalid_claim_count",
               "missing_or_excess_citations", "inline_citation_in_claim", "invalid_source_binding",
               "unverified_evidence_bundle", "invalid_calculation_expression", "invalid_calculation_schema"},
    "review": {"unsupported_claim", "question_not_answered", "verifier_reason_contradicts_verdict",
               "invalid_verifier_json", "invalid_verifier_schema", "incomplete_verifier_coverage",
               "semantic_verifier_failed", "verification_context_budget_exceeded", "unsupported_answer",
               "reference_verification_failed"},
}
_MODEL_ERRORS = {"model_unavailable", "empty_model_response", "incomplete_model_response", "invalid_model_choices",
                 "invalid_model_json", "invalid_model_request", "invalid_model_response_framing",
                 "model_refusal_or_tool_call", "model_response_too_large"}
STEP_OF_STAGE = {"evidence": 1, "sources": 2, "draft": 3, "quotes": 4, "grounding": 5, "review": 6}
_TIMED_STEPS = {"source_binding": 2, "generation": 3, "generation_retry": 3, "reference_validation": 4,
                "reference_revalidation": 4, "semantic_verification": 6, "verdict_validation": 6, "rendering": 6}
CALCULATION_PREFIX = escape_markdown("Calculated by the application:")

CONTEXT_LABELS = {
    "row_label_table": "financial table whose row labels match the question",
    "question_bm25": "keyword match for the question",
    "summary_theme": "keyword match for one theme of the question",
    "summary_neighbour": "paragraph next to a selected passage",
    "same_table": "continuation of a selected table",
    "title_table": "table that follows a selected passage",
}
KIND_LABELS = {"prose": "Text", "table": "Table", "table_fallback": "Table (layout inferred)",
               "page_fallback": "Page text"}
CITATION_KIND_LABELS = {"prose": "Paragraph", "row": "Table row", "table_source": "Table text",
                        "table_context": "Table heading or caption"}
FORECAST_NOTE = ("This asks for a forecast. Annual reports contain historical results, so the app will not "
                 "estimate future figures.")

SEARCH_ERROR = ("Search could not complete. Check the local models, indexes, and source artifacts; "
                "no answer was generated.")


def build_advice(corpus: str) -> str:
    """How to get the selected corpus built; v1 cannot be rebuilt with this checkout's parser."""
    if corpus == "v1":
        return ("Corpus v1 cannot be rebuilt with this checkout (it needs the parser from commit 435b7ae, "
                "Table A14); unset FINANCE_QA_CORPUS to use v2, which `python scripts/build_corpus.py` builds.")
    return "Build it with `python scripts/build_corpus.py`, which also fetches the search models."


def corpus_missing_text(corpus: str) -> str:
    return f"Corpus {corpus} is not built in this checkout. " + build_advice(corpus)


def search_error_text(code: str, corpus: str) -> str:
    """A fixed sentence per known search failure; unknown codes are never echoed."""
    if code == "invalid_query":
        return REASONS["invalid_query"]
    if code == "startup_failed":
        return (f"Search could not complete: the search models or the corpus {corpus} indexes could not be "
                "loaded. " + build_advice(corpus))
    if code == "invalid_corpus":
        return "Search could not complete: FINANCE_QA_CORPUS must be v1 or v2."
    return SEARCH_ERROR


def refusal_stage(code: str, audit: dict | None = None) -> str:
    """The step at which an answer stopped, from its reason code (setup and service have no step)."""
    audit = audit or {}
    for stage in ("setup", "evidence", "sources", "draft", "quotes", "review"):
        if code in _STAGE_CODES[stage]:
            return stage
    if code.startswith("calculation_"):
        return "quotes"
    if code.startswith("claim_"):
        return "grounding"
    if code in _MODEL_ERRORS or code.startswith("model_http_"):
        return "review" if audit.get("semantic_verifier_called") else "draft"
    if code == "deadline_exceeded":
        return "deadline"
    return "service"


def reason_text(code: str, audit: dict | None = None) -> str:
    if code in REASONS:
        return REASONS[code]
    match = re.fullmatch(r"model_http_(\d{3})", code or "")
    if match:
        return f"The answer model's service returned an error (HTTP {match.group(1)})."
    if code in _MODEL_ERRORS:
        return "The answer model could not be reached or returned an unusable response."
    return STAGE_DEFAULTS.get(refusal_stage(code, audit), GENERIC_REASON)


def headline(code: str, audit: dict | None = None) -> str:
    return CODE_HEADLINES.get(code) or HEADLINES[refusal_stage(code, audit)]


def retryable(code: str, audit: dict | None = None) -> bool:
    """Refusals that a settings change can fix, so the same evidence may be answered again."""
    return (refusal_stage(code, audit) in ("setup", "deadline") or code in _MODEL_ERRORS
            or code.startswith("model_http_"))


def _gate(audit: dict) -> tuple[dict, bool]:
    """The gate that decided: the re-check on the passages actually sent when it refused, else the first."""
    packed = audit.get("packed_confidence_gate") or {}
    if packed.get("allowed") is False:
        return packed, True
    return audit.get("confidence_gate") or {}, False


def next_step(code: str, audit: dict | None = None, scope: dict | None = None) -> str:
    audit, scope = audit or {}, scope or {}
    stage = refusal_stage(code, audit)
    if stage not in ("setup", "service", "deadline") and code not in _MODEL_ERRORS and not code.startswith("model_http_"):
        if scope.get("unknown_companies"):
            return ("That company is not one of the 40 covered here (see “Which companies are covered?”). "
                    "Ask about a covered company instead.")
        if scope.get("forecast_intent"):
            return "Ask about results the reports state for fiscal years 2021–2025, not projections."
        if scope.get("out_of_coverage_years"):
            return "Ask about fiscal years 2021–2025, which the reports cover."
    if code in _MODEL_ERRORS or code.startswith("model_http_"):
        return NEXT_STEPS["model"]
    if code == "context_budget_exceeded":
        return "Ask about fewer companies or years."
    if stage == "evidence":
        gate, packed = _gate(audit)
        if _number(gate.get("max_score")) and _number(gate.get("threshold")):
            where = " after leaving out passages to fit the context" if packed else ""
            return (f"The best passage scored {gate['max_score']:.2f}{where}, below your minimum of "
                    f"{format_threshold(gate['threshold'])}. "
                    "Name the company and fiscal year, use the report's wording for the line item, or ask for "
                    "one figure at a time.")
        return "Name the company and fiscal year, or use the report's wording for the line item."
    return NEXT_STEPS.get(stage, NEXT_STEPS["service"])


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_threshold(value: float) -> str:
    """'6.5', or '−1,000,000 (gate open, as in the evaluations)' for the evaluation setting."""
    text = f"{value:,.10g}".replace("-", "−")
    return text + " (gate open, as in the evaluations)" if value < GATE_OPEN_BELOW else text


def _model_name(audit: dict, role: str) -> str:
    model = (audit.get(role) or {}).get("model")
    return escape_markdown(model) if isinstance(model, str) and model else "not recorded"


def _confirmed_steps(answer: dict) -> list[tuple[bool | str, str]]:
    """(confirmed by the audit, or "open" for an open gate; detail) for each of the six steps."""
    audit = answer.get("audit") or {}
    gate, packed = _gate(audit)
    detail = ""
    if _number(gate.get("max_score")) and _number(gate.get("threshold")):
        detail = f"best passage scored {gate['max_score']:.2f}; your minimum is {format_threshold(gate['threshold'])}"
        if audit.get("packed_confidence_gate"):
            detail += ("; re-checked on the passages actually sent" if not packed
                       else " after leaving out passages to fit the context")
    gate_open = _number(gate.get("threshold")) and gate["threshold"] < GATE_OPEN_BELOW
    if gate.get("allowed") is True and gate_open:
        steps = [("open", "not checked: the minimum evidence score is open (evaluation setting), so every score "
                          f"passes; best passage scored {gate['max_score']:.2f}" if _number(gate.get("max_score"))
                  else "not checked: the minimum evidence score is open (evaluation setting)")]
    else:
        steps = [(gate.get("allowed") is True, detail)]

    detail = ""
    if isinstance(audit.get("evidence_count"), int) and isinstance(audit.get("context_passages"), int):
        detail = f"{audit['evidence_count']} source units from {audit['context_passages']} passages"
        if audit.get("context_passages_dropped"):
            detail += f"; {audit['context_passages_dropped']} lowest-ranked left out to fit"
    steps.append((audit.get("source_verified") is True, detail))

    detail = f"model {_model_name(audit, 'generator')}" if audit.get("generator") else ""
    if audit.get("generation_attempts") == 2:
        first = audit.get("first_attempt_reason")
        reason = escape_markdown(reason_text(first, audit).rstrip(".")) if isinstance(first, str) else "a check failed"
        detail += (f"; the first draft failed a check ({reason}) and was corrected once; the new draft went "
                   "through every check again")
    steps.append((audit.get("generation_called") is True, detail))

    citations, calculations = answer.get("citations") or [], answer.get("calculations") or []
    detail = ""
    if answer.get("status") == "answer":
        detail = f"{len(citations)} quotes found word for word · {len(calculations)} calculations recomputed"
        if audit.get("citation_repairs"):
            detail += f" · {len(audit['citation_repairs'])} citations re-pointed to the exact source text"
    steps.append((audit.get("reference_check_passed") is True, detail))

    grounding = audit.get("claim_grounding") or {}
    steps.append((grounding.get("ok") is True,
                  "every figure, year and company appears in its quote" if grounding.get("ok") is True else ""))

    reviewed = audit.get("semantic_verifier_called") is True and answer.get("status") == "answer"
    steps.append((reviewed, f"reviewer {_model_name(audit, 'verifier')}" if audit.get("verifier") else ""))
    return steps


def _state(ok) -> str:
    return "open" if ok == "open" else "passed" if ok is True else "not_recorded"


def check_steps(answer: dict) -> list[tuple[str, str, str]]:
    """(label, state, detail) per step; state is passed, open, failed, stopped, not_run or not_recorded.

    A step shows as passed only when an audit field confirms it; an open
    evidence gate is shown as open, never as passed. A refusal marks its own
    step failed (or stopped, when a check worked as intended); later steps show
    as passed only if they ran, else not run. Setup and service refusals, which
    never reached the checks, return no steps.
    """
    confirmed = _confirmed_steps(answer)
    if answer.get("status") == "answer":
        return [(label, _state(ok), detail) for label, (ok, detail) in zip(STEPS, confirmed)]
    audit = answer.get("audit") or {}
    code = answer.get("reason") or "answer_unavailable"
    stage = refusal_stage(code, audit)
    if stage in ("setup", "service"):
        return []
    failed = STEP_OF_STAGE.get(stage)
    if _gate(audit)[1]:
        failed = 1
    if stage == "deadline":
        failed = next((_TIMED_STEPS.get(entry.get("stage")) for entry in audit.get("stage_timings") or ()
                       if isinstance(entry, dict) and entry.get("status") != "completed"), None)
    rows = []
    for number, (label, (ok, detail)) in enumerate(zip(STEPS, confirmed), 1):
        if failed is not None and number == failed:
            if code in STOP_DETAILS:
                rows.append((label, "stopped", STOP_DETAILS[code]))
            else:
                rows.append((label, "failed", "stopped here: " + escape_markdown(reason_text(code, audit))))
        elif failed is not None and number > failed:
            rows.append((label, "passed", detail) if ok is True else (label, "not_run", ""))
        else:
            rows.append((label, _state(ok), detail))
    return rows


def answer_parts(answer: dict) -> list[dict] | None:
    """Claims of a verified answer, each with its source numbers and recomputed calculation.

    Footnote definitions are dropped (the source cards show them). Returns None
    when the rendered text does not have the expected shape, and the app then
    shows the answer string unchanged.
    """
    text, calculations = answer.get("answer"), answer.get("calculations") or []
    if not isinstance(text, str) or not text.strip():
        return None
    claims, lines = [], 0
    for part in text.split("\n\n"):
        if part.startswith("[^"):
            continue
        if part.startswith(CALCULATION_PREFIX):
            if not claims or lines >= len(calculations):
                return None
            calculation = calculations[lines]
            if calculation.get("claim_index") != len(claims) - 1:
                return None
            claims[-1]["calculation"] = calculation
            lines += 1
            continue
        refs = [int(number) for number in re.findall(r"\[\^(\d+)\]", part)]
        claims.append({"text": re.sub(r"\s*\[\^\d+\]", "", part).strip(), "refs": refs, "calculation": None,
                       "computed": False})
    if lines != len(calculations) or not claims:
        return None
    if (answer.get("audit") or {}).get("comparison_verdict_added"):
        claims[-1]["computed"] = True
    return claims


def calculation_lines(calculation: dict, citations: list[dict]) -> tuple[str, list[str]]:
    """Escaped expression line and operand lines for a recomputed calculation."""
    references = {}
    for entry in citations:
        references.setdefault(entry.get("evidence_id"), entry.get("reference_number"))
    expression = escape_markdown(f"{calculation.get('expression_display', '')} = {result_label(calculation)}")
    operands = []
    for operand in calculation.get("operands") or ():
        line = escape_markdown(f"{operand.get('name')} = {operand.get('label')}: {operand.get('value')}")
        reference = references.get(operand.get("evidence_id"))
        operands.append(line + (f" [{reference}]" if reference else ""))
    return expression, operands


def corpus_overview(root) -> dict | None:
    """Reports, companies, years, pages and sectors from the committed data/manifest.json."""
    try:
        rows = json.loads((Path(root) / "data/manifest.json").read_text(encoding="utf-8-sig"))
        rows = [row for row in rows if row.get("status") == "validated"]
        names = {row["ticker"]: row["company"] for row in rows}
        sectors: dict[str, list[tuple[str, str]]] = {}
        for ticker, company in sorted(names.items(), key=lambda item: item[1].lower()):
            sector = next(row.get("sector") or "Other" for row in rows if row["ticker"] == ticker)
            sectors.setdefault(sector, []).append((company, ticker))
        years = [int(row["fiscal_year"]) for row in rows]
        return {"reports": len(rows), "companies": len(names), "years": (min(years), max(years)),
                "pages": sum(int(row["page_count"]) for row in rows), "sectors": dict(sorted(sectors.items())),
                "names": names}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _years(years) -> str:
    years = sorted({int(year) for year in years})
    if len(years) > 2 and years == list(range(years[0], years[-1] + 1)):
        return f"FY{years[0]}–FY{years[-1]}"
    return ", ".join(f"FY{year}" for year in years)


def scope_summary(scope: dict | None, trace: dict | None, names: dict, report_count: int) -> dict:
    """How the question was read: companies, period, what was searched, badges and escaped notes."""
    scope, trace = scope or {}, trace or {}
    tickers = [ticker for ticker in scope.get("tickers") or () if isinstance(ticker, str)]
    companies = " · ".join(escape_markdown(f"{names.get(ticker, ticker)} ({ticker})") for ticker in tickers)
    mode = scope.get("period_mode")
    if not scope:
        period = "Scope not resolved"
    elif mode == "explicit" and scope.get("years"):
        period = _years(scope["years"])
    elif mode == "relative_latest":
        latest = scope.get("latest_year") or {}
        period = "Latest report: " + (", ".join(f"{names.get(t, t)} FY{y}" for t, y in latest.items()) or "—")
    else:
        period = "Year not stated: latest reports favoured"
    pools = [pool for pool in trace.get("pools") or () if isinstance(pool, dict)]
    if trace.get("mode") == "scoped" and pools:
        searched = " · ".join(
            (f"+ latest {pool.get('ticker')} " if pool.get("purpose") == "latest" else f"{pool.get('ticker')} ")
            + _years(pool.get("fiscal_years") or ()) for pool in pools)
    elif trace.get("mode") == "global":
        searched = f"all {report_count} reports"
    else:
        searched = "not recorded"
    badges = []
    if scope.get("comparison"):
        badges.append(("Comparison", "blue"))
    if scope.get("forecast_intent"):
        badges.append(("Forecast question", "orange"))
    badges += [(f"Not covered: {name}", "red") for name in scope.get("unknown_companies") or ()
               if isinstance(name, str)]
    badges += [(f"Not covered: FY{year}", "orange") for year in scope.get("out_of_coverage_years") or ()
               if isinstance(year, int)]
    notes = [FORECAST_NOTE if note.startswith("The question asks about future or projected results")
             else escape_markdown(note) for note in scope.get("notes") or () if isinstance(note, str)]
    warn = bool(scope.get("unknown_companies") or scope.get("out_of_coverage_years") or scope.get("forecast_intent"))
    return {"companies": companies or "No company recognised", "period": escape_markdown(period),
            "searched": "Searched: " + escape_markdown(searched), "badges": badges, "notes": notes, "warn": warn}


def evidence_breakdown(selected: list[dict], trace: dict | None) -> tuple[int, int, str]:
    """Best matches, added context passages and the breakdown caption."""
    best = sum(1 for chunk in selected if not chunk.get("context_reason"))
    added = len(selected) - best
    caption = f"{best} best {'match' if best == 1 else 'matches'} + {added} added for context"
    skipped = len((trace or {}).get("suppressed_duplicates") or ())
    if skipped:
        caption += f" · {skipped} near-duplicates from other report years skipped"
    return best, added, caption
