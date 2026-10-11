"""Run with: python -m streamlit run streamlit_app.py."""
from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from report_search.demo_service import DemoError, DemoService
from report_search.demo_sources import SourceCatalog, SourceCatalogError
from report_search.llm_client import ModelConfig


REASONS = {
    "model_not_configured": "Connect an answer model in the sidebar to generate a cited answer.",
    "invalid_model_configuration": "Check the answer model settings in the sidebar.",
    "unconfigured_threshold": "Set an explicit minimum evidence score in the sidebar before generating an answer.",
    "invalid_threshold": "The minimum evidence score must be a finite number.",
    "invalid_query": "Enter a valid question of up to 10,000 characters.",
    "below_threshold": "The retrieved evidence did not meet the configured confidence threshold.",
    "low_confidence": "The retrieved evidence did not meet the configured confidence threshold.",
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
}

PERIOD_LABELS = {"relative_latest": "latest report", "unspecified": "not stated"}


@st.cache_resource(show_spinner=False)
def get_service(root: str, device: str):
    service = DemoService(Path(root), device=device, corpus=os.getenv("FINANCE_QA_CORPUS", "v1"))
    atexit.register(service.close)
    return service


def refusal(reason: str):
    return {"status": "abstain", "answer": "I'm not sure", "citations": [], "reason": reason,
            "reference_verified": False, "semantic_supported": False,
            "audit": {"generation_called": False, "semantic_verifier_called": False}}


def finite_setting(value: str, *, positive=False):
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError("Invalid number")
    return number


def settings():
    with st.sidebar:
        st.title("Workspace")
        st.caption("Local annual-report research")
        st.divider()
        configured = bool(os.getenv("FINANCE_QA_BASE_URL") and os.getenv("FINANCE_QA_MODEL")
                          and os.getenv("FINANCE_QA_THRESHOLD"))
        generate = st.toggle("Generate a cited answer", value=configured, key="generate_answer")
        with st.expander("Answer model", expanded=not configured):
            base = st.text_input("API base URL", value=os.getenv("FINANCE_QA_BASE_URL", ""),
                                 placeholder="https://your-provider.example/v1", key="api_base")
            model = st.text_input("Model", value=os.getenv("FINANCE_QA_MODEL", ""), key="api_model")
            key = st.text_input("API key", value=os.getenv("FINANCE_QA_API_KEY", ""),
                                type="password", key="api_key")
            reviewer = st.text_input("Reviewer model (optional)",
                                     value=os.getenv("FINANCE_QA_VERIFIER_MODEL", ""), key="reviewer_model")
            token_options = ["max_completion_tokens", "max_tokens"]
            selected = os.getenv("FINANCE_QA_TOKEN_PARAMETER", token_options[0])
            parameter = st.selectbox("Completion token parameter", token_options,
                                    index=token_options.index(selected) if selected in token_options else 0,
                                    key="token_parameter")
            st.caption("Settings stay in this browser session. Questions and selected passages are sent to this endpoint only when you request an answer.")
        with st.expander("Answer controls"):
            threshold = st.text_input("Minimum evidence score", value=os.getenv("FINANCE_QA_THRESHOLD", ""),
                                      placeholder="Set your validated cutoff", key="threshold")
            st.caption("Raw reranker score, not a probability. A blank cutoff disables answer generation. The previous 6.359375 cutoff was diagnostic, not a production default.")
            deadline = st.text_input("Generation budget (seconds)",
                                     value=os.getenv("FINANCE_QA_DEADLINE_SECONDS", "30"), key="deadline")
            st.caption("The budget starts after retrieval. An active model call can finish after the deadline; its late answer is discarded.")
        st.divider()
        st.caption("Evidence: question scope → company-scoped vector pools (global Top 20 if no company) → reranked, balanced Top 5")
        st.caption("Search runs locally. First search loads the models.")
        if st.button("Clear results", key="clear_results"):
            for name in ("search_result", "answer_result", "search_error", "preview", "pending_answer"):
                st.session_state.pop(name, None)
        return {"generate": generate, "base": base, "model": model, "key": key, "reviewer": reviewer,
                "parameter": parameter, "threshold": threshold, "deadline": deadline}


def answer_options(config):
    if not config["threshold"].strip():
        return None, "unconfigured_threshold"
    try:
        threshold = finite_setting(config["threshold"])
    except ValueError:
        return None, "invalid_threshold"
    if not config["base"].strip() or not config["model"].strip():
        return None, "model_not_configured"
    try:
        generator = ModelConfig(config["base"].strip(), config["model"].strip(), api_key=config["key"],
                                token_parameter=config["parameter"])
        verifier = ModelConfig(config["base"].strip(), config["reviewer"].strip() or config["model"].strip(),
                               api_key=config["key"], token_parameter=config["parameter"])
        deadline = finite_setting(config["deadline"], positive=True)
    except (ValueError, TypeError):
        return None, "invalid_model_configuration"
    return {"threshold": threshold, "generator_config": generator, "verifier_config": verifier,
            "deadline_seconds": deadline}, None


def citation_list(chunk):
    citations = list(chunk.get("citations", []))
    citations.extend(atom["citation"] for atom in chunk.get("evidence", []) if atom.get("citation"))
    if chunk.get("citation"):
        citations.append(chunk["citation"])
    seen, result = set(), []
    for citation in citations:
        identity = (citation.get("document_id"), citation.get("pdf_page"), citation.get("sha256"))
        if identity not in seen:
            seen.add(identity)
            result.append(citation)
    return result


def description(catalog, citation):
    try:
        return catalog.describe(citation)
    except (SourceCatalogError, OSError, ValueError):
        return None


def source_actions(catalog, citation, key):
    info = description(catalog, citation)
    if info is None:
        st.caption("This source could not be matched to the verified report catalog.")
        return
    st.caption(info["source_label"])
    columns = st.columns(2)
    if info.get("public_url"):
        columns[0].link_button("Open source PDF ↗", info["public_url"], key=f"{key}_link")
    if columns[1].button("Preview source page", key=f"{key}_preview", disabled=not info["local_available"]):
        try:
            with st.spinner("Checking and opening the source PDF…"):
                png = catalog.page_png(citation)
                pdf = catalog.pdf_bytes(citation)
            st.session_state["preview"] = {"info": info, "png": png, "pdf": pdf}
        except (OSError, ValueError, RuntimeError):
            st.warning("The local PDF could not be verified or opened. Use the source link above.")


def evidence_panel(search, catalog):
    selected = search.get("selected", [])
    st.subheader("2 · Retrieved evidence")
    st.caption(f"{len(search.get('candidates', []))} candidate passages · {len(selected)} selected for the answer")
    scope = search.get("scope") or {}
    if scope.get("tickers") or scope.get("notes") or scope.get("unknown_companies"):
        with st.container(border=True):
            companies = ", ".join(scope.get("tickers") or []) or "not recognized (searching all reports)"
            mode = scope.get("period_mode", "")
            years = ", ".join(f"FY{year}" for year in scope.get("years") or []) or PERIOD_LABELS.get(mode, mode)
            st.caption(f"Interpreted scope · companies: {companies} · period: {years}")
            for note in scope.get("notes") or []:
                st.caption(note)
    for position, chunk in enumerate(selected, 1):
        citations = citation_list(chunk)
        info = description(catalog, citations[0]) if citations else None
        title = info["title"] if info else chunk.get("document_id", "Source report")
        with st.expander(f"{position}. {title}", expanded=position == 1):
            if info:
                st.caption(f"PDF page {info['pdf_page']} · reranker score {chunk.get('reranker_score', 0):.3f}")
            st.text(chunk.get("text", ""))
            for page_index, citation in enumerate(citations):
                source_actions(catalog, citation, f"selected_{position}_{page_index}")
    with st.expander("All retrieved candidates"):
        candidates = search.get("candidates", [])
        rows = []
        for position, chunk in enumerate(candidates, 1):
            citations = citation_list(chunk)
            info = description(catalog, citations[0]) if citations else None
            rows.append({"Rank": position, "Report": info["title"] if info else chunk.get("document_id", ""),
                         "PDF page": info["pdf_page"] if info else None, "Vector score": chunk.get("score")})
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
            index = st.selectbox("Inspect a candidate", range(len(rows)),
                                 format_func=lambda n: f"{n+1}. {rows[n]['Report']}", key="candidate_selection")
            st.text(candidates[index].get("text", ""))
            for page_index, citation in enumerate(citation_list(candidates[index])):
                source_actions(catalog, citation, f"candidate_{index}_{page_index}")


def answer_panel(search, config, catalog, device):
    st.subheader("3 · Answer & citations")
    if st.session_state.get("search_error") or not search.get("selected"):
        st.session_state.pop("pending_answer", None)
        st.info("No complete answer context was selected. Review any available candidates and retry the search.")
        return
    if st.session_state.pop("pending_answer", False):
        options, error = answer_options(config)
        if error:
            st.session_state["answer_result"] = refusal(error)
        else:
            try:
                with st.spinner("Preparing and verifying the answer…"):
                    st.session_state["answer_result"] = get_service(str(ROOT), device).answer(search, **options)
            except (DemoError, OSError, ValueError, RuntimeError):
                st.session_state["answer_result"] = refusal("answer_unavailable")
    answer = st.session_state.get("answer_result")
    if answer is None:
        st.info("Evidence is ready. Enable “Generate a cited answer” and submit your query to use an answer model.")
        return
    if answer.get("status") != "answer":
        st.warning("I'm not sure")
        reason = answer.get("reason", "answer_unavailable")
        st.caption(REASONS.get(reason, "A required answer check failed. No unverified answer was released."))
    else:
        st.markdown(answer["answer"])
        st.caption("Source references checked · claim support reviewed")
        for position, entry in enumerate(answer.get("citations", []), 1):
            with st.container(border=True):
                st.write(f"Source {entry.get('reference_number', position)}")
                st.text(entry.get("quote", ""))
                source_actions(catalog, entry["citation"], f"answer_{position}")
                with st.expander("Citation trace"):
                    st.json({"evidence_id": entry.get("evidence_id"), "locator": entry.get("source_locator"),
                             "quote_char_start": entry.get("quote_char_start"), "quote_char_end": entry.get("quote_char_end")})
    payload = {"query": search["query"], "answer": answer}
    st.download_button("Download answer & citations", json.dumps(payload, ensure_ascii=False, indent=2),
                       file_name="research_answer.json", mime="application/json", on_click="ignore")
    with st.expander("Run details"):
        st.json({"retrieval_timings": search.get("timings", {}), "answer_audit": answer.get("audit", {}),
                 "reason": answer.get("reason")})


def main():
    st.set_page_config(page_title="Financial report Q&A", page_icon="📑", layout="wide")
    config = settings()
    st.caption("FINANCIAL RESEARCH WORKSPACE")
    st.title("Ask the reports. Inspect the evidence.")
    st.write("Research annual reports with answers traced to the original source passages and PDF pages.")
    try:
        catalog = SourceCatalog(ROOT)
    except (OSError, ValueError):
        st.error("The report catalog is unavailable. Prepare data/manifest.json before starting the demo.")
        return
    device = os.getenv("FINANCE_QA_DEVICE", "cpu")
    with st.form("question_form"):
        query = st.text_area("1 · Your research question", key="query", height=95, max_chars=10_000,
                             placeholder="According to Microsoft's FY2024 annual report, what was its operating income?")
        submitted = st.form_submit_button("Search reports", type="primary")
    st.caption("Example: According to its cash-flow statement, how much cash did NVIDIA pay for share buybacks in FY2025?")
    if submitted:
        for name in ("search_result", "answer_result", "search_error", "preview", "pending_answer", "candidate_selection"):
            st.session_state.pop(name, None)
        try:
            if not query.strip():
                raise ValueError("invalid_query")
            with st.status("Finding relevant passages…", expanded=False) as status:
                search = get_service(str(ROOT), device).search(query)
                st.session_state["search_result"] = search
                st.session_state["pending_answer"] = config["generate"]
                status.update(label="Evidence retrieved", state="complete")
        except DemoError as error:
            st.session_state["search_error"] = error.code
            if error.partial_result:
                st.session_state["search_result"] = error.partial_result
        except (ValueError, OSError, RuntimeError):
            st.session_state["search_error"] = "invalid_query" if not query.strip() else "retrieval_unavailable"
    error = st.session_state.get("search_error")
    if error:
        st.error(REASONS.get(error, "Search could not complete. Check the local models, indexes, and source artifacts; no answer was generated."))
    search = st.session_state.get("search_result")
    if search:
        st.caption("Showing results for:")
        st.text(search["query"])
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            evidence_panel(search, catalog)
        with right:
            answer_panel(search, config, catalog, device)
    else:
        with st.container(border=True):
            st.subheader("From a question to a source you can check")
            st.write("Search the reports, inspect the selected passages, then read a cited answer. You can open each source PDF or preview its exact page.")
            st.caption("Evidence search is available without an answer-model connection.")
    preview = st.session_state.get("preview")
    if preview:
        st.divider()
        st.subheader("Source page")
        st.caption(preview["info"]["source_label"])
        st.download_button("Download verified PDF", preview["pdf"],
                           file_name=preview["info"]["source_file"], mime="application/pdf", on_click="ignore")
        st.image(preview["png"], width="stretch")


if __name__ == "__main__":
    main()
