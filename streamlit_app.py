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

from report_search import demo_presentation as view
from report_search.answer_verification import escape_markdown
from report_search.demo_service import CORPORA, DEFAULT_CORPUS, DemoError, DemoService
from report_search.demo_sources import SourceCatalog, SourceCatalogError
from report_search.llm_client import ModelConfig


ICONS = {"passed": ":green[:material/check_circle:]", "failed": ":red[:material/cancel:]",
         "stopped": ":orange[:material/do_not_disturb_on:]", "open": ":gray[:material/lock_open:]",
         "not_run": ":gray[:material/radio_button_unchecked:]", "not_recorded": ":gray[:material/remove:]"}
RESULT_KEYS = ("search_result", "answer_result", "search_error", "preview", "pending_answer", "run_example")
HOW_IT_WORKS = (
    (":material/manage_search:", "1 · Reads your question",
     "Spots the companies (names, tickers, typos, Chinese names) and fiscal years, and shows you how it read them."),
    (":material/find_in_page:", "2 · Finds the evidence",
     "Searches only those companies' reports; a second model re-ranks the passages and keeps the best five, "
     "plus supporting tables."),
    (":material/format_quote:", "3 · Answers from quotes",
     "A language model writes short claims, each backed by a word-for-word quote from a report page."),
    (":material/verified_user:", "4 · Checks, or refuses",
     "Code checks every quote, number, year and calculation, and a separate model call reviews the result. "
     "Anything unproven becomes “I'm not sure”."),
)


@st.cache_resource(show_spinner=False)
def get_service(root: str, device: str, corpus: str):
    service = DemoService(Path(root), device=device, corpus=corpus)
    atexit.register(service.close)
    return service


@st.cache_data(show_spinner=False)
def corpus_overview():
    return view.corpus_overview(ROOT)


def refusal(reason: str):
    return {"status": "abstain", "answer": "I'm not sure", "citations": [], "reason": reason,
            "reference_verified": False, "semantic_supported": False,
            "audit": {"generation_called": False, "semantic_verifier_called": False}}


def finite_setting(value: str, *, positive=False):
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError("Invalid number")
    return number


def answer_options(config):
    """Model and gate settings for DemoService.answer, or the setup reason it must refuse with."""
    if not config["threshold"].strip():
        return None, "unconfigured_threshold"
    try:
        threshold = finite_setting(config["threshold"])
    except ValueError:
        return None, "invalid_threshold"
    if not config["base"].strip() or not config["model"].strip():
        return None, "model_not_configured"
    # The server's key is used only with the server's own endpoint, never with one typed in the browser.
    server_base = os.getenv("FINANCE_QA_BASE_URL", "").strip().rstrip("/")
    same_endpoint = bool(server_base) and config["base"].strip().rstrip("/") == server_base
    key = config["key"] or (os.getenv("FINANCE_QA_API_KEY", "") if same_endpoint else "")
    try:
        generator = ModelConfig(config["base"].strip(), config["model"].strip(), api_key=key,
                                token_parameter=config["parameter"])
        verifier = ModelConfig(config["base"].strip(), config["reviewer"].strip() or config["model"].strip(),
                               api_key=key, token_parameter=config["parameter"])
        deadline = finite_setting(config["deadline"], positive=True)
    except (ValueError, TypeError):
        return None, "invalid_model_configuration"
    return {"threshold": threshold, "generator_config": generator, "verifier_config": verifier,
            "deadline_seconds": deadline}, None


def answer_mode(config):
    """Badge and explanation of whether answers are off, will refuse, or are ready (no network call)."""
    if not config["generate"]:
        return (("Evidence only", "gray", ":material/visibility:"),
                ("caption", ":material/info: Evidence search is available without an answer-model connection. "
                            "To get checked answers, turn on **Generate cited answers** in the sidebar, connect "
                            "a model and set a minimum evidence score."))
    options, error = answer_options(config)
    if error:
        return (("Answers will refuse", "orange", ":material/warning:"),
                ("warning", "Answers will refuse until setup is complete: " + view.REASONS[error]))
    reviewer = options["verifier_config"].model
    reviewer = "same model, separate call" if reviewer == options["generator_config"].model else reviewer
    threshold = view.format_threshold(options["threshold"])
    return (("Answers on", "green", ":material/check_circle:"),
            ("caption", f":material/verified_user: Answer model {escape_markdown(options['generator_config'].model)}"
                        f" · reviewer {escape_markdown(reviewer)} · minimum evidence score {threshold}"))


def settings(corpus, device):
    configured = bool(os.getenv("FINANCE_QA_BASE_URL") and os.getenv("FINANCE_QA_MODEL")
                      and os.getenv("FINANCE_QA_THRESHOLD"))
    with st.sidebar:
        st.subheader("Answer engine")
        badge = st.empty()
        generate = st.toggle("Generate cited answers", value=configured, key="generate_answer")
        with st.expander("Connect a model", expanded=not configured, icon=":material/key:"):
            base = st.text_input("API base URL", value=os.getenv("FINANCE_QA_BASE_URL", ""),
                                 placeholder="https://your-provider.example/v1", key="api_base")
            model = st.text_input("Model", value=os.getenv("FINANCE_QA_MODEL", ""), key="api_model")
            # The server's key is never sent to the browser; a blank field uses it on the server, and only
            # with the server's own FINANCE_QA_BASE_URL.
            server_key = bool(os.getenv("FINANCE_QA_API_KEY") and os.getenv("FINANCE_QA_BASE_URL"))
            key = st.text_input("API key", value="", type="password", key="api_key",
                                placeholder=("Blank uses the server's key, with the server's endpoint only" if server_key
                                             else "Optional for local endpoints"))
            reviewer = st.text_input("Reviewer model (optional)", value=os.getenv("FINANCE_QA_VERIFIER_MODEL", ""),
                                     key="reviewer_model",
                                     help="A separate call reviews every claim; blank reuses the answer model.")
            token_options = ["max_completion_tokens", "max_tokens"]
            selected = os.getenv("FINANCE_QA_TOKEN_PARAMETER", token_options[0])
            parameter = st.selectbox("Completion token parameter", token_options,
                                     index=token_options.index(selected) if selected in token_options else 0,
                                     key="token_parameter",
                                     help="Change only if your endpoint rejects max_completion_tokens.")
            st.caption("Settings stay in this browser session. The question and the selected passages are sent to "
                       "this endpoint only when an answer is requested.")
        with st.expander("Safety settings", expanded=not os.getenv("FINANCE_QA_THRESHOLD"), icon=":material/shield:"):
            threshold = st.text_input(
                "Minimum evidence score (required for answers)", value=os.getenv("FINANCE_QA_THRESHOLD", ""),
                placeholder="Required — there is no default", key="threshold",
                help="The best passage's re-ranker score must reach this value before any answer is drafted. "
                     "The 6.359375 value in the docs was a development diagnostic, not a recommended setting.")
            st.caption("Leave blank and every answer refuses. This is a raw re-ranker score, not a probability; "
                       "a cutoff tuned on one corpus or device is not calibrated for another.")
            deadline = st.text_input("Answer time limit (seconds)", value=os.getenv("FINANCE_QA_DEADLINE_SECONDS", "30"),
                                     key="deadline",
                                     help="Starts after the search. A model call already running can finish late; "
                                          "a late answer is discarded.")
        st.divider()
        st.caption("**About this run**")
        st.caption(view.CORPUS_LABELS.get(corpus, "Unknown corpus"))
        if corpus in CORPORA and any(not (ROOT / path).exists() for path in CORPORA[corpus].values()):
            st.warning(view.corpus_missing_text(corpus))
        st.caption(f"Device: {device}")
        stats = (st.session_state.get("search_result") or {}).get("stats") or {}
        if isinstance(stats.get("chunk_count"), int):
            st.caption(f"Index: {stats['chunk_count']:,} passages · {escape_markdown(stats.get('encoder_model', ''))} → "
                       f"{escape_markdown(stats.get('reranker_model', ''))}")
        st.caption("Search runs locally. The first search also loads the search models, so it takes longer.")
        if st.button("Clear results", key="clear_results", icon=":material/restart_alt:"):
            for name in RESULT_KEYS:
                st.session_state.pop(name, None)
    config = {"generate": generate, "base": base, "model": model, "key": key, "reviewer": reviewer,
              "parameter": parameter, "threshold": threshold, "deadline": deadline}
    (label, color, icon), _ = answer_mode(config)
    with badge.container():
        st.badge(label, color=color, icon=icon)
    return config


def use_example(question):
    st.session_state["query"] = question
    st.session_state["run_example"] = True


def answer_allowed():
    """A new answer only when there is none yet, or the last one refused for a reason settings can fix."""
    answer = st.session_state.get("answer_result")
    return answer is None or (answer.get("status") != "answer"
                              and view.retryable(answer.get("reason") or "", answer.get("audit") or {}))


def request_answer():
    # A stale or repeated click while an answer is shown never triggers another paid call.
    if answer_allowed():
        st.session_state["pending_answer"] = True


def close_preview():
    st.session_state.pop("preview", None)


def opening(info, has_result):
    st.caption("PORTFOLIO DEMO · QUESTION ANSWERING OVER ANNUAL REPORTS")
    st.title("Ask 200 annual reports. Check every answer.")
    st.write("Ask about the annual reports of 40 US companies, fiscal years 2021–2025. The app finds the relevant "
             "pages, answers only with exact quotes from them, and checks every quote, number and calculation "
             "before it shows an answer. If any check fails, it says “I'm not sure” and tells you why.")
    if info and has_result:
        st.caption(f"{info['reports']} annual reports · {info['companies']} companies · "
                   f"FY{info['years'][0]}–FY{info['years'][1]} · {info['pages']:,} PDF pages")
    elif info:
        columns = st.columns(4)
        columns[0].metric("Annual reports", info["reports"], border=True)
        columns[1].metric("Companies", info["companies"], border=True,
                          help=f"{len(info['sectors'])} sectors: " + ", ".join(info["sectors"]))
        columns[2].metric("Fiscal years", f"{info['years'][0]}–{info['years'][1]}", border=True)
        columns[3].metric("PDF pages", f"{info['pages']:,}", border=True)
    if info:
        with st.expander("Which companies are covered?", icon=":material/apartment:"):
            for sector, companies in info["sectors"].items():
                names = ", ".join(f"{company} ({ticker})" for company, ticker in companies)
                st.markdown(f"**{escape_markdown(sector)}:** {escape_markdown(names)}")
    if has_result:
        with st.expander("How it works", icon=":material/account_tree:"):
            how_it_works()
    else:
        how_it_works()


def how_it_works():
    for column, (icon, title, text) in zip(st.columns(len(HOW_IT_WORKS)), HOW_IT_WORKS):
        with column.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)


def examples(expanded):
    with st.expander("Try an example", expanded=expanded, icon=":material/lightbulb:"):
        st.caption("One click runs the search. These questions were written for this demo, not taken from the "
                   "evaluation sets; results depend on your settings and can vary between runs.")
        number = 0
        for column, (group, icon, questions) in zip(st.columns(len(view.EXAMPLES)), view.EXAMPLES):
            with column:
                st.markdown(f"{icon} **{group}**")
                for question in questions:
                    st.button(question, key=f"example_{number}", width="stretch", wrap=True, on_click=use_example,
                              args=(question,))
                    number += 1
                if group == "See it refuse":
                    st.caption("Turn answers on to see the refusal; the question panel flags the problem either way.")


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


def source_actions(catalog, citation, key, highlight=None):
    """Open-PDF and preview buttons; the page preview opens right under the card that asked for it."""
    info = description(catalog, citation)
    if info is None:
        st.caption("This source could not be matched to the verified report catalog.")
        return
    with st.container(horizontal=True):
        if info.get("public_url"):
            st.link_button(f"Open PDF at p. {info['pdf_page']}", info["public_url"], icon=":material/open_in_new:",
                           key=f"{key}_link")
        clicked = st.button(f"Preview p. {info['pdf_page']}", icon=":material/image:", key=f"{key}_preview",
                            disabled=not info["local_available"])
    if clicked:
        try:
            with st.spinner("Checking and opening the source PDF…"):
                if highlight:
                    png, drawn = catalog.page_png(citation, highlight=highlight, report_highlight=True)
                else:
                    png, drawn = catalog.page_png(citation), False
                pdf = catalog.pdf_bytes(citation)
            st.session_state["preview"] = {"key": key, "info": info, "png": png, "pdf": pdf, "highlighted": drawn}
        except (OSError, ValueError, RuntimeError):
            st.warning("The local PDF could not be verified or opened. Use the source link above.")
    preview = st.session_state.get("preview")
    if preview and preview.get("key") == key:
        with st.container(border=True):
            st.image(preview["png"], width="stretch", caption=preview["info"]["source_label"])
            if preview.get("highlighted"):  # only when a box was actually drawn
                st.caption("Box: approximate location of the cited text, from parser coordinates.")
            st.caption("The PDF's SHA-256 was checked against the manifest before display. PDF page = position "
                       "in the file.")
            with st.container(horizontal=True):
                st.download_button("Download verified PDF", preview["pdf"], file_name=preview["info"]["source_file"],
                                   mime="application/pdf", on_click="ignore", key=f"{key}_pdf",
                                   icon=":material/download:")
                st.button("Close preview", key=f"{key}_close", on_click=close_preview, icon=":material/close:")


def run_pending_answer(search, config, device, corpus):
    """Run the requested answer exactly once per request, before anything shows its status."""
    if st.session_state.get("search_error") or not search.get("selected"):
        st.session_state.pop("pending_answer", None)
        return
    # With answers turned off nothing is sent to the endpoint, whatever button set the request.
    if not st.session_state.pop("pending_answer", False) or not config["generate"] or not answer_allowed():
        return
    options, error = answer_options(config)
    if error:
        st.session_state["answer_result"] = refusal(error)
        return
    try:
        with st.spinner("Writing and checking the answer…"):
            st.session_state["answer_result"] = get_service(str(ROOT), device, corpus).answer(search, **options)
    except (DemoError, OSError, ValueError, RuntimeError):
        st.session_state["answer_result"] = refusal("answer_unavailable")


def results_strip(search, answer, info):
    trace = search.get("retrieval_trace") or {}
    summary = view.scope_summary(search.get("scope"), trace, (info or {}).get("names") or {},
                                 (info or {}).get("reports", 200))
    timings = search.get("timings") or {}
    columns = st.columns(3)
    with columns[0].container(border=True, height="stretch"):
        st.markdown("**1 · Question read**")
        st.markdown(summary["companies"])
        if st.session_state.get("search_error"):
            st.caption("Search did not finish")
        else:
            st.caption(summary["period"])
            st.caption(summary["searched"])
        if summary["badges"]:
            with st.container(horizontal=True):
                for label, color in summary["badges"]:
                    st.badge(label, color=color)
    with columns[1].container(border=True, height="stretch"):
        st.markdown("**2 · Evidence found**")
        if st.session_state.get("search_error"):
            st.markdown("Search incomplete")
        else:
            best, added, _ = view.evidence_breakdown(search.get("selected") or [], trace)
            st.markdown(f"{len(search.get('candidates') or [])} candidates → {best} best + {added} added")
            total, startup = timings.get("total_ms"), timings.get("startup_ms")
            if isinstance(total, (int, float)):
                text = f"{total / 1000:.1f} s"
                if isinstance(startup, (int, float)) and startup > 1000:
                    text += f" (includes {startup / 1000:.0f} s one-time model loading)"
                st.caption(text)
    with columns[2].container(border=True, height="stretch"):
        st.markdown("**3 · Answer**")
        if answer is None:
            st.badge("Not requested", color="gray")
        elif answer.get("status") == "answer":
            st.badge("Verified answer", color="green", icon=":material/verified:")
        else:
            st.badge("I'm not sure", color="orange", icon=":material/help:")
        elapsed = ((answer or {}).get("audit") or {}).get("generation_elapsed_ms")
        if isinstance(elapsed, (int, float)):
            st.caption(f"{elapsed / 1000:.1f} s to write and check")
    if summary["warn"] and summary["notes"]:
        st.warning("  \n".join(summary["notes"]), icon=":material/report:")
    else:
        for note in summary["notes"]:
            st.caption(note)


def checklist(answer):
    steps = view.check_steps(answer)
    confirmed = sum(state == "passed" for _, state, _ in steps)
    title = (f"How this answer was checked ({confirmed} of {len(steps)} steps confirmed"
             + ("; evidence gate open)" if any(state == "open" for _, state, _ in steps) else ")")
             if answer.get("status") == "answer" else "Where the checks stopped")
    with st.expander(title, expanded=True, icon=":material/checklist:"):
        for label, state, detail in steps:
            st.markdown(f"{ICONS[state]} **{label}**")
            if detail:
                st.caption(detail)
        st.caption("Checks reduce errors; they are not a guarantee of correctness.")
    return steps


def verified_answer(answer, search, catalog):
    st.badge("Verified answer", icon=":material/verified:", color="green")
    parts = view.answer_parts(answer)
    if parts is None:
        st.markdown(answer["answer"])
    else:
        for part in parts:
            prefix = ":violet-badge[Computed by the app] " if part["computed"] else ""
            references = " ".join(f":blue-badge[{number}]" for number in part["refs"])
            st.markdown(f"{prefix}{part['text']} {references}")
            if part["calculation"]:
                expression, operands = view.calculation_lines(part["calculation"], answer.get("citations") or [])
                with st.container(border=True):
                    st.markdown(":material/calculate: **Recomputed by the app, not the model**")
                    st.markdown(expression)
                    for operand in operands:
                        st.caption(operand)
    checklist(answer)
    selected = {chunk.get("chunk_id"): position for position, chunk in enumerate(search.get("selected") or [], 1)}
    st.markdown("**Sources**")
    for position, entry in enumerate(answer.get("citations", []), 1):
        number = entry.get("reference_number", position)
        info = description(catalog, entry["citation"])
        title = info["title"] if info else entry["citation"].get("document_id", "Source report")
        with st.container(border=True):
            st.markdown(f"**Source {number}** · {escape_markdown(title)}")
            details = [view.CITATION_KIND_LABELS.get(entry.get("kind"), "Source text")]
            if info:
                details.append(info["source_label"])
            passages = sorted({selected[chunk] for chunk in entry.get("chunk_ids") or () if chunk in selected})
            if passages:
                details.append("from passage " + ", ".join(map(str, passages)))
            st.caption(escape_markdown(" · ".join(details)))
            st.text(entry.get("quote", ""))
            source_actions(catalog, entry["citation"], f"answer_{number}", highlight=entry["citation"].get("bbox"))
            with st.expander("Citation trace"):
                st.json({"evidence_id": entry.get("evidence_id"), "locator": entry.get("source_locator"),
                         "quote_char_start": entry.get("quote_char_start"), "quote_char_end": entry.get("quote_char_end"),
                         "document_id": entry["citation"].get("document_id"),
                         "pdf_page": entry["citation"].get("pdf_page"), "chunk_ids": entry.get("chunk_ids")})


def refusal_card(answer, scope):
    code = answer.get("reason") or "answer_unavailable"
    audit = answer.get("audit") or {}
    stage = view.refusal_stage(code, audit)
    st.warning("I'm not sure", icon=":material/help:")
    st.markdown(f"**{view.headline(code, audit)}**")
    st.caption(view.reason_text(code, audit))
    if stage == "setup":
        st.caption("No checks ran: setup needed.")
    elif stage != "service":
        steps = checklist(answer)
        if stage == "deadline" and not any(state == "failed" for _, state, _ in steps):
            st.caption("The answer was ready but arrived after the time limit, so it was discarded.")
    with st.container(border=True):
        st.markdown(":material/lightbulb: **What to try next**")
        st.caption(view.next_step(code, audit, scope))
        if view.retryable(code, audit) and st.session_state.get("generate_answer"):
            st.button("Retry answer with current settings", key="answer_retry", on_click=request_answer,
                      icon=":material/replay:")
    st.caption(f"Reason code: `{code}` · No unverified draft is shown. The passages on the right are search "
               "results, not proof.")


def answer_panel(search, answer, config, catalog):
    st.subheader("Answer")
    if st.session_state.get("search_error"):
        st.info("No answer was attempted because the search did not finish. Any passages found so far are shown "
                "on the right.")
        return
    if not search.get("selected"):
        st.info("No passages were selected, so no answer was attempted.")
        return
    if answer is None:
        if config["generate"]:
            st.info("The evidence is ready. Turn it into a checked answer:")
            st.button("Generate answer from this evidence", key="answer_now", type="primary", on_click=request_answer,
                      icon=":material/auto_awesome:")
        else:
            st.info("Answers are off, so only the evidence is shown. Turn on “Generate cited answers” in the sidebar "
                    "to have it turned into a checked answer.")
        return
    if answer.get("status") == "answer":
        verified_answer(answer, search, catalog)
    else:
        refusal_card(answer, search.get("scope"))
    payload = {"query": search["query"], "answer": answer}
    st.download_button("Download answer & citations", json.dumps(payload, ensure_ascii=False, indent=2),
                       file_name="research_answer.json", mime="application/json", on_click="ignore",
                       icon=":material/download:")
    with st.expander("Technical run record"):
        st.json({"retrieval_timings": search.get("timings", {}), "answer_audit": answer.get("audit", {}),
                 "reason": answer.get("reason")})


def evidence_panel(search, answer, catalog):
    st.subheader("Evidence from the reports")
    selected, candidates = search.get("selected", []), search.get("candidates", [])
    trace = search.get("retrieval_trace") or {}
    st.caption(f"{len(candidates)} candidate passages · {len(selected)} selected for the answer")
    st.caption(view.evidence_breakdown(selected, trace)[2])
    cited = {}
    for entry in (answer or {}).get("citations") or ():
        for chunk_id in entry.get("chunk_ids") or ():
            cited.setdefault(chunk_id, []).append(str(entry.get("reference_number")))
    scores = [chunk.get("reranker_score") for chunk in selected if isinstance(chunk.get("reranker_score"), (int, float))]
    best_score = max(scores) if scores else None
    reserved = set(trace.get("reserved_latest") or ())
    used_tab, all_tab = st.tabs([f"Passages used ({len(selected)})", f"All candidates ({len(candidates)})"])
    with used_tab:
        if not selected:
            st.caption("No passages were selected.")
        for position, chunk in enumerate(selected, 1):
            citations = citation_list(chunk)
            info = description(catalog, citations[0]) if citations else None
            kind = view.KIND_LABELS.get(chunk.get("kind"), "Passage")
            label = (f"{position}. {info['company']} FY{info['fiscal_year']} · PDF p. {info['pdf_page']} · {kind}"
                     if info else f"{position}. {chunk.get('document_id', 'Source report')}")
            with st.expander(label, expanded=position == 1):
                with st.container(horizontal=True):
                    if chunk.get("context_reason"):
                        st.badge("Added context", color="violet", icon=":material/add_circle:")
                    else:
                        st.badge(f"Best match #{chunk.get('rank', position)}", color="blue", icon=":material/star:")
                    if chunk.get("chunk_id") in reserved:
                        st.badge("Latest-report slot", color="gray")
                    if chunk.get("structured_reliable") is False:
                        st.badge("Table layout unverified", color="orange", icon=":material/warning:")
                if chunk.get("context_reason"):
                    st.caption("Why added: " + view.CONTEXT_LABELS.get(chunk["context_reason"], "added context passage"))
                score = chunk.get("reranker_score")
                page = info["pdf_page"] if info else "?"
                st.caption(f"PDF page {page}" + (f" · relevance {score:.2f}" if isinstance(score, (int, float)) else ""),
                           help="Re-ranker score: higher means more relevant. It is a ranking score, not a "
                                "probability. PDF page numbers can differ from printed page numbers.")
                if best_score is not None and score == best_score:
                    st.caption(":material/flag: Highest relevance: this is the score compared with your minimum "
                               "evidence score.")
                if chunk.get("chunk_id") in cited:
                    st.caption("Cited in the answer as Source " + ", ".join(cited[chunk["chunk_id"]]))
                with st.container(height=320 if len(chunk.get("text", "")) > 1200 else "content"):
                    st.text(chunk.get("text", ""))
                for page_index, citation in enumerate(citations):
                    source_actions(catalog, citation, f"selected_{position}_{page_index}")
    with all_tab:
        used = {chunk.get("chunk_id") for chunk in selected}
        rows = []
        for position, chunk in enumerate(candidates, 1):
            citations = citation_list(chunk)
            info = description(catalog, citations[0]) if citations else None
            rows.append({"Rank": position, "Report": info["title"] if info else chunk.get("document_id", ""),
                         "PDF page": info["pdf_page"] if info else None,
                         "Kind": view.KIND_LABELS.get(chunk.get("kind"), "Passage"),
                         "Used for answer": chunk.get("chunk_id") in used, "Similarity": chunk.get("score")})
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch", column_config={
                "Used for answer": st.column_config.CheckboxColumn(),
                "Similarity": st.column_config.NumberColumn(
                    format="%.3f", help="Cosine similarity between the question and passage embeddings; "
                                        "not comparable with relevance scores")})
            index = st.selectbox("Inspect a candidate", range(len(rows)),
                                 format_func=lambda n: f"{n + 1}. {rows[n]['Report']}", key="candidate_selection")
            st.text(candidates[index].get("text", ""))
            for page_index, citation in enumerate(citation_list(candidates[index])):
                source_actions(catalog, citation, f"candidate_{index}_{page_index}")


def footer():
    st.divider()
    st.caption(f"[Source code and write-up]({view.REPO_URL}) · "
               f"[Architecture and experiments]({view.REPO_URL}/blob/main/docs/system_architecture.md)")
    st.caption("Checks reduce errors but do not prove correctness. Evaluation labels and grades were written by AI "
               "agents, with no human review.")


def main():
    configured = bool(os.getenv("FINANCE_QA_BASE_URL") and os.getenv("FINANCE_QA_MODEL")
                      and os.getenv("FINANCE_QA_THRESHOLD"))
    st.set_page_config(page_title="Annual Report Q&A", page_icon=":material/description:", layout="wide",
                       initial_sidebar_state="collapsed" if configured else "auto")
    corpus = (os.getenv("FINANCE_QA_CORPUS") or DEFAULT_CORPUS).strip()
    device = os.getenv("FINANCE_QA_DEVICE", "cpu")
    config = settings(corpus, device)
    info = corpus_overview()
    # Filled after the search, so the first result already gets the compact header.
    top, example_area = st.container(), st.container()
    if corpus not in CORPORA:
        with top:
            opening(info, False)
        st.error("Unknown corpus. Set FINANCE_QA_CORPUS to v1 or v2 and restart the app.")
        return
    try:
        catalog = SourceCatalog(ROOT)
    except (OSError, ValueError):
        with top:
            opening(info, False)
        st.error("The report catalog is unavailable. Prepare data/manifest.json before starting the demo.")
        return
    with st.form("question_form"):
        query = st.text_area("Or ask your own question", key="query", height=95, max_chars=10_000,
                             placeholder="e.g. How much did Lowe's spend on share repurchases in fiscal 2024?")
        submitted = st.form_submit_button("Search reports", type="primary", icon=":material/search:")
    _, (kind, text) = answer_mode(config)
    (st.warning if kind == "warning" else st.caption)(text)
    if submitted or st.session_state.pop("run_example", False):
        for name in RESULT_KEYS + ("candidate_selection",):
            st.session_state.pop(name, None)
        try:
            if not query.strip():
                raise ValueError("invalid_query")
            with st.status("Finding relevant passages…", expanded=False) as status:
                search = get_service(str(ROOT), device, corpus).search(query)
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
        st.error(view.search_error_text(error, corpus))
    search = st.session_state.get("search_result")
    with top:
        opening(info, bool(search))
    with example_area:
        examples(expanded=not search)
    if search:
        st.caption("Showing results for:")
        st.text(search["query"])
        run_pending_answer(search, config, device, corpus)
        answer = st.session_state.get("answer_result")
        results_strip(search, answer, info)
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            answer_panel(search, answer, config, catalog)
        with right:
            evidence_panel(search, answer, catalog)
    footer()


if __name__ == "__main__":
    main()
