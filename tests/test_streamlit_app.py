"""Exercise the real demo UI without initializing models or contacting a provider."""
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

from report_search.answer_verification import render_verified_answer
from report_search.demo_presentation import EXAMPLES
from report_search.demo_service import DemoError


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "According to Microsoft's FY2024 annual report, what was operating income?"
CITATION = {"document_id": "MSFT_FY2024_annual_report-" + "a" * 12,
            "source_file": "MSFT_FY2024_annual_report.pdf", "sha256": "a" * 64,
            "ticker": "MSFT", "fiscal_year": 2024, "pdf_page": 48}


def retrieval_result(query=QUESTION):
    selected = {"chunk_id": "statement", "document_id": CITATION["document_id"],
                "text": "Operating income | 109,433 | 88,523 | 83,383",
                "score": 0.81, "reranker_score": 7.5, "citations": [deepcopy(CITATION)]}
    other = {"chunk_id": "context", "document_id": CITATION["document_id"],
             "text": "Amounts in millions; fiscal years 2024, 2023 and 2022.",
             "score": 0.71, "citations": [deepcopy(CITATION)]}
    return {"query": query, "candidates": [selected, other], "selected": [deepcopy(selected)],
            "timings": {"vector_search_ms": 1.0, "reranking_ms": 2.0}, "stats": {}}


def verified_answer():
    return {"status": "answer", "answer": "Microsoft reported $109,433 million of operating income. [^1]",
            "reference_verified": True, "semantic_supported": True, "reason": None,
            "audit": {"generation_called": True, "semantic_verifier_called": True},
            "citations": [{"reference_number": 1, "evidence_id": "E15",
                           "quote": "Operating income | 109,433 | 88,523 | 83,383",
                           "quote_char_start": 0, "quote_char_end": 44,
                           "citation": deepcopy(CITATION),
                           "source_locator": {"table_id": "income_statement", "row_index": 13}}]}


def realistic_answer():
    """A real rendered answer: escaped claims, a recomputed calculation, footnotes and a full audit."""
    evidence = {}
    for identifier, text, year in (("E3", "Net revenue | $ 54,228 | $ 63,054", 2023),
                                   ("E9", "Net revenue | $ 79,024 | $ 77,867", 2021)):
        evidence[identifier] = {
            "evidence_id": identifier, "kind": "row", "text": text, "chunk_ids": ["statement"],
            "citation": dict(deepcopy(CITATION), fiscal_year=year, bbox=[10, 10, 500, 80]),
            "source_locator": {"table_id": f"INTC_FY{year}:p40:table1", "row_index": 3}}
    claims = [{"text": "Intel's net revenue was $54,228 million in FY2023.",
               "citations": [{"evidence_id": "E3", "quote": "Net revenue | $ 54,228"}]},
              {"text": "Intel's revenue fell 31.4% from FY2021 to FY2023.",
               "citations": [{"evidence_id": "E3", "quote": "Net revenue | $ 54,228"},
                             {"evidence_id": "E9", "quote": "Net revenue | $ 79,024"}],
               "calculation": {"expression": "(a - b) / b * 100", "result": "-31.4", "unit": "%", "operands": [
                   {"name": "a", "evidence_id": "E3", "value": "54,228", "label": "Net revenue FY2023"},
                   {"name": "b", "evidence_id": "E9", "value": "79,024", "label": "Net revenue FY2021"}]}}]
    verdict = {"answers_question": True, "claim_verdicts": [
        {"claim_index": index, "supported": True, "reason": "The quoted rows state these figures."} for index in (0, 1)]}
    answer = render_verified_answer(claims, {"source_verified": True, "chunks": [], "evidence": evidence}, verdict)
    answer["audit"] = {
        "confidence_gate": {"allowed": True, "reason": "allowed", "threshold": 6.5, "max_score": 7.5},
        "source_verified": True, "evidence_count": 2, "context_passages": 1, "generation_called": True,
        "generator": {"model": "generator-fixture"}, "generation_attempts": 2,
        "first_attempt_reason": "quote_not_in_source", "reference_check_passed": True,
        "claim_grounding": {"ok": True}, "semantic_verifier_called": True, "verifier": {"model": "reviewer-fixture"},
        "comparison_verdict_added": True}
    return answer


def refused(reason, audit):
    return {"status": "abstain", "answer": "I'm not sure", "citations": [], "reason": reason,
            "reference_verified": False, "semantic_supported": False, "audit": audit}


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        # Keep platform variables; remove only app configuration/credentials.
        environment = {key: value for key, value in os.environ.items() if not key.startswith("FINANCE_QA_")}
        self.environment = patch.dict(os.environ, environment, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.service = MagicMock()
        self.service.search.side_effect = retrieval_result
        self.service.answer.side_effect = lambda *_args, **_kwargs: verified_answer()
        self.service_class = patch("report_search.demo_service.DemoService", return_value=self.service)
        self.factory = self.service_class.start()
        self.addCleanup(self.service_class.stop)
        self.catalog = MagicMock()
        self.catalog.describe.side_effect = self.describe
        picture = BytesIO()
        Image.new("RGB", (2, 2), "white").save(picture, "PNG")
        self.catalog.page_png.return_value = picture.getvalue()
        self.catalog.pdf_bytes.return_value = b"%PDF-1.4\ncontrolled download fixture"
        self.catalog_class = patch("report_search.demo_sources.SourceCatalog", return_value=self.catalog)
        self.catalog_class.start()
        self.addCleanup(self.catalog_class.stop)
        self.app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20).run()
        self.assert_no_exception()

    @staticmethod
    def describe(citation):
        citation = citation.get("citation", citation)
        return {**citation, "company": "Microsoft", "page_count": 96,
                "title": "Microsoft (MSFT) FY2024 annual report",
                "source_label": "Microsoft (MSFT) FY2024 annual report — PDF page 48 of 96",
                "public_url": "https://reports.example/msft.pdf#page=48", "local_available": True}

    def assert_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, [item.message for item in self.app.exception])

    def submit(self, query=QUESTION):
        self.app.text_area(key="query").set_value(query)
        self.app.button(key="FormSubmitter:question_form-Search reports").click().run()
        self.assert_no_exception()

    def configure_generation(self):
        self.app.toggle(key="generate_answer").set_value(True)
        self.app.text_input(key="api_base").set_value("https://provider.example/v1")
        self.app.text_input(key="api_model").set_value("generator-fixture")
        self.app.text_input(key="reviewer_model").set_value("reviewer-fixture")
        self.app.text_input(key="api_key").set_value("secret-fixture-never-contacted")
        self.app.text_input(key="threshold").set_value("6.5")

    def visible_text(self):
        values = []
        for kind in ("text", "markdown", "caption", "warning", "error", "info"):
            values.extend(str(item.value) for item in getattr(self.app, kind))
        return "\n".join(values)

    def test_initial_page_does_not_construct_service_or_search(self):
        self.factory.assert_not_called()
        self.service.search.assert_not_called()
        self.service.answer.assert_not_called()
        self.assertFalse(self.app.toggle(key="generate_answer").value)
        self.assertIn("Evidence search is available without an answer-model connection", self.visible_text())

    def test_submit_search_displays_selected_and_all_candidate_evidence(self):
        self.submit()
        self.service.search.assert_called_once_with(QUESTION)
        self.service.answer.assert_not_called()
        text = self.visible_text()
        self.assertIn("Operating income | 109,433 | 88,523 | 83,383", text)
        self.assertIn("2 candidate passages · 1 selected", text)
        self.assertIn("PDF page 48", text)
        self.assertEqual(len(self.app.dataframe), 1)
        self.app.selectbox(key="candidate_selection").set_value(1).run()
        self.assert_no_exception()
        self.assertIn("Amounts in millions", self.visible_text())
        self.service.search.assert_called_once()
        self.service.answer.assert_not_called()

    def test_answer_citations_preview_and_settings_reruns_do_not_regenerate(self):
        self.configure_generation()
        self.submit()
        self.service.answer.assert_called_once()
        kwargs = self.service.answer.call_args.kwargs
        self.assertEqual(kwargs["threshold"], 6.5)
        self.assertEqual(kwargs["generator_config"].model, "generator-fixture")
        self.assertEqual(kwargs["verifier_config"].model, "reviewer-fixture")
        self.assertIn("Microsoft reported $109,433 million", self.visible_text())
        self.assertIn("Source 1", self.visible_text())
        self.assertTrue(any("E15" in item.value for item in self.app.json))
        self.app.button(key="answer_1_preview").click().run()
        self.assert_no_exception()
        self.catalog.page_png.assert_called_once_with(CITATION)
        self.catalog.pdf_bytes.assert_called_once_with(CITATION)
        self.assertIn("preview", self.app.session_state)
        self.app.text_input(key="api_model").set_value("different-model").run()
        self.app.text_input(key="threshold").set_value("99").run()
        self.app.run()
        self.assert_no_exception()
        self.service.search.assert_called_once()
        self.service.answer.assert_called_once()
        self.assertIn("Microsoft reported $109,433 million", self.visible_text())

    def test_clear_results_removes_answer_evidence_and_preview_without_calls(self):
        self.configure_generation()
        self.submit()
        self.app.button(key="selected_1_0_preview").click().run()
        self.app.button(key="clear_results").click().run()
        self.assert_no_exception()
        for key in ("search_result", "answer_result", "preview", "pending_answer"):
            self.assertNotIn(key, self.app.session_state)
        self.assertNotIn("Microsoft reported $109,433 million", self.visible_text())
        self.assertEqual(len(self.app.dataframe), 0)
        self.service.search.assert_called_once()
        self.service.answer.assert_called_once()

    def test_missing_threshold_then_missing_endpoint_refuse_but_keep_evidence(self):
        self.app.toggle(key="generate_answer").set_value(True)
        self.submit()
        self.assertEqual(self.app.session_state["answer_result"]["reason"], "unconfigured_threshold")
        self.assertIn("I'm not sure", self.visible_text())
        self.assertIn("Operating income | 109,433", self.visible_text())
        self.app.text_input(key="threshold").set_value("6.5")
        self.submit()
        self.assertEqual(self.app.session_state["answer_result"]["reason"], "model_not_configured")
        self.assertIn("Operating income | 109,433", self.visible_text())
        self.assertEqual(self.service.search.call_count, 2)
        self.service.answer.assert_not_called()

    def test_partial_retrieval_error_preserves_candidates_without_answer(self):
        self.configure_generation()
        partial = retrieval_result()
        partial["selected"] = []
        self.service.search.side_effect = DemoError("private-sensitive-error", partial_result=partial)
        self.submit()
        self.assertIn("Search could not complete", self.visible_text())
        self.assertNotIn("private-sensitive-error", self.visible_text())
        self.assertIn("0 selected for the answer", self.visible_text())
        self.assertIn("Operating income | 109,433", self.visible_text())
        self.assertEqual(len(self.app.dataframe), 1)
        self.service.answer.assert_not_called()

    def test_answer_error_is_sanitized_and_does_not_retry_on_rerun(self):
        self.configure_generation()
        self.service.answer.side_effect = RuntimeError("private-token-sensitive-provider-body")
        self.submit()
        self.assertIn("I'm not sure", self.visible_text())
        self.assertNotIn("private-token-sensitive-provider-body", self.visible_text())
        self.assertEqual(self.app.session_state["answer_result"]["reason"], "answer_unavailable")
        self.assertIn("Operating income | 109,433", self.visible_text())
        self.app.run()
        self.assert_no_exception()
        self.service.answer.assert_called_once()

    def restart(self, **environment):
        os.environ.update(environment)
        st.cache_resource.clear()
        self.app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20).run()
        self.assert_no_exception()

    def test_default_corpus_is_v2_and_v1_can_be_selected(self):
        self.submit()
        self.assertEqual(self.factory.call_args.kwargs["corpus"], "v2")
        self.factory.reset_mock()
        self.restart(FINANCE_QA_CORPUS="v1")
        self.submit()
        self.assertEqual(self.factory.call_args.kwargs["corpus"], "v1")

    def test_unknown_corpus_stops_before_the_form(self):
        self.restart(FINANCE_QA_CORPUS="v9")
        self.assertIn("Unknown corpus", self.visible_text())
        self.assertEqual(len(self.app.text_area), 0)
        self.factory.assert_not_called()

    def test_example_button_runs_its_search_without_an_answer(self):
        question = EXAMPLES[0][2][0]
        self.app.button(key="example_0").click().run()
        self.assert_no_exception()
        self.service.search.assert_called_once_with(question)
        self.assertEqual(self.app.text_area(key="query").value, question)
        self.service.answer.assert_not_called()
        self.app.run()
        self.service.search.assert_called_once()

    def test_question_card_shows_scope_and_warns_about_uncovered_companies(self):
        def scoped(query=QUESTION):
            result = retrieval_result(query)
            result["scope"] = {"tickers": ["MSFT"], "years": [2024], "period_mode": "explicit",
                               "unknown_companies": ["Netflix"],
                               "notes": ["Netflix is not in the corpus [click](http://evil.example)."]}
            result["retrieval_trace"] = {"mode": "scoped", "pools": [
                {"ticker": "MSFT", "fiscal_years": [2024], "top_k": 20, "purpose": "scope"}]}
            return result
        self.service.search.side_effect = scoped
        self.submit()
        text = self.visible_text()
        for fragment in ("Microsoft", "FY2024", "MSFT FY2024", "Not covered: Netflix"):
            self.assertIn(fragment, text)
        self.assertTrue(any("Netflix is not in the corpus" in item.value for item in self.app.warning))
        self.assertNotIn("[click](http", text)

    def test_added_context_passages_are_labelled(self):
        def with_extra(query=QUESTION):
            result = retrieval_result(query)
            result["selected"].append(dict(result["candidates"][1], rank=2, context_reason="same_table",
                                           structured_reliable=False, reranker_score=1.5))
            return result
        self.service.search.side_effect = with_extra
        self.submit()
        text = self.visible_text()
        for fragment in ("1 best match + 1 added for context", "continuation of a selected table",
                         "Table layout unverified", "Added context"):
            self.assertIn(fragment, text)
        self.assertEqual(len(self.app.dataframe), 1)

    def test_verified_answer_shows_claims_calculation_and_checklist(self):
        answer = realistic_answer()
        self.service.answer.side_effect = lambda *_args, **_kwargs: deepcopy(answer)
        self.configure_generation()
        self.submit()
        text = self.visible_text()
        for fragment in ("Intel", "Recomputed by the app", "7.50", "corrected once", "Computed by the app",
                         "Source 2", "Table row"):
            self.assertIn(fragment, text)
        self.assertFalse(any("[^1]" in item.value for item in self.app.markdown))
        self.assertTrue(any("6 of 6 steps confirmed" in item.label for item in self.app.expander))

    def test_minimal_audit_is_not_shown_as_fully_checked(self):
        self.configure_generation()
        self.submit()
        self.assertTrue(any("2 of 6 steps confirmed" in item.label for item in self.app.expander))

    def test_refusal_explains_where_checks_stopped_without_draft_details(self):
        audit = {"confidence_gate": {"allowed": True, "threshold": 6.5, "max_score": 7.5}, "source_verified": True,
                 "evidence_count": 3, "context_passages": 1, "generation_called": True,
                 "reference_check_passed": True, "semantic_verifier_called": False,
                 "claim_grounding": {"ok": False, "reason": "claim_number_not_in_evidence",
                                     "detail": "draft-figure-123"}}
        self.service.answer.side_effect = lambda *_args, **_kwargs: refused("claim_number_not_in_evidence", audit)
        self.configure_generation()
        self.submit()
        text = self.visible_text()
        for fragment in ("I'm not sure", "A figure, year or company was not backed by its quote",
                         "claim_number_not_in_evidence", "What to try next", "stopped here"):
            self.assertIn(fragment, text)
        self.assertNotIn("draft-figure-123", text)

    def test_low_evidence_refusal_shows_the_score_and_cutoff(self):
        audit = {"confidence_gate": {"allowed": False, "threshold": 6.5, "max_score": 5.1}}
        self.service.answer.side_effect = lambda *_args, **_kwargs: refused("below_threshold", audit)
        self.configure_generation()
        self.submit()
        self.assertIn("The best passage scored 5.10, below your minimum of 6.5", self.visible_text())

    def test_answer_can_be_requested_after_an_evidence_only_search(self):
        self.submit()
        self.service.answer.assert_not_called()
        self.configure_generation()
        self.app.run()
        self.service.answer.assert_not_called()
        self.app.button(key="answer_now").click().run()
        self.assert_no_exception()
        self.service.answer.assert_called_once()
        self.service.search.assert_called_once()
        self.assertIn("Microsoft reported $109,433 million", self.visible_text())

    def test_page_preview_opens_inline_and_closes(self):
        self.configure_generation()
        self.submit()
        self.app.button(key="answer_1_preview").click().run()
        self.assertEqual(len(self.app.image), 1)
        self.app.button(key="answer_1_close").click().run()
        self.assert_no_exception()
        self.assertNotIn("preview", self.app.session_state)
        self.assertEqual(len(self.app.image), 0)

    def test_cited_row_is_highlighted_on_the_preview(self):
        answer = realistic_answer()
        self.service.answer.side_effect = lambda *_args, **_kwargs: deepcopy(answer)
        picture = self.catalog.page_png.return_value
        for drawn in (True, False):
            with self.subTest(drawn=drawn):
                self.catalog.page_png.return_value = (picture, drawn)
                self.app.button(key="clear_results").click().run()
                self.configure_generation()
                self.submit()
                self.app.button(key="answer_1_preview").click().run()
                self.assert_no_exception()
                self.assertEqual(self.catalog.page_png.call_args.kwargs,
                                 {"highlight": [10, 10, 500, 80], "report_highlight": True})
                self.assertEqual("Box: approximate location" in self.visible_text(), drawn)

    def test_server_api_key_is_never_sent_to_the_browser(self):
        self.restart(FINANCE_QA_API_KEY="server-secret-fixture", FINANCE_QA_BASE_URL="https://provider.example/v1")
        self.assertEqual(self.app.text_input(key="api_key").value, "")
        self.app.toggle(key="generate_answer").set_value(True)
        self.app.text_input(key="api_model").set_value("generator-fixture")
        self.app.text_input(key="threshold").set_value("6.5")
        self.submit()
        self.assertEqual(self.service.answer.call_args.kwargs["generator_config"].api_key, "server-secret-fixture")
        self.assertNotIn("server-secret-fixture", self.visible_text())

    def test_server_api_key_is_not_sent_to_an_endpoint_typed_in_the_browser(self):
        self.restart(FINANCE_QA_API_KEY="server-secret-fixture", FINANCE_QA_BASE_URL="https://provider.example/v1")
        self.app.toggle(key="generate_answer").set_value(True)
        self.app.text_input(key="api_base").set_value("https://elsewhere.example/v1")
        self.app.text_input(key="api_model").set_value("generator-fixture")
        self.app.text_input(key="threshold").set_value("6.5")
        self.submit()
        self.assertEqual(self.service.answer.call_args.kwargs["generator_config"].api_key, "")

    def test_failed_search_does_not_claim_what_was_searched(self):
        self.service.search.side_effect = DemoError("vector_search_failed", partial_result=retrieval_result())
        self.submit()
        text = self.visible_text()
        self.assertIn("Search did not finish", text)
        self.assertNotIn("all 200 reports", text)

    def test_open_gate_is_labelled_and_not_counted_as_confirmed(self):
        answer = realistic_answer()
        answer["audit"]["confidence_gate"] = {"allowed": True, "threshold": -1_000_000, "max_score": 2.8}
        self.service.answer.side_effect = lambda *_args, **_kwargs: deepcopy(answer)
        self.configure_generation()
        self.submit()
        self.assertTrue(any("5 of 6 steps confirmed; evidence gate open" in item.label for item in self.app.expander))
        self.assertIn("not checked", self.visible_text())

    def test_a_shown_answer_is_never_regenerated_by_a_stale_request(self):
        self.configure_generation()
        self.submit()
        self.app.session_state["pending_answer"] = True
        self.app.run()
        self.assert_no_exception()
        self.service.answer.assert_called_once()

    def test_model_connection_refusal_can_be_retried_without_a_new_search(self):
        self.service.answer.side_effect = [refused("model_http_401", {"generation_called": False}),
                                           deepcopy(verified_answer())]
        self.configure_generation()
        self.submit()
        self.assertIn("Check the endpoint", self.visible_text())
        self.app.button(key="answer_retry").click().run()
        self.assert_no_exception()
        self.assertEqual(self.service.answer.call_count, 2)
        self.service.search.assert_called_once()
        self.assertIn("Microsoft reported $109,433 million", self.visible_text())

    def test_a_model_that_declines_is_shown_as_a_stop_not_a_failure(self):
        audit = {"confidence_gate": {"allowed": True, "threshold": 6.5, "max_score": 7.5}, "source_verified": True,
                 "evidence_count": 3, "context_passages": 1, "generation_called": True}
        self.service.answer.side_effect = lambda *_args, **_kwargs: refused("generator_abstained", audit)
        self.configure_generation()
        self.submit()
        self.assertTrue(any("do_not_disturb_on" in item.value and "Draft written" in item.value
                            for item in self.app.markdown))
        self.assertIn("The sources do not answer this, so the model declined", self.visible_text())
        self.assertNotIn("answer_retry", [button.key for button in self.app.button])

    def test_retry_is_hidden_and_inert_when_answers_are_turned_off(self):
        self.service.answer.side_effect = lambda *_args, **_kwargs: refused("model_http_401", {"generation_called": False})
        self.configure_generation()
        self.submit()
        self.app.toggle(key="generate_answer").set_value(False).run()
        self.assertNotIn("answer_retry", [button.key for button in self.app.button])
        self.app.session_state["pending_answer"] = True
        self.app.run()
        self.assert_no_exception()
        self.service.answer.assert_called_once()

    def test_time_limit_refusal_can_be_retried(self):
        audit = {"generation_called": True,
                 "stage_timings": [{"stage": "source_binding", "status": "completed"},
                                   {"stage": "generation", "status": "deadline_exceeded"}]}
        self.service.answer.side_effect = [refused("deadline_exceeded", audit), deepcopy(verified_answer())]
        self.configure_generation()
        self.submit()
        self.assertIn("The answer ran out of time", self.visible_text())
        self.app.button(key="answer_retry").click().run()
        self.assert_no_exception()
        self.assertEqual(self.service.answer.call_count, 2)
        self.service.search.assert_called_once()

    def test_setup_refusal_retry_uses_the_fixed_settings(self):
        self.configure_generation()
        self.app.text_input(key="threshold").set_value("")
        self.submit()
        self.assertEqual(self.app.session_state["answer_result"]["reason"], "unconfigured_threshold")
        self.app.text_input(key="threshold").set_value("6.5")
        self.app.button(key="answer_retry").click().run()
        self.assert_no_exception()
        self.service.answer.assert_called_once()
        self.service.search.assert_called_once()
        self.assertIn("Microsoft reported $109,433 million", self.visible_text())

    def test_example_with_answers_on_answers_once(self):
        self.configure_generation()
        self.app.button(key="example_3").click().run()
        self.assert_no_exception()
        self.service.search.assert_called_once()
        self.service.answer.assert_called_once()

    def test_startup_failure_explains_how_to_build_the_corpus(self):
        self.service.search.side_effect = DemoError("startup_failed")
        self.submit()
        text = self.visible_text()
        self.assertIn("Search could not complete", text)
        self.assertIn("build_corpus.py", text)
        self.assertNotIn("startup_failed", text)


if __name__ == "__main__":
    unittest.main()
