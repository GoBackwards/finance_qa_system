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


if __name__ == "__main__":
    unittest.main()
