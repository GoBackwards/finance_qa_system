"""Wording and layout rules behind the demo UI, tested without Streamlit."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from report_search import demo_presentation as view
from report_search.answer_verification import render_verified_answer

ROOT = Path(__file__).resolve().parents[1]


def intel_bundle():
    """Two Intel net-revenue rows (USD millions) from different reports."""
    evidence = {}
    for identifier, text, year in (("E3", "Net revenue | $ 54,228 | $ 63,054", 2023),
                                   ("E9", "Net revenue | $ 79,024 | $ 77,867", 2021)):
        evidence[identifier] = {
            "evidence_id": identifier, "kind": "row", "text": text, "chunk_ids": [f"chunk_{identifier}"],
            "citation": {"document_id": f"INTC_FY{year}_annual_report", "pdf_page": 40,
                         "source_file": f"data/raw/INTC/INTC_FY{year}_annual_report.pdf", "sha256": "a" * 64,
                         "ticker": "INTC", "fiscal_year": year, "bbox": [10, 10, 500, 80]},
            "source_locator": {"table_id": f"INTC_FY{year}:p40:table1", "row_index": 3}}
    return {"source_verified": True, "chunks": [], "evidence": evidence}


def rendered_answer():
    claims = [
        {"text": "Intel's net revenue was $54,228 million in FY2023.",
         "citations": [{"evidence_id": "E3", "quote": "Net revenue | $ 54,228"}]},
        {"text": "Intel's revenue fell 31.4% from FY2021 to FY2023.",
         "citations": [{"evidence_id": "E3", "quote": "Net revenue | $ 54,228"},
                       {"evidence_id": "E9", "quote": "Net revenue | $ 79,024"}],
         "calculation": {"expression": "(a - b) / b * 100", "result": "-31.4", "unit": "%", "operands": [
             {"name": "a", "evidence_id": "E3", "value": "54,228", "label": "Intel net revenue FY2023 (USD millions)"},
             {"name": "b", "evidence_id": "E9", "value": "79,024", "label": "Intel net revenue FY2021 (USD millions)"}]}},
    ]
    verdict = {"answers_question": True, "claim_verdicts": [
        {"claim_index": index, "supported": True, "reason": "The quoted rows state these figures."}
        for index in range(2)]}
    result = render_verified_answer(claims, intel_bundle(), verdict)
    assert result["status"] == "answer", result
    return result


FULL_AUDIT = {
    "confidence_gate": {"allowed": True, "reason": "allowed", "threshold": 6.5, "max_score": 7.5, "candidate_count": 2},
    "source_verified": True, "evidence_count": 12, "context_passages": 5, "context_passages_dropped": 0,
    "generation_called": True, "generator": {"model": "generator-fixture"}, "generation_attempts": 2,
    "first_attempt_reason": "quote_not_in_source", "reference_check_passed": True,
    "claim_grounding": {"ok": True, "reason": None, "claim_index": None, "detail": None},
    "semantic_verifier_called": True, "verifier": {"model": "reviewer-fixture"}}


class RefusalStageTests(unittest.TestCase):
    def test_every_emitted_reason_code_has_a_step(self):
        # Codes written as literals where answers abstain; the legacy answer_with_retrieval path is not in the demo.
        pattern = re.compile(r'(?:_abstain|_draft_failure|_semantic_failure|_grounding_result|_calculation_failure'
                             r'|ModelError)\("([a-z_]+)"|GateDecision\(False, "([a-z_]+)"|"reason": "([a-z_]+)"')
        codes = set()
        for name in ("generation.py", "answer_verification.py", "confidence.py", "llm_client.py"):
            text = (ROOT / "src/report_search" / name).read_text(encoding="utf-8")
            codes |= {next(code for code in match if code) for match in pattern.findall(text)}
        # Codes built at runtime or used as fallbacks.
        codes |= {"below_threshold", "model_http_503", "claim_not_grounded", "unsupported_answer"}
        codes -= {"retrieval_failed"}
        self.assertGreater(len(codes), 40)
        for code in sorted(codes):
            with self.subTest(code=code):
                self.assertNotEqual(view.refusal_stage(code), "service")
                self.assertTrue(view.reason_text(code))
                self.assertTrue(view.headline(code))

    def test_stage_examples_and_unknown_codes(self):
        self.assertEqual(view.refusal_stage("unconfigured_threshold"), "setup")
        self.assertEqual(view.refusal_stage("below_threshold"), "evidence")
        self.assertEqual(view.refusal_stage("claim_number_not_in_evidence"), "grounding")
        self.assertEqual(view.refusal_stage("calculation_result_mismatch"), "quotes")
        self.assertEqual(view.refusal_stage("model_http_503"), "draft")
        self.assertEqual(view.refusal_stage("model_http_503", {"semantic_verifier_called": True}), "review")
        self.assertEqual(view.refusal_stage("deadline_exceeded"), "deadline")
        self.assertEqual(view.refusal_stage("private-sensitive-code"), "service")
        self.assertIn("HTTP 503", view.reason_text("model_http_503"))
        self.assertEqual(view.reason_text("private-sensitive-code"), view.GENERIC_REASON)

    def test_search_errors_never_echo_unknown_codes(self):
        self.assertNotIn("private-sensitive-error", view.search_error_text("private-sensitive-error", "v2"))
        self.assertIn("Search could not complete", view.search_error_text("private-sensitive-error", "v2"))
        self.assertIn("build_corpus.py", view.search_error_text("startup_failed", "v2"))
        v1 = view.search_error_text("startup_failed", "v1")
        self.assertIn("cannot be rebuilt", v1)
        self.assertNotIn("FINANCE_QA_CORPUS=v1", v1)
        self.assertIn("cannot be rebuilt", view.corpus_missing_text("v1"))

    def test_only_settings_problems_are_retryable(self):
        for code in ("unconfigured_threshold", "model_not_configured", "model_http_401", "model_unavailable",
                     "deadline_exceeded"):
            self.assertTrue(view.retryable(code), code)
        for code in ("claim_number_not_in_evidence", "generator_abstained", "below_threshold", "answer_unavailable"):
            self.assertFalse(view.retryable(code), code)

    def test_next_step_uses_scores_and_question_scope(self):
        audit = {"confidence_gate": {"allowed": False, "max_score": 5.1, "threshold": 6.5}}
        self.assertIn("5.10", view.next_step("below_threshold", audit))
        self.assertIn("6.5", view.next_step("below_threshold", audit))
        self.assertIn("not one of the 40", view.next_step("generator_abstained", {}, {"unknown_companies": ["Netflix"]}))
        self.assertIn("not projections", view.next_step("generator_abstained", {}, {"forecast_intent": True}))
        self.assertIn("sidebar", view.next_step("unconfigured_threshold"))
        self.assertEqual(view.format_threshold(-1_000_000), "−1,000,000 (gate open, as in the evaluations)")
        self.assertEqual(view.format_threshold(6.5), "6.5")


class ChecklistTests(unittest.TestCase):
    def test_minimal_audit_never_shows_unconfirmed_steps_as_passed(self):
        answer = {"status": "answer", "citations": [], "audit": {"generation_called": True,
                                                                   "semantic_verifier_called": True}}
        states = {label: state for label, state, _ in view.check_steps(answer)}
        for label in (view.STEPS[0], view.STEPS[1], view.STEPS[3], view.STEPS[4]):
            self.assertEqual(states[label], "not_recorded")

    def test_full_audit_confirms_every_step_with_details(self):
        answer = dict(rendered_answer(), audit=FULL_AUDIT)
        steps = view.check_steps(answer)
        self.assertEqual([state for _, state, _ in steps], ["passed"] * 6)
        details = " ".join(detail for _, _, detail in steps)
        for fragment in ("7.50", "6.5", "12 source units from 5 passages", "corrected once",
                         "2 quotes found word for word", "1 calculations recomputed", "reviewer"):
            self.assertIn(fragment, details)

    def test_open_gate_is_never_counted_as_a_passed_check(self):
        audit = dict(FULL_AUDIT, confidence_gate={"allowed": True, "threshold": -1_000_000, "max_score": 2.83})
        steps = view.check_steps(dict(rendered_answer(), audit=audit))
        self.assertEqual(steps[0][1], "open")
        self.assertIn("not checked", steps[0][2])
        self.assertEqual(sum(state == "passed" for _, state, _ in steps), 5)

    def test_a_deliberate_decline_is_a_stop_not_a_failure(self):
        audit = {key: FULL_AUDIT[key] for key in ("confidence_gate", "source_verified", "evidence_count",
                                                  "context_passages", "generation_called", "generator")}
        states = [state for _, state, _ in view.check_steps({"status": "abstain", "reason": "generator_abstained",
                                                            "audit": audit})]
        self.assertEqual(states, ["passed", "passed", "stopped", "not_run", "not_run", "not_run"])

    def test_refusal_on_the_recheck_after_dropping_passages_explains_both(self):
        audit = {"confidence_gate": {"allowed": True, "threshold": 6.5, "max_score": 7.5},
                 "packed_confidence_gate": {"allowed": False, "threshold": 6.5, "max_score": 5.0},
                 "source_verified": True, "evidence_count": 4, "context_passages": 3}
        answer = {"status": "abstain", "reason": "below_threshold", "audit": audit}
        self.assertEqual([state for _, state, _ in view.check_steps(answer)][:3], ["failed", "passed", "not_run"])
        self.assertIn("5.00 after leaving out passages", view.next_step("below_threshold", audit))

    def test_refusal_marks_its_step_failed_and_later_steps_not_run(self):
        audit = {key: FULL_AUDIT[key] for key in ("confidence_gate", "source_verified", "evidence_count",
                                                  "context_passages", "generation_called", "generator",
                                                  "reference_check_passed")}
        audit["claim_grounding"] = {"ok": False, "reason": "claim_number_not_in_evidence",
                                    "detail": "draft-figure-123"}
        answer = {"status": "abstain", "reason": "claim_number_not_in_evidence", "audit": audit}
        states = [state for _, state, _ in view.check_steps(answer)]
        self.assertEqual(states, ["passed", "passed", "passed", "passed", "failed", "not_run"])
        self.assertNotIn("draft-figure-123", json.dumps(view.check_steps(answer)))
        self.assertEqual(view.check_steps({"status": "abstain", "reason": "model_not_configured"}), [])
        self.assertEqual(view.check_steps({"status": "abstain", "reason": "answer_unavailable"}), [])

    def test_deadline_stops_at_the_stage_that_ran_out(self):
        audit = {"stage_timings": [{"stage": "source_binding", "status": "completed"},
                                   {"stage": "generation", "status": "deadline_exceeded"}]}
        states = [state for _, state, _ in view.check_steps({"status": "abstain", "reason": "deadline_exceeded",
                                                            "audit": audit})]
        self.assertEqual(states[2], "failed")
        self.assertEqual(states[3:], ["not_run"] * 3)


class AnswerLayoutTests(unittest.TestCase):
    def test_answer_parts_on_real_rendered_answer(self):
        answer = rendered_answer()
        parts = view.answer_parts(answer)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["refs"], [1])
        self.assertEqual(parts[1]["refs"], [1, 2])
        self.assertIsNone(parts[0]["calculation"])
        self.assertEqual(parts[1]["calculation"]["claim_index"], 1)
        self.assertNotIn("[^", " ".join(part["text"] for part in parts))
        expression, operands = view.calculation_lines(parts[1]["calculation"], answer["citations"])
        self.assertIn("31&#46;4", expression)
        self.assertTrue(operands[0].endswith("[1]") and operands[1].endswith("[2]"))
        parts = view.answer_parts(dict(answer, audit={"comparison_verdict_added": True}))
        self.assertTrue(parts[-1]["computed"])

    def test_unexpected_answer_shape_falls_back(self):
        answer = rendered_answer()
        self.assertIsNone(view.answer_parts(dict(answer, calculations=[])))
        self.assertIsNone(view.answer_parts(dict(answer, answer="")))
        plain = view.answer_parts({"answer": "Microsoft reported $109,433 million. [^1]"})
        self.assertEqual(plain, [{"text": "Microsoft reported $109,433 million.", "refs": [1],
                                  "calculation": None, "computed": False}])


class OverviewAndScopeTests(unittest.TestCase):
    def test_corpus_overview_reads_the_committed_manifest(self):
        info = view.corpus_overview(ROOT)
        self.assertEqual((info["reports"], info["companies"], info["years"], info["pages"]),
                         (200, 40, (2021, 2025), 23280))
        self.assertEqual(len(info["sectors"]), 5)
        self.assertEqual(info["names"]["MSFT"], "Microsoft")
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "data").mkdir()
            (Path(temporary) / "data/manifest.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(view.corpus_overview(temporary))
            self.assertIsNone(view.corpus_overview(Path(temporary) / "missing"))

    def test_scope_summary_escapes_notes_and_flags_uncovered_questions(self):
        scope = {"tickers": ["MSFT"], "years": [2024], "period_mode": "explicit", "unknown_companies": ["Netflix"],
                 "notes": ["'[ms](http://x)' was interpreted as Microsoft.",
                           "The question asks about future or projected results; so do not estimate figures."],
                 "comparison": False, "forecast_intent": True}
        trace = {"mode": "scoped", "pools": [{"ticker": "MSFT", "fiscal_years": [2024], "top_k": 20, "purpose": "scope"},
                                             {"ticker": "MSFT", "fiscal_years": [2025], "top_k": 4, "purpose": "latest"}]}
        summary = view.scope_summary(scope, trace, {"MSFT": "Microsoft"}, 200)
        self.assertIn("Microsoft", summary["companies"])
        self.assertEqual(summary["period"], "FY2024")
        self.assertIn("MSFT FY2024", summary["searched"])
        self.assertIn("latest MSFT FY2025", summary["searched"])
        self.assertIn(("Not covered: Netflix", "red"), summary["badges"])
        self.assertIn(("Forecast question", "orange"), summary["badges"])
        self.assertTrue(summary["warn"])
        self.assertNotIn("[ms](", summary["notes"][0])
        self.assertEqual(summary["notes"][1], view.FORECAST_NOTE)
        empty = view.scope_summary(None, None, {}, 200)
        self.assertEqual((empty["period"], empty["searched"]), ("Scope not resolved", "Searched: not recorded"))
        self.assertFalse(empty["warn"])
        self.assertEqual(view.scope_summary({"tickers": []}, {"mode": "global"}, {}, 200)["searched"],
                         "Searched: all 200 reports")

    def test_examples_are_not_evaluation_questions(self):
        questions = set()

        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("query", "question") and isinstance(item, str):
                        questions.add(item.strip().lower())
                    else:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        for path in (ROOT / "data/evaluation").glob("*.json"):
            if path.name != "demo_examples_check.json":  # the record of these examples' own demo run
                collect(json.loads(path.read_text(encoding="utf-8")))
        self.assertGreater(len(questions), 300)
        for _, _, examples in view.EXAMPLES:
            for example in examples:
                self.assertNotIn(example.strip().lower(), questions)

    def test_evidence_breakdown_counts_added_context(self):
        selected = [{"chunk_id": "a"}, {"chunk_id": "b", "context_reason": "same_table"}]
        self.assertEqual(view.evidence_breakdown(selected, {"suppressed_duplicates": ["x"]}),
                         (1, 1, "1 best match + 1 added for context · 1 near-duplicates from other report years "
                                "skipped"))


if __name__ == "__main__":
    unittest.main()
