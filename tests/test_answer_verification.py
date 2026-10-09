"""Adversarial reference validation and independent semantic gate contracts."""

import copy
import json
import unittest

from report_search.answer_verification import (
    ABSTENTION, MAX_CLAIMS, MAX_CLAIM_CHARS, MAX_JSON_CHARS,
    render_verified_answer, validate_draft, validate_semantic_verdict,
)


def bundle():
    text = "Consolidated revenue was $10 million in FY2025. Revenue increased by 5%."
    return {"source_verified": True, "chunks": [], "evidence": {"E1": {
        "evidence_id": "E1", "kind": "prose", "text": text,
        "citation": {"document_id": "DEMO_FY2025_annual_report", "pdf_page": 17,
                     "source_file": "data/raw/DEMO/DEMO_FY2025_annual_report.pdf",
                     "sha256": "a" * 64, "ticker": "DEMO", "fiscal_year": 2025,
                     "bbox": [10, 10, 500, 80]},
        "source_locator": {"element_id": "p17_b3", "char_start": 50,
                           "char_end": 50 + len(text)},
        "chunk_ids": ["chunk_one", "chunk_overlap"],
    }}}


def draft(text="Consolidated revenue was $10 million in FY2025."):
    return {"status": "answer", "claims": [{"text": text, "citations": [
        {"evidence_id": "E1", "quote": "revenue was $10 million in FY2025."}]}]}


def verdict(count=1):
    return {"answers_question": True, "claim_verdicts": [
        {"claim_index": index, "supported": True,
         "reason": "The quoted paragraph explicitly states this revenue and fiscal year."}
        for index in range(count)]}


class AnswerVerificationTests(unittest.TestCase):
    def test_strict_reference_verification_preserves_source_and_input(self):
        data, source = draft(), bundle()
        before = copy.deepcopy((data, source))
        checked = validate_draft(json.dumps(data), source)
        self.assertTrue(checked["ok"])
        self.assertTrue(checked["reference_verified"])
        self.assertNotIn("semantic_supported", checked)
        checked["claims"][0]["text"] = "Changing the output does not change the input."
        self.assertEqual((data, source), before)

    def test_unknown_or_malicious_evidence_ids_are_rejected(self):
        for identifier in ("E2", "E0", "E01", "e1", "E1<script>", None, 1, ["E1"]):
            with self.subTest(identifier=identifier):
                data = draft()
                data["claims"][0]["citations"][0]["evidence_id"] = identifier
                result = validate_draft(data, bundle())
                self.assertEqual(result["reason"], "unknown_evidence_id")
                self.assertEqual(result["claims"], [])

    def test_quote_matching_does_not_normalize_altered_evidence(self):
        for quote in ("Revenue was $10 million in FY2025.",
                      "revenue  was $10 million in FY2025.",
                      "revenue was $100 million in FY2025.", " ", "", None):
            with self.subTest(quote=quote):
                data = draft()
                data["claims"][0]["citations"][0]["quote"] = quote
                self.assertFalse(validate_draft(data, bundle())["ok"])

    def test_uncited_claim_and_duplicate_citations_fail(self):
        for citations in ([], None, "E1"):
            data = draft()
            data["claims"][0]["citations"] = citations
            self.assertEqual(validate_draft(data, bundle())["reason"], "missing_or_excess_citations")
        data = draft()
        data["claims"][0]["citations"] *= 2
        self.assertEqual(validate_draft(data, bundle())["reason"], "duplicate_citation")

    def test_model_cannot_supply_source_metadata_or_add_unverified_prose(self):
        changed = draft()
        changed["claims"][0]["citations"][0]["pdf_page"] = 999
        self.assertEqual(validate_draft(changed, bundle())["reason"], "invalid_citation_schema")
        changed = draft()
        changed["answer"] = "Uncited misleading summary."
        self.assertEqual(validate_draft(changed, bundle())["reason"], "invalid_draft_schema")
        changed = draft()
        changed["claims"][0]["source"] = "Fake source"
        self.assertEqual(validate_draft(changed, bundle())["reason"], "invalid_claim_schema")

    def test_json_cannot_be_repaired_or_silently_accept_duplicate_keys(self):
        invalid = ("```json\n" + json.dumps(draft()) + "\n```", "not JSON", "[]", "null",
                   '{"status":"abstain","status":"answer","claims":[]}',
                   '{"status":"answer","claims":NaN}',
                   json.dumps(draft()).replace('"evidence_id": "E1"',
                                             '"evidence_id": "E2", "evidence_id": "E1"'),
                   {"status": "answer", "claims": [], "x": float("inf")},
                   "{" * 2_000, None)
        for data in invalid:
            with self.subTest(data=str(data)[:100]):
                result = validate_draft(data, bundle())
                self.assertFalse(result["ok"])
                self.assertEqual(result["claims"], [])

    def test_bounded_outputs_empty_claims_and_abstention_are_fail_closed(self):
        for data in (draft("x" * (MAX_CLAIM_CHARS + 1)), draft(" "),
                     {"status": "answer", "claims": []},
                     {"status": "answer", "claims": draft()["claims"] * (MAX_CLAIMS + 1)},
                     " " * (MAX_JSON_CHARS + 1)):
            self.assertFalse(validate_draft(data, bundle())["ok"])
        result = validate_draft({"status": "abstain", "claims": []}, bundle())
        self.assertEqual(result["reason"], "generator_abstained")
        result = validate_draft({"status": "abstain", "claims": draft()["claims"]}, bundle())
        self.assertEqual(result["reason"], "invalid_draft_schema")

    def test_bundle_requires_verified_source_and_valid_original_locators(self):
        mutations = [lambda data: data.update(source_verified=False),
                     lambda data: data.update(source_verified=1),
                     lambda data: data.update(evidence=[]),
                     lambda data: data["evidence"]["E1"].update(kind=[]),
                     lambda data: data["evidence"]["E1"]["citation"].update(pdf_page=0),
                     lambda data: data["evidence"]["E1"]["citation"].update(pdf_page=True),
                     lambda data: data["evidence"]["E1"]["source_locator"].update(char_end=1),
                     lambda data: data["evidence"]["E1"]["source_locator"].pop("element_id"),
                     lambda data: data["evidence"]["E1"].update(chunk_ids=[])]
        for mutation in mutations:
            source = bundle()
            mutation(source)
            self.assertFalse(validate_draft(draft(), source)["ok"])

    def test_true_quote_cannot_bypass_semantic_rejection_of_false_claim(self):
        data = draft("Revenue was $100 million in FY2025.")
        reference = validate_draft(data, bundle())
        self.assertTrue(reference["reference_verified"])
        rejected = verdict()
        rejected["claim_verdicts"][0].update(supported=False, reason="The source says $10 million, not $100 million.")
        semantic = validate_semantic_verdict(rejected, reference["claims"])
        self.assertFalse(semantic["semantic_supported"])
        rendered = render_verified_answer(reference["claims"], bundle(), rejected)
        self.assertEqual(rendered["answer"], ABSTENTION)
        self.assertEqual(rendered["citations"], [])
        self.assertEqual(rendered["reason"], "unsupported_claim")
        self.assertTrue(rendered["reference_verified"])
        self.assertNotIn("$100", json.dumps(rendered))

    def test_semantic_verifier_must_cover_every_claim_once(self):
        claims = draft()["claims"] * 2
        for judgments in ([], verdict()["claim_verdicts"],
                          verdict()["claim_verdicts"] * 2,
                          verdict(3)["claim_verdicts"],
                          [dict(verdict()["claim_verdicts"][0], claim_index=2),
                           verdict()["claim_verdicts"][0]]):
            self.assertFalse(validate_semantic_verdict(
                {"answers_question": True, "claim_verdicts": judgments}, claims)["ok"])
        correct = verdict(2)
        correct["claim_verdicts"].reverse()
        result = validate_semantic_verdict(correct, claims)
        self.assertTrue(result["ok"])
        self.assertEqual([item["claim_index"] for item in result["claim_verdicts"]], [0, 1])

    def test_verifier_wrong_types_extra_fields_and_missing_question_support_fail(self):
        changes = [lambda data: data.update(answers_question="true"),
                   lambda data: data.update(answers_question=False),
                   lambda data: data.update(extra="untrusted"),
                   lambda data: data["claim_verdicts"][0].update(supported="true"),
                   lambda data: data["claim_verdicts"][0].update(supported=1),
                   lambda data: data["claim_verdicts"][0].update(claim_index=True),
                   lambda data: data["claim_verdicts"][0].update(reason=""),
                   lambda data: data["claim_verdicts"][0].update(extra="untrusted")]
        for change in changes:
            data = verdict()
            change(data)
            self.assertFalse(validate_semantic_verdict(data, draft()["claims"])["ok"])
        for data in (None, "{bad json}", "[]", {}, {"claim_verdicts": []}):
            self.assertFalse(validate_semantic_verdict(data, draft()["claims"])["ok"])

    def test_renderer_rechecks_both_gates_and_never_returns_failed_draft(self):
        data = draft()
        data["claims"][0]["citations"][0]["quote"] = "Invented quotation."
        result = render_verified_answer(data["claims"], bundle(), verdict())
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertEqual(result["citations"], [])
        self.assertFalse(result["reference_verified"])
        result = render_verified_answer(draft()["claims"], bundle(), None)
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertFalse(result["semantic_supported"])

    def test_renderer_keeps_whole_source_paragraph_quote_and_original_location(self):
        source, data = bundle(), draft()
        result = render_verified_answer(data["claims"], source, verdict())
        self.assertEqual(result["status"], "answer")
        self.assertIn("[^1]", result["answer"])
        self.assertIn("PDF page 17", result["answer"])
        self.assertIn("paragraph p17", result["answer"])
        reference = result["citations"][0]
        self.assertEqual(reference["source_text"], source["evidence"]["E1"]["text"])
        self.assertEqual(reference["source_locator"], source["evidence"]["E1"]["source_locator"])
        self.assertEqual(reference["citation"], source["evidence"]["E1"]["citation"])
        self.assertEqual(reference["source_text"][reference["quote_char_start"]:reference["quote_char_end"]],
                         data["claims"][0]["citations"][0]["quote"])
        reference["citation"]["pdf_page"] = 1
        self.assertEqual(source["evidence"]["E1"]["citation"]["pdf_page"], 17)

    def test_shared_reference_is_reused_across_claims(self):
        data = draft()
        data["claims"].append(copy.deepcopy(data["claims"][0]))
        result = render_verified_answer(data["claims"], bundle(), verdict(2))
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["answer"].count("[^1]"), 3)

    def test_all_table_binding_kinds_render_original_locators(self):
        for kind, locator, expected in (
            ("row", {"table_id": "p17_t1", "row_index": 4}, "source row index 4"),
            ("table_source", {"table_id": "p17_t1", "char_start": 0, "char_end": 68}, "source text"),
            ("table_context", {"table_id": "p17_t1", "role": "header", "context_index": 0}, "context index 0"),
        ):
            with self.subTest(kind=kind):
                source = bundle()
                source["evidence"]["E1"].update(kind=kind, source_locator=locator)
                result = render_verified_answer(draft()["claims"], source, verdict())
                self.assertEqual(result["status"], "answer")
                self.assertIn(expected, result["answer"])
                self.assertEqual(result["citations"][0]["source_locator"], locator)

    def test_markdown_html_urls_and_control_injection_cannot_create_active_markup(self):
        malicious = '<script>alert(1)</script> [click](https://evil.example/) ![image](x) **bold**'
        source = bundle()
        source["evidence"]["E1"]["citation"]["document_id"] = malicious
        data = draft(malicious)
        result = render_verified_answer(data["claims"], source, verdict())
        self.assertEqual(result["status"], "answer")
        for raw in ("<script>", "[click]", "https://", "![image]", "**bold**"):
            self.assertNotIn(raw, result["answer"])
        self.assertIn("&#60;script&#62;", result["answer"])
        for claim in ("Revenue is $100 [1].", "Revenue is $100 [E2].", "Revenue is $100 [^3].",
                      "Revenue is $100 【99†source】.", "Revenue is $100\u202e."):
            self.assertFalse(validate_draft(draft(claim), bundle())["ok"])


if __name__ == "__main__":
    unittest.main()
