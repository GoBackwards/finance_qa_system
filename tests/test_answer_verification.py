"""Adversarial reference validation and independent semantic gate contracts."""

import copy
import html
import json
import unittest

from report_search.answer_verification import (
    ABSTENTION, MAX_CLAIMS, MAX_CLAIM_CHARS, MAX_JSON_CHARS,
    render_verified_answer, validate_draft, validate_semantic_verdict, verify_calculation,
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


def intel_bundle():
    """Intel net revenue rows from the FY2023 and FY2021 reports (USD millions)."""
    source = bundle()
    template = source["evidence"].pop("E1")
    for identifier, text, year, row in (("E3", "Net revenue | $ 54,228 | $ 63,054", 2023, 3),
                                        ("E9", "Net revenue | $ 79,024 | $ 77,867", 2021, 9),
                                        ("E5", "Operating income (loss) | (1,234) | \u22125", 2023, 5)):
        entry = copy.deepcopy(template)
        entry.update(evidence_id=identifier, text=text, kind="row", chunk_ids=[f"chunk_{identifier}"],
                     source_locator={"table_id": f"INTC_FY{year}:p40:table1", "row_index": row})
        entry["citation"].update(ticker="INTC", fiscal_year=year, document_id=f"INTC_FY{year}_annual_report")
        source["evidence"][identifier] = entry
    return source


def growth_claim(text="Intel's revenue fell 31.4% from FY2021 to FY2023.", **changes):
    calculation = {"expression": "(a - b) / b * 100", "result": "-31.4", "unit": "%", "operands": [
        {"name": "a", "evidence_id": "E3", "value": "54,228", "label": "Intel net revenue FY2023 (USD millions)"},
        {"name": "b", "evidence_id": "E9", "value": "79,024", "label": "Intel net revenue FY2021 (USD millions)"}]}
    calculation.update(changes)
    return {"text": text, "citations": [{"evidence_id": "E3", "quote": "Net revenue | $ 54,228"},
                                        {"evidence_id": "E9", "quote": "Net revenue | $ 79,024"}],
            "calculation": calculation}


class CalculationTests(unittest.TestCase):
    def reason(self, claim):
        return validate_draft({"status": "answer", "claims": [claim]}, intel_bundle())["reason"]

    def test_application_recomputes_growth_from_cited_operands(self):
        claim = growth_claim()
        checked = validate_draft({"status": "answer", "claims": [claim]}, intel_bundle())
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(checked["claims"][0]["calculation"], claim["calculation"])
        result = verify_calculation(claim, intel_bundle()["evidence"])
        self.assertTrue(result["ok"])
        self.assertIsNone(result["reason"])
        self.assertTrue(result["computed"].startswith("-31.37780"))
        self.assertEqual(result["expression_display"], "(54,228 \u2212 79,024) \u00f7 79,024 \u00d7 100")

    def test_result_must_match_recomputation_at_its_written_precision(self):
        for result, text, expected in (("-31.38", "fell 31.38%", None), ("-31", "fell 31%", None),
                                       ("-31.378", "fell 31.378%", None),
                                       ("-31.37", "fell 31.37%", "calculation_result_mismatch"),
                                       ("-31.5", "fell 31.5%", "calculation_result_mismatch"),
                                       ("31.4", "fell 31.4%", "calculation_result_mismatch")):
            with self.subTest(result=result):
                self.assertEqual(self.reason(growth_claim(f"Intel's revenue {text}.", result=result)), expected)

    def test_result_must_be_written_in_the_claim(self):
        for text in ("Intel's revenue fell sharply.", "Intel's revenue fell 131.4%.", "Intel's revenue fell 31.45%.",
                     "Intel's revenue fell 31.40%."):
            with self.subTest(text=text):
                self.assertEqual(self.reason(growth_claim(text)), "calculation_result_not_in_claim")
        difference = growth_claim("Intel's revenue was $24,796 million lower in FY2023 than in FY2021.",
                                  expression="b - a", result="24796", unit="USD millions")
        self.assertIsNone(self.reason(difference))

    def test_operands_must_be_cited_and_copied_from_their_source_unit(self):
        claim = growth_claim()
        claim["calculation"]["operands"][1]["evidence_id"] = "E5"
        self.assertEqual(self.reason(claim), "calculation_operand_not_cited")
        for value, expected in (("54,229", "calculation_operand_not_in_source"),
                                ("228", "calculation_operand_not_in_source"),
                                ("54228", "calculation_operand_not_in_source"),
                                ("63,054", "calculation_result_mismatch"),
                                ("$ 54,228", None)):
            with self.subTest(value=value):
                claim = growth_claim()
                claim["calculation"]["operands"][0]["value"] = value
                self.assertEqual(self.reason(claim), expected)

    def test_operands_keep_the_sign_they_have_in_the_source(self):
        def operand_reason(text, kind, value):
            source = intel_bundle()["evidence"]
            entry = source["E5"]
            entry.update(text=text, kind=kind)
            if kind == "prose":
                entry["source_locator"] = {"element_id": "p40_b1", "char_start": 0, "char_end": len(text)}
            claim = {"text": "The change was 1,801 in FY2023.",
                     "citations": [{"evidence_id": "E5", "quote": text}],
                     "calculation": {"expression": "a - b", "result": "1801", "unit": "", "operands": [
                         {"name": "a", "evidence_id": "E5", "value": value, "label": "Figure FY2023"},
                         {"name": "b", "evidence_id": "E5", "value": "567", "label": "Figure FY2022"}]}}
            return verify_calculation(claim, source)["reason"]

        # The claim states a - b = 2,368 - 567 = 1,801, so only an unsigned 2,368 can pass.
        cases = (
            ("Operating income (loss) | (2,368) | 567", "row", "2,368", "calculation_operand_not_in_source"),
            ("Operating income (loss) | $ (2,368) | 567", "row", "2,368", "calculation_operand_not_in_source"),
            ("Operating income (loss) | -2,368 | 567", "row", "2,368", "calculation_operand_not_in_source"),
            ("Operating income (loss) | (2,368) | 567", "row", "(2,368)", "calculation_result_mismatch"),
            ("Operating income | (2,368) | 2,368 | 567", "row", "2,368", None),
            ("Senior notes (2,368 tranche) | 1 | 567", "row", "2,368", None),
            ("Volume/mix | - | 2,368 | 567", "row", "2,368", None),
            ("Net revenues increased $567 million (2,368 units) in 2022-2023.", "prose", "2,368", None),
            ("Operating income fell from 567 to -2,368.", "prose", "2,368", "calculation_operand_not_in_source"),
        )
        for text, kind, value, expected in cases:
            with self.subTest(text=text, value=value):
                self.assertEqual(operand_reason(text, kind, value), expected)

    def test_parenthesized_negatives_and_unicode_minus_are_parsed(self):
        claim = {"text": "Operating loss narrowed by 1,229 in FY2023.",
                 "citations": [{"evidence_id": "E5", "quote": "Operating income (loss) | (1,234)"}],
                 "calculation": {"expression": "-(a - b)", "result": "1229", "unit": "USD millions", "operands": [
                     {"name": "a", "evidence_id": "E5", "value": "(1,234)", "label": "Operating loss FY2023"},
                     {"name": "b", "evidence_id": "E5", "value": "\u22125", "label": "Operating loss FY2022"}]}}
        self.assertIsNone(self.reason(claim))
        self.assertEqual(verify_calculation(claim, intel_bundle()["evidence"])["expression_display"],
                         "\u2212((1,234) \u2212 \u22125)")

    def test_expression_whitelist_rejects_anything_but_arithmetic_on_operands(self):
        for expression in ("a ** 2 + b", "abs(a) + b", "a.real + b", "a + b + c", "a", "a + b + 'x'",
                           "a + b + True", "a / (b - b)", "a +", "a // b", "a % b", "a < b", "[a, b]",
                           "a if b else b", "__import__('os').system('x') + a + b", "1e999 + a + b",
                           "a + b + 1j", "(lambda: a)() + b", "a; b"):
            with self.subTest(expression=expression):
                self.assertEqual(self.reason(growth_claim(expression=expression)), "invalid_calculation_expression")
        self.assertIsNone(self.reason(growth_claim(expression="((a) - b) / +b * 100.0")))

    def test_calculation_schema_is_exact(self):
        operand = growth_claim()["calculation"]["operands"][0]
        mutations = [
            lambda c: c.update(extra=1), lambda c: c.pop("unit"), lambda c: c.update(operands=[]),
            lambda c: c.update(operands=[dict(operand, name=f"x{i}") for i in range(9)]),
            lambda c: c.update(operands="a"), lambda c: c["operands"][0].update(name="A"),
            lambda c: c["operands"][0].update(name="1a"), lambda c: c["operands"][0].update(name="b"),
            lambda c: c["operands"][0].update(name="if"), lambda c: c["operands"][0].update(name="a" * 17),
            lambda c: c["operands"][0].update(label=""), lambda c: c["operands"][0].update(label="x" * 121),
            lambda c: c["operands"][0].update(value="n/a"), lambda c: c["operands"][0].update(value="1" * 65),
            lambda c: c["operands"][0].update(evidence_id=3), lambda c: c["operands"][0].pop("label"),
            lambda c: c["operands"][0].update(source="E3"), lambda c: c.update(result="1e3"),
            lambda c: c.update(result="31.4%"), lambda c: c.update(result="31."), lambda c: c.update(result=-31.4),
            lambda c: c.update(unit="x" * 41), lambda c: c.update(unit="%\u202e"), lambda c: c.update(unit=None),
            lambda c: c.update(expression="a" * 201), lambda c: c.update(expression=" "),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                claim = growth_claim()
                mutation(claim["calculation"])
                self.assertEqual(self.reason(claim), "invalid_calculation_schema")
        self.assertEqual(verify_calculation(None, intel_bundle()["evidence"])["reason"], "invalid_calculation_schema")

    def test_claim_keys_stay_exact(self):
        claim = growth_claim()
        claim["calc"] = claim.pop("calculation")
        self.assertEqual(self.reason(claim), "invalid_claim_schema")
        claim = growth_claim()
        claim["calculation"] = None
        self.assertEqual(self.reason(claim), "invalid_calculation_schema")

    def test_renderer_shows_recomputed_calculation_with_operand_references(self):
        claim = growth_claim()
        result = render_verified_answer([claim], intel_bundle(), verdict())
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["calculations"], [{
            "claim_index": 0, "expression": "(a - b) / b * 100",
            "expression_display": "(54,228 \u2212 79,024) \u00f7 79,024 \u00d7 100",
            "operands": claim["calculation"]["operands"], "result": "-31.4", "unit": "%"}])
        paragraphs = result["answer"].split("\n\n")
        self.assertEqual(html.unescape(paragraphs[1]),
                         "Calculated by the application: (54,228 \u2212 79,024) \u00f7 79,024 \u00d7 100 = \u221231.4% "
                         "(operands: a = Intel net revenue FY2023 (USD millions), 54,228 [^1]; "
                         "b = Intel net revenue FY2021 (USD millions), 79,024 [^2])")
        self.assertIn("[^1]", paragraphs[1])
        self.assertNotIn("(", paragraphs[1])

    def test_renderer_escapes_model_supplied_operand_labels(self):
        claim = growth_claim()
        claim["calculation"]["operands"][0]["label"] = "<script>x</script> [link](https://evil.example)"
        result = render_verified_answer([claim], intel_bundle(), verdict())
        self.assertEqual(result["status"], "answer")
        for raw in ("<script>", "[link]", "https://"):
            self.assertNotIn(raw, result["answer"])

    def test_calculations_list_is_empty_without_calculations_or_on_abstention(self):
        self.assertEqual(render_verified_answer(draft()["claims"], bundle(), verdict())["calculations"], [])
        wrong = growth_claim(result="-32.0", text="Intel's revenue fell 32.0%.")
        result = render_verified_answer([wrong], intel_bundle(), verdict())
        self.assertEqual((result["status"], result["calculations"], result["reason"]),
                         ("abstain", [], "calculation_result_mismatch"))



class CalculationHardeningTests(unittest.TestCase):
    def reason(self, claim):
        return validate_draft({"status": "answer", "claims": [claim]}, intel_bundle())["reason"]

    def test_only_scale_constants_are_allowed(self):
        injected = growth_claim("Intel's revenue fell 12.5%.", expression="a - a + b * 0 - 12.5", result="-12.5")
        self.assertEqual(self.reason(injected), "invalid_calculation_expression")
        scaled = growth_claim("Intel's revenue was $54.228 billion in FY2023.", expression="a / 1000",
                              result="54.228", unit="USD billions",
                              operands=[growth_claim()["calculation"]["operands"][0]])
        self.assertIsNone(self.reason(scaled))

    def test_negative_result_must_read_as_a_decrease(self):
        self.assertEqual(self.reason(growth_claim("Intel's revenue grew 31.4% from FY2021 to FY2023.")),
                         "calculation_direction_mismatch")
        self.assertIsNone(self.reason(growth_claim("Intel's revenue changed by -31.4% from FY2021 to FY2023.")))
        positive = growth_claim("Intel's revenue changed by -45.7%.", expression="(b - a) / a * 100", result="45.7")
        self.assertEqual(self.reason(positive), "calculation_direction_mismatch")

    def test_results_with_separators_are_accepted(self):
        difference = growth_claim("Intel's revenue was $24,796 million lower in FY2023 than in FY2021.",
                                  expression="a - b", result="-24,796", unit="USD millions")
        self.assertIsNone(self.reason(difference))



class VerdictDoubtTests(unittest.TestCase):
    def test_supported_verdict_whose_reason_expresses_doubt_is_rejected(self):
        claims = [{"text": "Revenue was $10 million.", "citations": [{"evidence_id": "E1", "quote": "x"}]}]
        for reason in ("The payment is not in the cited evidence but implied by the context.",
                       "This is a projection for 2025, not for 2027; however it can reasonably be interpreted as one.",
                       "The source does not specify the period."):
            verdict = {"answers_question": True,
                       "claim_verdicts": [{"claim_index": 0, "supported": True, "reason": reason}]}
            with self.subTest(reason=reason):
                self.assertEqual(validate_semantic_verdict(verdict, claims)["reason"],
                                 "verifier_reason_contradicts_verdict")
        for reason in ("E1 states revenue of $10 million.",
                       "E1 confirms a $7.2 billion purchase price including assumed debt."):  # live C02
            verdict = {"answers_question": True,
                       "claim_verdicts": [{"claim_index": 0, "supported": True, "reason": reason}]}
            self.assertTrue(validate_semantic_verdict(verdict, claims)["ok"], reason)


if __name__ == "__main__":
    unittest.main()
