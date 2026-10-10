"""Citation canonicalization: exact source spans for retyped or misattributed quotes.

Fixtures mirror real gpt-4.1-mini drafts from the live usage audit
(data/experiments/live_usage/baseline.jsonl): A09, A12, C04, C05, D08, D13.
"""

import copy
import json
import unittest

from report_search.answer_verification import canonicalize_citations, validate_draft


def unit(identifier, text, *, kind="prose", chunks=("c1",), ticker="DEMO", year=2025, page=7):
    locators = {
        "prose": {"element_id": f"p{page}_{identifier}", "char_start": 0, "char_end": len(text)},
        "table_source": {"table_id": f"p{page}_t1", "char_start": 0, "char_end": len(text)},
        "row": {"table_id": f"p{page}_t1", "row_index": int(identifier[1:])},
        "table_context": {"table_id": f"p{page}_t1", "role": "caption"},
    }
    return {"evidence_id": identifier, "kind": kind, "text": text,
            "citation": {"document_id": f"{ticker}_FY{year}_annual_report",
                         "source_file": f"{ticker}_FY{year}_annual_report.pdf", "sha256": "a" * 64,
                         "ticker": ticker, "fiscal_year": year, "pdf_page": page, "bbox": [0, 0, 1, 1]},
            "source_locator": locators[kind], "chunk_ids": list(chunks)}


def make_bundle(*units):
    chunks = {}
    for entry in units:
        for chunk_id in entry["chunk_ids"]:
            chunk = chunks.setdefault(chunk_id, {
                "chunk_id": chunk_id, "text": "", "evidence_ids": [], "kind": "prose", "company": "Demo",
                "ticker": entry["citation"]["ticker"], "fiscal_year": entry["citation"]["fiscal_year"]})
            chunk["evidence_ids"].append(entry["evidence_id"])
            chunk["text"] += entry["text"] + "\n"
    return {"source_verified": True, "evidence": {entry["evidence_id"]: entry for entry in units},
            "chunks": list(chunks.values())}


def draft(*claims):
    return {"status": "answer", "claims": [
        {"text": text, "citations": [{"evidence_id": identifier, "quote": quote}
                                     for identifier, quote in citations]}
        for text, citations in claims]}


def repair(data, source):
    repaired, repairs = canonicalize_citations(data, source)
    return repaired, repairs, validate_draft(repaired, source)


def kinds(repairs):
    return [(item["claim_index"], item["citation_index"], item["kind"],
             item["from_evidence_id"], item["to_evidence_id"]) for item in repairs]


# A09: ADBE FY2024 Digital Media table, one chunk.
A09 = make_bundle(
    unit("E1", "Revenue by major offerings in our Digital Media reportable segment for fiscal 2024, 2023 and "
               "2022 were as follows:\n(in millions)\n2024\n2023\n2022\nCreative Cloud\n$\n12,682 $\n11,517 $\n10,459",
         kind="table_context", ticker="ADBE", year=2024),
    unit("E2", "Document Cloud | 3,182 | 2,699 | 2,383", kind="row", ticker="ADBE", year=2024),
    unit("E3", "Total Digital Media revenue | $ 15,864 | $ 14,216 | $ 12,842", kind="row", ticker="ADBE", year=2024),
)

# A12: MU balance sheet rows (FY2025 report) and a label-free debt row (FY2024 report).
A12 = make_bundle(
    unit("E1", "(In millions, except par value amounts)\nAugust 28,\nAugust 29,\nAs of\n2025\n2024\nAssets",
         kind="table_context", ticker="MU"),
    unit("E18", "Current debt | 560 | 431", kind="row", ticker="MU"),
    unit("E21", "Long-term debt | 14,017 | 12,966", kind="row", ticker="MU"),
    unit("E72", " |  |  |  |  | $ | 13,491 | $ | 431 | $ | 12,966 | $ | 13,397 |  | $ | 13,465 | $ | 278 | "
                "$ | 13,052 | $ | 13,330", kind="row", chunks=("c2",), ticker="MU", year=2024),
)

# D08: COST FY2024 membership-fee table extracted as one prose atom per fragment.
D08 = make_bundle(*[
    unit(identifier, text, ticker="COST", year=2024, chunks=("c59",)) for identifier, text in (
        ("E72", "4,828"), ("E73", "$"), ("E74", "4,580"), ("E75", "$"),
        ("E76", "4,224 Membership fees increase . . . . . . . . . . . . . . . . . . . . ."),
        ("E77", "5 %"),
        ("E80", "Membership fee revenue increased 5% in 2024, driven by new member sign-ups and upgrades "
                "to Executive Membership."))])


class TypographyRepairTests(unittest.TestCase):
    def test_nbsp_typed_as_space_is_repaired_to_exact_source_text_c04(self):
        source_text = ("Inclusive of fiscal year 2025 charges, the Company estimates that it will incur approximately "
                       "$1.0\u00a0billion in total pre-tax restructuring charges related to the \u201cBack to Starbucks\u201d "
                       "restructuring plan, in addition to the $137\u00a0million incurred. During the fiscal year ended "
                       "September\u00a028, 2025, 627 stores were closed and approximately $892.0\u00a0million was recorded.")
        source = make_bundle(unit("E4", source_text, ticker="SBUX"))
        quote = source_text.replace("\u00a0", " ")
        data = draft(("Starbucks expects about $1.0 billion of restructuring charges.", [("E4", quote)]))
        repaired, repairs, checked = repair(data, source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"], source_text)
        self.assertEqual(kinds(repairs), [(0, 0, "typography", "E4", "E4")])
        self.assertEqual(validate_draft(data, source)["reason"], "quote_not_in_source")

    def test_ligatures_typed_as_letters_are_repaired_c05(self):
        source_text = ("The e\ufb00ectiveness of our internal control over \ufb01nancial reporting, as of September 30, "
                       "2025, has been audited by Deloitte & Touche LLP.")
        source = make_bundle(unit("E12", source_text, ticker="ROK"))
        quote = source_text.replace("\ufb00", "ff").replace("\ufb01", "fi")
        repaired, repairs, checked = repair(draft(("Deloitte & Touche LLP audits Rockwell.", [("E12", quote)])),
                                            source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"], source_text)
        self.assertEqual(repairs[0]["kind"], "typography")

    def test_nul_emitted_for_nbsp_is_repaired_instead_of_invalid_quote_d13(self):
        source_text = ("The dollar amount of backlog believed to be firm was approximately $30.4 billion at "
                       "December 31, 2022 and $23.1 billion at December\u00a031, 2021.")
        source = make_bundle(unit("E5", source_text, ticker="CAT", year=2022))
        data = draft(("Caterpillar backlog was $30.4 billion at December 31, 2022.",
                      [("E5", source_text.replace("\u00a0", "\x00"))]))
        self.assertEqual(validate_draft(data, source)["reason"], "invalid_quote")
        repaired, _, checked = repair(data, source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"], source_text)

    def test_newlines_curly_quotes_and_dashes_map_back_to_original_span(self):
        source = make_bundle(unit("E1", "We announced our \u201cBack to Starbucks\u201d plan \u2013 a turnaround.\n"
                                        "Margins\u2014excluding charges\u2014improved."))
        data = draft(("A plan was announced.", [("E1", 'our "Back to Starbucks" plan - a turnaround. Margins-excluding')]))
        repaired, _, checked = repair(data, source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"],
                         "our \u201cBack to Starbucks\u201d plan \u2013 a turnaround.\nMargins\u2014excluding")

    def test_table_context_line_breaks_are_typography_a09(self):
        data = draft(("Creative Cloud revenue was $12,682 million in fiscal 2024.",
                      [("E1", "Creative Cloud\n$ 12,682")]))
        repaired, repairs, checked = repair(data, A09)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"], "Creative Cloud\n$\n12,682")
        self.assertEqual(kinds(repairs), [(0, 0, "typography", "E1", "E1")])

    def test_exact_quotes_are_untouched_and_inputs_are_not_mutated(self):
        data = draft(("Document Cloud revenue was $3,182 million.", [("E2", "Document Cloud | 3,182")]))
        before = copy.deepcopy((data, A09))
        repaired, repairs = canonicalize_citations(data, A09)
        self.assertEqual(repairs, [])
        self.assertEqual(repaired, data)
        self.assertIsNot(repaired, data)
        repaired["claims"][0]["text"] = "changed"
        self.assertEqual((data, A09), before)

    def test_case_and_wording_differences_are_never_repaired(self):
        source = make_bundle(unit("E1", "Revenue was $10\u00a0million in FY2025."))
        for quote in ("revenue was $10 million in FY2025.", "Revenue was $100 million", "Revenue  is $10 million"):
            with self.subTest(quote=quote):
                repaired, repairs, checked = repair(draft(("Revenue was $10 million.", [("E1", quote)])), source)
                self.assertEqual(repairs, [])
                self.assertEqual(checked["reason"], "quote_not_in_source")

    def test_ellipsis_quote_is_not_repaired_a12(self):
        data = draft(("Micron cash was $7.041 billion.",
                      [("E1", "As of 2024...Cash and cash equivalents | $ 7,041")]))
        repaired, repairs, checked = repair(data, A12)
        self.assertEqual(repairs, [])
        self.assertEqual(checked["reason"], "quote_not_in_source")

    def test_full_width_digits_are_matched_through_nfkc(self):
        source = make_bundle(unit("E1", "数据中心收入为１１５，１８６百万美元。"))
        repaired, repairs, checked = repair(draft(("数据中心收入为115,186百万美元。", [("E1", "收入为115,186百万美元")])),
                                            source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"], "收入为１１５，１８６百万美元")


class RowRepairTests(unittest.TestCase):
    def test_label_plus_selected_column_becomes_the_full_row_a12(self):
        data = draft(("Micron's FY2024 debt was $431 million current and $12,966 million long-term.",
                      [("E18", "Current debt | 431"), ("E21", "Long-term debt | 12,966")]))
        repaired, repairs, checked = repair(data, A12)
        self.assertTrue(checked["ok"], checked["reason"])
        quotes = [citation["quote"] for citation in repaired["claims"][0]["citations"]]
        self.assertEqual(quotes, ["Current debt | 560 | 431", "Long-term debt | 14,017 | 12,966"])
        self.assertEqual(kinds(repairs), [(0, 0, "row_cells", "E18", "E18"), (0, 1, "row_cells", "E21", "E21")])

    def test_label_free_row_fragment_maps_to_exact_contiguous_span_a12(self):
        data = draft(("Total debt was $13,397 million.", [("E72", "$ 13,491 | $ 431 | $ 12,966 | $ 13,397")]))
        repaired, repairs, checked = repair(data, A12)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"],
                         "$ | 13,491 | $ | 431 | $ | 12,966 | $ | 13,397")
        self.assertEqual(repairs[0]["kind"], "row_cells")

    def test_wrong_label_order_or_value_is_not_repaired(self):
        for quote in ("Current debt | 431 | 560", "Long-term debt | 431", "Current debt | 999",
                      "current debt | 431", "Current debt | 43"):
            with self.subTest(quote=quote):
                _, repairs, checked = repair(draft(("Current debt was $431 million.", [("E18", quote)])), A12)
                self.assertEqual(repairs, [])
                self.assertEqual(checked["reason"], "quote_not_in_source")

    def test_nbsp_inside_a_row_is_typography_not_cell_surgery(self):
        source = make_bundle(unit("E3", "Total net revenue | $\u00a034,639 | $\u00a025,785", kind="row", ticker="AMD"))
        repaired, repairs, checked = repair(
            draft(("AMD revenue was $34,639 million.", [("E3", "Total net revenue | $ 34,639")])), source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"], "Total net revenue | $\u00a034,639")
        self.assertEqual(repairs[0]["kind"], "typography")


class RebindingTests(unittest.TestCase):
    def test_correct_row_cited_to_sibling_unit_is_rebound_a09(self):
        data = draft(("Adobe's Digital Media revenue for fiscal 2024 was $15,864 million.",
                      [("E1", "Total Digital Media revenue | $ 15,864 | $ 14,216 | $ 12,842")]))
        repaired, repairs, checked = repair(data, A09)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0],
                         {"evidence_id": "E3", "quote": "Total Digital Media revenue | $ 15,864 | $ 14,216 | $ 12,842"})
        self.assertEqual(kinds(repairs), [(0, 0, "rebound_same_chunk", "E1", "E3")])

    def test_number_cited_to_adjacent_fragment_atom_is_rebound_d08(self):
        data = draft(("Membership fee revenue was $4,828 million in 2024, $4,580 million in 2023 and "
                      "$4,224 million in 2022.",
                      [("E72", "4,828"), ("E74", "4,580"), ("E75", "4,224")]))
        repaired, repairs, checked = repair(data, D08)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][2], {"evidence_id": "E76", "quote": "4,224"})
        self.assertEqual(kinds(repairs), [(0, 2, "rebound_same_chunk", "E75", "E76")])

    def test_unique_match_elsewhere_in_bundle_is_rebound(self):
        source = make_bundle(unit("E1", "Net income was $7.4 billion.", chunks=("c1",)),
                             unit("E2", "Membership fee revenue increased 5% to $4.8\u00a0billion.", chunks=("c2",)))
        repaired, repairs, checked = repair(
            draft(("Membership fees rose to $4.8 billion.", [("E1", "increased 5% to $4.8 billion")])), source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0],
                         {"evidence_id": "E2", "quote": "increased 5% to $4.8\u00a0billion"})
        self.assertEqual(repairs[0]["kind"], "rebound_bundle")

    def test_ambiguous_matches_are_left_for_validation_to_reject(self):
        source = make_bundle(unit("E1", "$"), unit("E2", "Revenue 4,224 in 2022"), unit("E3", "Cost 4,224 in 2022"),
                             unit("E4", "Other 4,224", chunks=("c2",)))
        data = draft(("Revenue was $4,224 million in 2022.", [("E1", "4,224")]))
        repaired, repairs, checked = repair(data, source)
        self.assertEqual(repairs, [])
        self.assertEqual(repaired, data)
        self.assertEqual(checked["reason"], "quote_not_in_source")
        # A unique hit inside the cited chunk wins even when another chunk repeats the figure.
        source = make_bundle(unit("E1", "$"), unit("E2", "Revenue 4,224 in 2022"),
                             unit("E4", "Other 4,224", chunks=("c2",)))
        _, repairs, checked = repair(data, source)
        self.assertEqual(kinds(repairs), [(0, 0, "rebound_same_chunk", "E1", "E2")])
        self.assertTrue(checked["ok"])

    def test_short_or_unspecific_quotes_are_never_rebound(self):
        source = make_bundle(unit("E1", "Revenue"), unit("E2", "$ 12 Total 5 % growth"))
        for quote in ("$", "12", "5 %", "Total", "$ 12"):
            with self.subTest(quote=quote):
                _, repairs, checked = repair(draft(("Revenue grew.", [("E1", quote)])), source)
                self.assertEqual(repairs, [])
                self.assertFalse(checked["ok"])

    def test_rebinding_never_cuts_through_a_number_or_word(self):
        source = make_bundle(unit("E1", "$"), unit("E2", "Debt | 1,431 | 431.5"), unit("E3", "Long-term debts total"))
        for quote in ("431", "4,31", "term debt"):
            with self.subTest(quote=quote):
                _, repairs, _ = repair(draft(("Debt was 431.", [("E1", quote)])), source)
                self.assertEqual(repairs, [])

    def test_rebound_duplicate_with_identical_quote_is_dropped(self):
        data = draft(("Membership fee revenue was $4,224 million in 2022.",
                      [("E76", "4,224"), ("E75", "4,224")]))
        repaired, repairs, checked = repair(data, D08)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"], [{"evidence_id": "E76", "quote": "4,224"}])
        self.assertEqual(kinds(repairs), [(0, 1, "duplicate_dropped", "E75", "E76")])

    def test_rebound_duplicate_with_different_quote_is_left_unrepaired(self):
        data = draft(("Membership fee revenue was $4,224 million in 2022.",
                      [("E76", "Membership fees increase"), ("E75", "4,224")]))
        repaired, repairs, checked = repair(data, D08)
        self.assertEqual(repairs, [])
        self.assertEqual(repaired, data)
        self.assertEqual(checked["reason"], "quote_not_in_source")

    def test_unknown_or_malformed_evidence_ids_are_never_rebound(self):
        for identifier in ("E999", "E01", "e3", None, 3):
            with self.subTest(identifier=identifier):
                data = draft(("Revenue was $15,864 million.",
                              [(identifier, "Total Digital Media revenue | $ 15,864")]))
                repaired, repairs, checked = repair(data, A09)
                self.assertEqual(repairs, [])
                self.assertEqual(checked["reason"], "unknown_evidence_id")


class ReboundScopeTests(unittest.TestCase):
    def test_bare_short_number_is_never_moved_to_another_companys_chunk(self):
        source = make_bundle(unit("E1", "Apple revenue table", ticker="AAPL", chunks=("c1",)),
                             unit("E2", "Microsoft other income | 391", kind="row", ticker="MSFT", chunks=("c2",)))
        data = draft(("Apple revenue was $391 million.", [("E1", "391")]))
        repaired, repairs, checked = repair(data, source)
        self.assertEqual((repairs, checked["reason"]), ([], "quote_not_in_source"))
        self.assertEqual(repaired, data)
        # Even a specific bare figure never moves to another report: the same
        # digits in another company's table are a different fact.
        source["evidence"]["E2"]["text"] = "Microsoft other income | 12,391"
        _, repairs, checked = repair(draft(("Revenue was $12,391 million.", [("E1", "12,391")])), source)
        self.assertEqual((repairs, checked["reason"]), ([], "quote_not_in_source"))


def growth_draft(operand_id="E75", value="4,224"):
    return {"status": "answer", "claims": [{
        "text": "Costco's membership fee revenue grew 14.3% from 2022 to 2024.",
        "citations": [{"evidence_id": "E72", "quote": "4,828"}, {"evidence_id": "E75", "quote": "4,224"}],
        "calculation": {"expression": "(a - b) / b * 100", "result": "14.3", "unit": "%", "operands": [
            {"name": "a", "evidence_id": "E72", "value": "4,828", "label": "Membership fees FY2024"},
            {"name": "b", "evidence_id": operand_id, "value": value, "label": "Membership fees FY2022"}]}}]}


class OperandReboundTests(unittest.TestCase):
    def test_operand_follows_its_rebound_citation_d08(self):
        data = growth_draft()
        repaired, repairs, checked = repair(data, D08)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["calculation"]["operands"][1]["evidence_id"], "E76")
        self.assertEqual(repairs[-1], {"claim_index": 0, "citation_index": 1, "kind": "operand_rebound",
                                       "from_evidence_id": "E75", "to_evidence_id": "E76", "operand_index": 1})
        self.assertEqual(data["claims"][0]["calculation"]["operands"][1]["evidence_id"], "E75")

    def test_operand_moves_only_when_its_value_is_in_the_new_unit(self):
        for operand_id, value in (("E75", "4,580"), ("E74", "4,580")):
            with self.subTest(operand_id=operand_id):
                repaired, repairs, checked = repair(growth_draft(operand_id, value), D08)
                self.assertEqual([item["kind"] for item in repairs], ["rebound_same_chunk"])
                self.assertEqual(repaired["claims"][0]["calculation"]["operands"][1]["evidence_id"], operand_id)
                self.assertEqual(checked["reason"], "calculation_operand_not_cited")


class RobustnessTests(unittest.TestCase):
    def test_drafts_beyond_validation_limits_are_copied_unrepaired(self):
        claim = ("Creative Cloud revenue was $12,682 million.", [("E1", "Creative Cloud\n$ 12,682")])
        repaired, repairs = canonicalize_citations(draft(*[claim] * 21), A09)
        self.assertEqual(repairs, [])
        self.assertEqual(validate_draft(repaired, A09)["reason"], "invalid_claim_count")
        many = draft(("Creative Cloud revenue was $12,682 million.", [("E1", "Creative Cloud\n$ 12,682")] * 9))
        repaired, repairs = canonicalize_citations(many, A09)
        self.assertEqual(repairs, [])
        self.assertEqual(validate_draft(repaired, A09)["reason"], "missing_or_excess_citations")

    def test_unrepairable_draft_shapes_are_returned_unchanged(self):
        duplicate_keys = '{"status":"answer","status":"answer","claims":[]}'
        for value in (None, 7, "not JSON", "[]", duplicate_keys, {"status": "answer"},
                      {"status": "answer", "claims": "E1"}, [draft()], {"claims": float("nan")}):
            with self.subTest(value=str(value)[:60]):
                result, repairs = canonicalize_citations(value, A09)
                self.assertIs(result, value)
                self.assertEqual(repairs, [])

    def test_json_text_drafts_are_parsed_and_repaired(self):
        text = json.dumps(draft(("Creative Cloud revenue was $12,682 million.",
                                 [("E1", "Creative Cloud\n$ 12,682")])))
        repaired, repairs = canonicalize_citations(text, A09)
        self.assertIsInstance(repaired, dict)
        self.assertEqual(repairs[0]["kind"], "typography")
        self.assertTrue(validate_draft(repaired, A09)["ok"])

    def test_malformed_bundles_and_citations_never_raise(self):
        data = draft(("Creative Cloud revenue was $12,682 million.", [("E1", "Creative Cloud\n$ 12,682")]))
        broken_entry = copy.deepcopy(A09)
        broken_entry["evidence"]["E2"] = {"text": None, "chunk_ids": 5}
        broken_chunks = copy.deepcopy(A09)
        broken_chunks["evidence"]["E3"]["chunk_ids"] = [["c1"]]
        for source in (None, [], {"evidence": []}, {"evidence": {}}, broken_entry, broken_chunks):
            with self.subTest(source=str(source)[:40]):
                result, repairs = canonicalize_citations(data, source)
                self.assertIsInstance(repairs, list)
                self.assertIsInstance(result, dict)
        odd = {"status": "answer", "claims": [None, {"text": "x", "citations": [None, 5, {"evidence_id": "E1"},
                                                                                   {"evidence_id": "E1", "quote": 5}]}]}
        result, repairs = canonicalize_citations(odd, A09)
        self.assertEqual((result, repairs), (odd, []))

    def test_other_schema_failures_keep_their_reason_codes(self):
        data = draft(("Creative Cloud revenue was $12,682 million.", [("E1", "Creative Cloud\n$ 12,682")]))
        data["claims"][0]["citations"][0]["pdf_page"] = 3
        repaired, _ = canonicalize_citations(data, A09)
        self.assertEqual(validate_draft(repaired, A09)["reason"], "invalid_citation_schema")
        data = draft(("Creative Cloud revenue was $12,682 million.", [("E1", "Creative Cloud\n$ 12,682")]))
        data["claims"][0]["source"] = "fake"
        repaired, _ = canonicalize_citations(data, A09)
        self.assertEqual(validate_draft(repaired, A09)["reason"], "invalid_claim_schema")



class FragmentRepairTests(unittest.TestCase):
    """After-fix live drafts: quotes running across parser fragments (D08, C05, B05)."""

    def test_quote_across_consecutive_fragments_becomes_exact_per_unit_quotes(self):
        source = make_bundle(unit("E49", "Membership fees . . . . . . . . . . . . $"), unit("E50", "4,828"),
                             unit("E51", "$"), unit("E52", "4,580"))
        quote = "Membership fees . . . . . . . . . $ 4,828"  # the model miscounted the dot leaders
        repaired, repairs, checked = repair(draft(("Membership fees were $4,828 million.", [("E49", quote)])), source)
        self.assertTrue(checked["ok"], checked["reason"])
        citations = repaired["claims"][0]["citations"]
        self.assertEqual([c["evidence_id"] for c in citations], ["E49", "E50"])
        self.assertEqual(citations[1]["quote"], "4,828")
        self.assertIn(citations[0]["quote"], source["evidence"]["E49"]["text"])
        self.assertEqual(repairs[-1]["kind"], "split_across_units")
        self.assertEqual(repairs[-1]["to_evidence_ids"], ["E49", "E50"])

    def test_sentence_split_across_paragraph_atoms(self):
        source = make_bundle(unit("E39", "Goodwill was tested as of the fourth"),
                             unit("E40", "quarter testing date, it was determined that the fair value exceeded "
                                         "its carrying value."))
        quote = "as of the fourth quarter testing date, it was determined that the fair value exceeded"
        repaired, _, checked = repair(draft(("The fair value exceeded the carrying value.", [("E40", quote)])), source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual([c["evidence_id"] for c in repaired["claims"][0]["citations"]], ["E39", "E40"])

    def test_split_requires_a_unique_specific_match(self):
        source = make_bundle(unit("E1", "Revenue"), unit("E2", "$ 4,828"), unit("E5", "Note", chunks=("c3",)),
                             unit("E3", "Revenue", chunks=("c2",)), unit("E4", "$ 4,828", chunks=("c2",)))
        # The cited unit's own chunk decides an otherwise repeated match.
        _, _, checked = repair(draft(("Revenue was $4,828 million.", [("E1", "Revenue $ 4,828")])), source)
        self.assertTrue(checked["ok"], checked["reason"])
        # Cited to a unit outside both chunks' match, the repeated text is ambiguous.
        _, _, checked = repair(draft(("Revenue was $4,828 million.", [("E5", "Revenue $ 4,828")])), source)
        self.assertEqual(checked["reason"], "quote_not_in_source")

    def test_ellipsis_inside_one_unit_becomes_the_exact_covering_span(self):
        text = ("As a percent of net sales, SG&A was 17.4% in fiscal 2023 compared to 16.7% in fiscal 2022. "
                "Depreciation and amortization was 1.8% in fiscal 2023 compared to 1.6% in fiscal 2022.")
        source = make_bundle(unit("E28", text))
        quote = "SG&A was 17.4% in fiscal 2023... Depreciation and amortization was 1.8% in fiscal 2023"
        repaired, repairs, checked = repair(
            draft(("SG&A was 17.4% of net sales in fiscal 2023.", [("E28", quote)])), source)
        self.assertTrue(checked["ok"], checked["reason"])
        released = repaired["claims"][0]["citations"][0]["quote"]
        self.assertTrue(released.startswith("SG&A was 17.4%") and released.endswith("1.8% in fiscal 2023"))
        self.assertIn("16.7%", released)  # nothing between the parts is hidden
        self.assertEqual(repairs[0]["kind"], "ellipsis_span")
        _, _, checked = repair(draft(("x", [("E28", "SG&A was 17.4%... fiscal 2099")])), source)
        self.assertEqual(checked["reason"], "quote_not_in_source")

    def test_bare_number_never_rebinds_to_another_report(self):
        intel = unit("E1", "Net revenue | $ 53,000 | $ 63,054", kind="row", ticker="INTC", year=2023)
        amd = unit("E2", "Research and development | $ 54,228", kind="row", ticker="AMD", year=2023, chunks=("c2",))
        source = make_bundle(intel, amd)
        _, repairs, checked = repair(draft(("Net revenue was $54,228 million.", [("E1", "54,228")])), source)
        self.assertEqual(checked["reason"], "quote_not_in_source")
        self.assertEqual(repairs, [])



class RoundThreeRepairTests(unittest.TestCase):
    def test_escaped_curly_quote_artifact_is_repaired(self):
        source = make_bundle(unit("E12", "we announced our \u201cBack to Starbucks\u201d strategy"))
        quote = "we announced our \x0022Back to Starbucks\x0022 strategy"
        repaired, _, checked = repair(draft(("Starbucks announced Back to Starbucks.", [("E12", quote)])), source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"][0]["quote"], source["evidence"]["E12"]["text"])

    def test_terminal_punctuation_and_hyphenation_differences_are_repaired(self):
        source = make_bundle(unit("E30", "Membership fee revenue increased 5% in 2024; renewal rates were strong"),
                             unit("E31", "a 110 basis-point increase from higher pricing"))
        data = draft(("Membership fees rose 5% in 2024.", [("E30", "Membership fee revenue increased 5% in 2024.")]),
                     ("Pricing added 110 basis points.", [("E31", "a 110-basis-point increase from higher pricing")]))
        repaired, repairs, checked = repair(data, source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual([item["kind"] for item in repairs], ["terminal_punctuation", "hyphenation"])

    def test_two_quotes_of_one_unit_merge_into_the_covering_span(self):
        text = "Net earnings were $10,335 million. Sales and revenues were $67,060 million."
        source = make_bundle(unit("E3", text))
        data = draft(("Net margin inputs.", [("E3", "Sales and revenues were $67,060 million."),
                                             ("E3", "Net earnings were $10,335 million.")]))
        repaired, repairs, checked = repair(data, source)
        self.assertTrue(checked["ok"], checked["reason"])
        self.assertEqual(repaired["claims"][0]["citations"], [{"evidence_id": "E3", "quote": text}])
        self.assertEqual(repairs[-1]["kind"], "merged_same_unit")


if __name__ == "__main__":
    unittest.main()
