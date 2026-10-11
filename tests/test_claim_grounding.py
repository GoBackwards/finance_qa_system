"""Deterministic claim grounding between reference validation and semantic review.

Claim texts mirror real drafts from the live usage audit
(data/experiments/live_usage/baseline.jsonl): A06, A11, B12, C01, C07, D05, D09.
"""

import json
import unittest

from report_search.answer_verification import check_claim_grounding, comparison_verdict


def unit(identifier, text, *, kind="prose", chunks=("c1",), ticker="DEMO", year=2025, page=7):
    locators = {
        "prose": {"element_id": f"p{page}_{identifier}", "char_start": 0, "char_end": len(text)},
        "row": {"table_id": f"p{page}_t1", "row_index": int(identifier[1:])},
        "table_context": {"table_id": f"p{page}_t1", "role": "caption"},
    }
    return {"evidence_id": identifier, "kind": kind, "text": text,
            "citation": {"document_id": f"{ticker}_FY{year}_annual_report",
                         "source_file": f"{ticker}_FY{year}_annual_report.pdf", "sha256": "a" * 64,
                         "ticker": ticker, "fiscal_year": year, "pdf_page": page, "bbox": [0, 0, 1, 1]},
            "source_locator": locators[kind], "chunk_ids": list(chunks)}


def make_bundle(*units, headers=None):
    """Chunk text is each unit's text, after an optional uncited header (table headings)."""
    chunks = {}
    for entry in units:
        for chunk_id in entry["chunk_ids"]:
            chunk = chunks.setdefault(chunk_id, {
                "chunk_id": chunk_id, "text": (headers or {}).get(chunk_id, ""), "evidence_ids": [],
                "kind": "table", "company": "Demo", "ticker": entry["citation"]["ticker"],
                "fiscal_year": entry["citation"]["fiscal_year"]})
            chunk["evidence_ids"].append(entry["evidence_id"])
            chunk["text"] += "\n" + entry["text"]
    return {"source_verified": True, "evidence": {entry["evidence_id"]: entry for entry in units},
            "chunks": list(chunks.values())}


def claim(text, *citations, calculation=None):
    result = {"text": text, "citations": [{"evidence_id": identifier, "quote": quote}
                                          for identifier, quote in citations]}
    if calculation is not None:
        result["calculation"] = calculation
    return result


def check(claims, source, **options):
    return check_claim_grounding(claims if isinstance(claims, list) else [claims], source, **options)


PROSE = make_bundle(unit("E1", "Net sales were $249.6 billion in 2024. Membership fee revenue increased 5% to "
                               "$4.8 billion, or 4,828 in millions, and was not affected by the special dividend.",
                         ticker="COST", year=2024))
QUOTE = ("E1", "Net sales were $249.6 billion in 2024.")

# B12: Mondelez fragments; KHC never appears in the cited sources.
B12 = make_bundle(
    unit("E60", "For the Years Ended December 31, 2023 2022 2021", kind="table_context", ticker="MDLZ", year=2023),
    unit("E66", "8,302 $", ticker="MDLZ", year=2023),
    unit("E74", "Net revenues increased $1,398 million (14.4%), due to higher net pricing (9.5 pp), the impact of "
                "an acquisition (5.6 pp) and flat volume/mix (- pp).", ticker="MDLZ", year=2023))
B12_CLAIM = claim("KHC revenue grew from $8,302 million in 2021 to $9,700 million in 2023, an increase of "
                  "$1,398 million or 16.9%.",
                  ("E66", "8,302 $"), ("E74", "Net revenues increased $1,398 million (14.4%)"))

# D05: AMD FY2025 income statement; the claim turns the FY2025 actual into a 2027 projection.
D05 = make_bundle(unit("E9", "Total net revenue | $ | 34,639 |  | $ | 25,785 |  | $ | 22,680", kind="row",
                       ticker="AMD"),
                  headers={"c1": "Year Ended December 27, 2025 December 28, 2024 December 30, 2023"})

# Intel FY2023 and FY2021 reports, for application-computed growth.
INTEL = make_bundle(unit("E3", "Net revenue | $ 54,228 | $ 63,054", kind="row", ticker="INTC", year=2023),
                    unit("E9", "Net revenue | $ 79,024 | $ 77,867", kind="row", ticker="INTC", year=2021,
                         chunks=("c9",)))
INTEL_CALCULATION = {"expression": "(a - b) / b * 100", "result": "-31.4", "unit": "%", "operands": [
    {"name": "a", "evidence_id": "E3", "value": "54,228", "label": "Intel net revenue FY2023 (USD millions)"},
    {"name": "b", "evidence_id": "E9", "value": "79,024", "label": "Intel net revenue FY2021 (USD millions)"}]}
INTEL_CITATIONS = (("E3", "Net revenue | $ 54,228"), ("E9", "Net revenue | $ 79,024"))


class AdmissionTests(unittest.TestCase):
    def test_real_claims_that_admit_missing_support_are_rejected_c01_c07(self):
        c01 = ("On May 16, 2022, Emerson contributed two of its stand-alone industrial software businesses "
               "along with approximately $6.0 billion in cash to Heritage AspenTech stockholders to create New "
               "AspenTech. Later in 2022, there is evidence that AspenTech, under Emerson majority ownership, "
               "entered agreements for acquisitions such as Micromine, and no separate mention of an additional "
               "purchase of the remaining shares by Emerson is provided in the supplied evidence.")
        c07 = ("For the years after 2025, Illinois Tool Works long-term debt maturities are as follows: 2026 due "
               "debt reclassified as short-term at $999 million, and total long-term debt carrying value as of "
               "December 31, 2025 includes amounts for subsequent years but individual yearly maturities beyond "
               "2025 are not fully detailed in the provided evidence.")
        for text in (c01, c07):
            with self.subTest(text=text[:40]):
                result = check(claim(text, QUOTE), PROSE)
                self.assertEqual(result, {"ok": False, "reason": "claim_admits_missing_evidence",
                                          "claim_index": 0, "detail": result["detail"]})

    def test_common_admission_phrasings_are_rejected(self):
        for text in ("The segment split is not provided in the provided sources.",
                     "Quarterly figures are not explicitly disclosed in the cited excerpts.",
                     "The provided evidence does not specify the 2025 amount.",
                     "The supplied excerpts do not break out Document Cloud.",
                     "The evidence does not mention a special dividend.",
                     "No specific information about the buyback is available.",
                     "No further details on the acquisition price were given.",
                     "The maturity schedule is not fully disclosed.",
                     "The margin cannot be determined from the provided figures.",
                     "所提供的证据中未提及具体金额。"):
            with self.subTest(text=text):
                self.assertEqual(check(claim(text, QUOTE), PROSE)["reason"], "claim_admits_missing_evidence")

    def test_ordinary_negations_are_not_admissions(self):
        for text in ("Net sales were not affected by the special dividend.",
                     "The impairment was not material.",
                     "No dividends were declared after the special dividend.",
                     "Its revenue sources do not include advertising.",
                     "Goodwill was not included in the available-for-sale portfolio.",
                     "Net sales were $249.6 billion in 2024."):
            with self.subTest(text=text):
                self.assertTrue(check(claim(text, QUOTE), PROSE)["ok"])

    def test_paraphrased_admissions_are_rejected(self):
        for text in ("The evidence provided does not specify the segment split.",
                     "The sources provided don't include segment data.",
                     "The segment split is not mentioned in the sources.",
                     "The segment split isn't available in the provided excerpts.",
                     "There is no information about the buyback in the provided evidence.",
                     "No breakdown by segment was given.",
                     "The cited passages do not break out Document Cloud.",
                     "There is insufficient information to determine the segment split.",
                     "The annual report does not break out the segment split.",
                     "具体金额未在所提供的资料中披露。",
                     "无法从所提供的资料中确定具体金额。",
                     "由于证据不足，无法回答。"):
            with self.subTest(text=text):
                self.assertEqual(check(claim(text, QUOTE), PROSE)["reason"], "claim_admits_missing_evidence")

    def test_admission_lookalikes_are_not_rejected(self):
        source = make_bundle(unit("E1", "No data breaches were disclosed in 2024. The Company's raw materials do not "
                                        "include conflict minerals. Net cash provided by operating activities did not "
                                        "decline. The sources of liquidity do not include asset sales.",
                                  ticker="COST", year=2024))
        quote = ("E1", "No data breaches were disclosed in 2024.")
        for text in ("No data breaches were disclosed in 2024.",
                     "公司原材料中未包含冲突矿产。",
                     "Net cash provided by operating activities did not decline in 2024.",
                     "The sources of liquidity do not include asset sales.",
                     "No information technology impairments were included in 2024 results."):
            with self.subTest(text=text):
                self.assertTrue(check(claim(text, quote), source)["ok"])


class PeriodTests(unittest.TestCase):
    def test_historical_value_restated_as_a_2027_projection_is_rejected_d05(self):
        text = "AMD is projected to make $34,639 million in net revenue in 2027."
        result = check(claim(text, ("E9", "Total net revenue | $ | 34,639")), D05)
        self.assertEqual(result["reason"], "claim_period_not_in_evidence")
        self.assertEqual(result["claim_index"], 0)
        ok = check(claim("AMD's total net revenue was $34,639 million in fiscal 2025.",
                         ("E9", "Total net revenue | $ | 34,639")), D05)
        self.assertTrue(ok["ok"], ok)

    def test_years_are_grounded_by_report_fiscal_year_or_cited_passage_text(self):
        quote = ("E9", "Total net revenue | $ | 34,639")
        for text in ("AMD revenue was $34,639 million in FY2025.",
                     "AMD revenue was $25,785 million in 2024 and $22,680 million in 2023.",
                     "AMD revenue was $25,785 million in FY24.",
                     "AMD 2025财年营收为34,639百万美元。",
                     "In fiscal 2025 AMD revenue was $34,639 million."):
            with self.subTest(text=text):
                self.assertTrue(check(claim(text, quote), D05)["ok"])
        for text in ("AMD revenue was $34,639 million in fiscal 2022.", "AMD revenue in FY26 was $34,639 million.",
                     "AMD revenue was $34,639 million in 2019."):
            with self.subTest(text=text):
                self.assertEqual(check(claim(text, quote), D05)["reason"], "claim_period_not_in_evidence")

    def test_years_only_in_uncited_chunks_do_not_count(self):
        source = make_bundle(unit("E1", "Revenue was $10 million.", chunks=("c1",)),
                             unit("E2", "The 2027 notes mature in 2027.", chunks=("c2",)))
        result = check(claim("Revenue will be $10 million in 2027.", ("E1", "Revenue was $10 million.")), source)
        self.assertEqual(result["reason"], "claim_period_not_in_evidence")

    def test_calendar_dates_are_periods_checked_across_formats(self):
        source = make_bundle(unit("E2", "Purchases of property and equipment | (82,999) | (52,729)", kind="row",
                                  ticker="AMZN", year=2024),
                             headers={"c1": "Year Ended December 31, 2023 2024"})
        quote = ("E2", "Purchases of property and equipment | (82,999)")
        for text in ("Amazon's capital expenditures for the year ended December 31, 2024 were $82,999 million.",
                     "Amazon's capex for the year ended Dec. 31, 2024 was $82,999 million.",
                     "截至2024年12月31日，亚马逊资本支出为82,999百万美元。"):
            with self.subTest(text=text):
                self.assertTrue(check(claim(text, quote), source)["ok"])
        result = check(claim("Amazon's capex for the year ended December 30, 2024 was $82,999 million.", quote),
                       source)
        self.assertEqual(result["reason"], "claim_period_not_in_evidence")

    def test_upper_case_source_dates_count_but_a_lower_case_verb_is_not_a_date(self):
        source = make_bundle(unit("E1", "Revenue | 9,700 | 8,302", kind="row", ticker="MDLZ", year=2023),
                             headers={"c1": "FOR THE YEARS ENDED DECEMBER 31, 2023 AND 2022"})
        quote = ("E1", "Revenue | 9,700")
        self.assertTrue(check(claim("Revenue for the year ended December 31, 2023 was $9,700 million.", quote),
                              source)["ok"])
        # '25 may rise' is a number (checked, and absent), not the date 25 May.
        self.assertEqual(check(claim("Revenue of 25 may rise in 2023.", quote), source)["reason"],
                         "claim_number_not_in_evidence")

    def test_amounts_that_look_like_years_are_checked_as_numbers(self):
        source = make_bundle(unit("E1", "Revenue | 100 | 90", kind="row", ticker="DEMO", year=2024))
        quote = ("E1", "Revenue | 100")
        for text in ("Revenue was $2024 million in fiscal 2024.", "Revenue grew 2024% in fiscal 2024.",
                     "Revenue was 2024 billion in fiscal 2024.", "Revenue was USD 2024 in fiscal 2024.",
                     "2024年营收为2024百万美元。"):
            with self.subTest(text=text):
                self.assertEqual(check(claim(text, quote), source)["reason"], "claim_number_not_in_evidence")
        self.assertTrue(check(claim("In fiscal 2024 revenue was $100 million.", quote), source)["ok"])

    def test_year_ranges_check_both_years(self):
        source = make_bundle(unit("E1", "Revenue | 100 | 90", kind="row", ticker="DEMO", year=2024),
                             headers={"c1": "Fiscal 2024 2023"})
        quote = ("E1", "Revenue | 100")
        self.assertTrue(check(claim("Revenue was $100 million in fiscal 2023-24.", quote), source)["ok"])
        self.assertEqual(check(claim("Revenue was $100 million in fiscal 2024-25.", quote), source)["reason"],
                         "claim_period_not_in_evidence")
        # A non-consecutive tail is not a year, so its digits must be grounded as a number.
        self.assertEqual(check(claim("Revenue was $100 million in 2024-27.", quote), source)["reason"],
                         "claim_number_not_in_evidence")


class NumberTests(unittest.TestCase):
    def test_uncited_and_self_computed_numbers_are_rejected_b12(self):
        result = check(B12_CLAIM, B12)
        self.assertEqual((result["reason"], result["claim_index"]), ("claim_number_not_in_evidence", 0))
        citations = B12_CLAIM["citations"]
        for text, expected in (("Revenue was $8,302 million in 2021 and $9,700 million in 2023.",
                                "claim_number_not_in_evidence"),
                               ("Net revenues increased $1,398 million (16.9%) in 2023.", "claim_number_not_in_evidence"),
                               ("Revenue was $8,302 million in 2021; net revenues increased $1,398 million "
                                "(14.4%) in 2023.", None)):
            with self.subTest(text=text):
                self.assertEqual(check(dict(B12_CLAIM, text=text, citations=citations), B12)["reason"], expected)

    def test_scale_restatements_of_source_figures_are_grounded(self):
        cases = (
            ("Oracle's research and development expenses for fiscal year 2024 were $6.690 billion.",
             "Research and development(1) |  | $ | 6,690 |  |  | 1% |  | 1% |  | $ | 6,640 | ", "ORCL", 2024),
            ("Costco's membership fee revenue for fiscal 2024 was $4.8 billion.", "Membership fees | 4,828 | 4,580",
             "COST", 2024),
            ("英伟达2025财年数据中心业务收入为115,186百万美元。", "Data Center | $ 115,186 | $ 47,525 | $ 15,005",
             "NVDA", 2025),
            ("英伟达2025财年数据中心业务收入约为1,151.86亿美元。", "Data Center | $ 115,186 | $ 47,525", "NVDA", 2025),
            ("Revenue was $1.5 trillion yen in 2025.", "Revenue | 1,500,000 | 1,400,000", "DEMO", 2025),
            ("Revenue was 34.6 bn in 2025.", "Revenue | $ 34,639", "DEMO", 2025),
        )
        for text, row, ticker, year in cases:
            with self.subTest(text=text):
                source = make_bundle(unit("E1", row, kind="row", ticker=ticker, year=year))
                result = check(claim(text, ("E1", row.strip(" |"))), source)
                self.assertTrue(result["ok"], result)

    def test_written_precision_controls_rounding(self):
        source = make_bundle(unit("E1", "Membership fees | 4,828 | 4,580", kind="row", ticker="COST", year=2024))
        quote = ("E1", "Membership fees | 4,828")
        for text, expected in (("Fees were $4.83 billion in 2024.", None), ("Fees were $4.82 billion in 2024.",
                                                                            "claim_number_not_in_evidence"),
                               ("Fees were $4.9 billion in 2024.", "claim_number_not_in_evidence"),
                               ("Fees were $5 billion in 2024.", None),
                               ("Fees were $7 billion in 2024.", "claim_number_not_in_evidence"),
                               ("Fees were $4,828 million in 2024.", None),
                               ("Fees were $4,830 million in 2024.", "claim_number_not_in_evidence"),
                               ("Fees were $48.28 million in 2024.", "claim_number_not_in_evidence")):
            with self.subTest(text=text):
                self.assertEqual(check(claim(text, quote), source)["reason"], expected)

    def test_percentages_follow_the_same_rule(self):
        source = make_bundle(unit("E1", "Gross margin | 25.4 % | 24.1 %", kind="row"))
        quote = ("E1", "Gross margin | 25.4 %")
        self.assertTrue(check(claim("Gross margin was 25.4% in 2025.", quote), source)["ok"])
        self.assertTrue(check(claim("Gross margin was 25% in 2025.", quote), source)["ok"])
        self.assertEqual(check(claim("Gross margin was 25.37% in 2025.", quote), source)["reason"],
                         "claim_number_not_in_evidence")
        self.assertEqual(check(claim("Gross margin was 26.4% in 2025.", quote), source)["reason"],
                         "claim_number_not_in_evidence")

    def test_calculation_result_and_operands_count_as_grounded(self):
        text = "Intel's revenue fell 31.4% from FY2021 to FY2023, from $79,024 million to $54,228 million."
        self.assertTrue(check(claim(text, *INTEL_CITATIONS, calculation=INTEL_CALCULATION), INTEL)["ok"])
        self.assertEqual(check(claim(text, *INTEL_CITATIONS), INTEL)["reason"], "claim_number_not_in_evidence")
        # The calculation grounds its own result as written, not a different rounding of it.
        rounded = text.replace("31.4%", "31.38%")
        self.assertEqual(check(claim(rounded, *INTEL_CITATIONS, calculation=INTEL_CALCULATION), INTEL)["reason"],
                         "calculation_result_not_in_claim")
        stray = text.replace("from $79,024 million", "from $79,024 million (a 32.5% drop)")
        self.assertEqual(check(claim(stray, *INTEL_CITATIONS, calculation=INTEL_CALCULATION), INTEL)["reason"],
                         "claim_number_not_in_evidence")

    def test_identifiers_ordinals_forms_and_single_digits_are_not_quantities(self):
        text = ("In its 10-K for fiscal 2024, Costco's 3rd largest region reported Scope 1 and Scope 2 data, "
                "Q4 sales of H100 systems, 401(k) matching, COVID-19 costs and S&P 500 membership; net sales "
                "were $249.6 billion, up 5%.")
        self.assertTrue(check(claim(text, QUOTE), PROSE)["ok"])

    def test_signs_and_parentheses_compare_absolute_values(self):
        source = make_bundle(unit("E1", "Operating income (loss) | (1,234) | 567", kind="row"))
        quote = ("E1", "Operating income (loss) | (1,234)")
        for text in ("The operating loss was $1,234 million in 2025.", "Operating income was -1,234 in 2025.",
                     "Operating income was (1,234) in 2025."):
            with self.subTest(text=text):
                self.assertTrue(check(claim(text, quote), source)["ok"])

    def test_currency_codes_and_units_glued_to_figures_are_checked(self):
        source = make_bundle(unit("E1", "Gross margin | 25.4 % | 24.1 % | Revenue | 4,828", kind="row"))
        quote = ("E1", "Gross margin | 25.4 % | 24.1 %")
        for text in ("Revenue was USD9,700 million in 2025.", "Revenue was 9,700USD in 2025.",
                     "Revenue was RMB5.2 billion in 2025.", "Gross margin rose 16.9pp in 2025.",
                     "Gross margin rose 50bps in 2025.", "Gross margin was 26.4pct in 2025."):
            with self.subTest(text=text):
                self.assertEqual(check(claim(text, quote), source)["reason"], "claim_number_not_in_evidence")
        for text in ("Revenue was USD4,828 million in 2025.", "Gross margin was 25.4pct in 2025.",
                     "Gross margin was 25.4% versus 24.1% in 2025.",
                     "Q4 sales of MI300X and H100s rose in 2025; revenue was 4,828."):
            with self.subTest(text=text):
                self.assertTrue(check(claim(text, quote), source)["ok"])

    def test_percentages_are_never_rescaled_into_amounts(self):
        source = make_bundle(unit("E74", "Net revenues increased $1,398 million (9.7%) in 2023.", ticker="MDLZ",
                                  year=2023))
        quote = ("E74", "Net revenues increased $1,398 million (9.7%)")
        for text in ("Revenue was $9,700 million in 2023.", "Revenue was $9.7 billion in 2023.",
                     "Revenue was $9,700 thousand in 2023."):
            with self.subTest(text=text):
                self.assertEqual(check(claim(text, quote), source)["reason"], "claim_number_not_in_evidence")
        self.assertTrue(check(claim("Net revenues rose 9.7%, or $1.4 billion, in 2023.", quote), source)["ok"])
        # The same holds for an application-computed percentage.
        relabelled = claim("Intel's revenue fell to $31.4 billion in FY2023.", *INTEL_CITATIONS,
                           calculation=INTEL_CALCULATION)
        self.assertEqual(check(relabelled, INTEL)["reason"], "claim_number_not_in_evidence")

    def test_leading_decimal_point_figures_are_checked(self):
        source = make_bundle(unit("E1", "Dividend yield | 0.75 % | 0.70 %", kind="row"))
        quote = ("E1", "Dividend yield | 0.75 %")
        self.assertTrue(check(claim("The dividend yield was .75% in 2025.", quote), source)["ok"])
        self.assertEqual(check(claim("The dividend yield was .95% in 2025.", quote), source)["reason"],
                         "claim_number_not_in_evidence")

    def test_numbers_in_uncited_units_do_not_ground_a_claim(self):
        source = make_bundle(unit("E1", "Revenue | 100", kind="row"), unit("E2", "Cost | 9,700", kind="row"))
        result = check(claim("Revenue was $9,700 million in 2025.", ("E1", "Revenue | 100")), source)
        self.assertEqual(result["reason"], "claim_number_not_in_evidence")


class EntityTests(unittest.TestCase):
    @staticmethod
    def resolver(text):
        return {ticker for name, ticker in (("KHC", "KHC"), ("Kraft Heinz", "KHC"), ("Mondelez", "MDLZ"))
                if name in text}

    def test_competitor_figures_relabelled_as_another_company_are_rejected_b12(self):
        relabelled = claim("KHC net revenues increased $1,398 million (14.4%) in 2023.",
                           ("E74", "Net revenues increased $1,398 million (14.4%)"))
        result = check(relabelled, B12, company_resolver=self.resolver)
        self.assertEqual(result["reason"], "claim_entity_not_in_evidence")
        self.assertEqual(result["detail"], "claim names a company outside the cited sources (cited: MDLZ)")
        self.assertTrue(check(relabelled, B12)["ok"])  # without a resolver the entity rule is skipped
        honest = dict(relabelled, text=relabelled["text"].replace("KHC", "Mondelez"))
        self.assertTrue(check(honest, B12, company_resolver=self.resolver)["ok"])

    def test_resolver_failures_fail_closed(self):
        relabelled = claim("Kraft Heinz net revenues increased $1,398 million in 2023.",
                           ("E74", "Net revenues increased $1,398 million (14.4%)"))

        def broken(text):
            raise RuntimeError("index unavailable")

        self.assertEqual(check(relabelled, B12, company_resolver=broken)["reason"], "claim_entity_not_in_evidence")
        self.assertEqual(check(relabelled, B12, company_resolver=lambda text: "KHC" if "Kraft" in text else None)["reason"],
                         "claim_entity_not_in_evidence")
        self.assertTrue(check(relabelled, B12, company_resolver=lambda text: None)["ok"])
        self.assertTrue(check(relabelled, B12, company_resolver=lambda text: {"mdlz"})["ok"])


class ContractTests(unittest.TestCase):
    def test_rules_run_in_order_across_claims_and_report_the_claim_index(self):
        number_failure = claim("Net sales were $250.9 billion in 2024.", QUOTE)
        admission = claim("The segment split is not provided in the supplied evidence.", QUOTE)
        period_failure = claim("Net sales were $249.6 billion in 2026.", QUOTE)
        result = check([number_failure, admission], PROSE)
        self.assertEqual((result["reason"], result["claim_index"]), ("claim_admits_missing_evidence", 1))
        result = check([number_failure, period_failure], PROSE)
        self.assertEqual((result["reason"], result["claim_index"]), ("claim_period_not_in_evidence", 1))
        result = check([claim("Net sales were $249.6 billion in 2024.", QUOTE), number_failure], PROSE)
        self.assertEqual((result["reason"], result["claim_index"]), ("claim_number_not_in_evidence", 1))
        self.assertEqual(check([claim("Net sales were $249.6 billion in 2024.", QUOTE)], PROSE),
                         {"ok": True, "reason": None, "claim_index": None, "detail": None})

    def test_unvalidated_claims_fail_closed_with_the_validation_reason(self):
        self.assertEqual(check(claim("Revenue was 10.", ("E9", "Revenue")), PROSE)["reason"], "unknown_evidence_id")
        self.assertEqual(check(claim("Revenue was 10.", ("E1", "Invented")), PROSE)["reason"], "quote_not_in_source")
        self.assertEqual(check_claim_grounding("not a list", PROSE)["reason"], "invalid_draft_schema")
        self.assertEqual(check_claim_grounding([], PROSE)["reason"], "invalid_claim_count")
        self.assertEqual(check(claim("Net sales were $249.6 billion in 2024.", QUOTE),
                               dict(PROSE, source_verified=False))["reason"], "unverified_evidence_bundle")

    def test_details_never_repeat_claim_text(self):
        for text in ("Revenue was $987,654 million in 2024.", "Revenue was $249.6 billion in 2031.",
                     "Secret 777 detail is not provided in the provided evidence."):
            with self.subTest(text=text):
                result = check(claim(text, QUOTE), PROSE, company_resolver=lambda value: {"ZZZZ"})
                self.assertFalse(result["ok"])
                serialized = json.dumps(result)
                for token in ("987,654", "2031", "777", "Secret"):
                    self.assertNotIn(token, serialized)

    def test_bundle_without_chunk_metadata_uses_unit_citations(self):
        source = make_bundle(unit("E1", "Revenue was $10 million.", ticker="DEMO", year=2025))
        source["chunks"] = []
        quote = ("E1", "Revenue was $10 million.")
        self.assertTrue(check(claim("Demo revenue was $10 million in FY2025.", quote), source,
                              company_resolver=lambda text: {"DEMO"})["ok"])
        self.assertEqual(check(claim("Revenue was $10 million in FY2024.", quote), source)["reason"],
                         "claim_period_not_in_evidence")



class ReviewHardeningTests(unittest.TestCase):
    def test_unknown_company_in_claim_is_rejected_unless_the_cited_text_names_it(self):
        def resolver(text):
            return ({"unknown:Apple"} if "Apple" in text else set()) | ({"MSFT"} if "Microsoft" in text else set())

        source = make_bundle(unit("E1", "Revenue | $ 211,915", kind="row", ticker="MSFT", year=2023))
        relabelled = claim("Apple's FY2023 revenue was $211,915 million.", ("E1", "Revenue | $ 211,915"))
        self.assertEqual(check(relabelled, source, company_resolver=resolver)["reason"],
                         "claim_entity_not_in_evidence")
        partner = make_bundle(unit("E1", "Microsoft and Apple are among our largest customers.",
                                   ticker="MSFT", year=2023))
        named = claim("Microsoft's FY2023 report names Apple as a customer.",
                      ("E1", "Microsoft and Apple are among our largest customers."))
        self.assertTrue(check(named, partner, company_resolver=resolver)["ok"])

    def test_competitors_named_in_the_cited_text_are_supported(self):
        def resolver(text):
            return {ticker for ticker, name in (("NVDA", "NVIDIA"), ("AMD", "AMD"), ("INTC", "Intel")) if name in text}

        source = make_bundle(unit("E1", "Our competitors include suppliers of GPUs such as AMD and Intel.",
                                  ticker="NVDA", year=2025))
        stated = claim("NVIDIA's FY2025 report names AMD and Intel as competitors.",
                       ("E1", "Our competitors include suppliers of GPUs such as AMD and Intel."))
        self.assertTrue(check(stated, source, company_resolver=resolver)["ok"])

    def test_calculation_labels_are_checked_for_periods(self):
        operands = [dict(INTEL_CALCULATION["operands"][0], label="Intel FY2027 projected revenue"),
                    INTEL_CALCULATION["operands"][1]]
        future = dict(INTEL_CALCULATION, operands=operands)
        relabelled = claim("Intel's revenue fell 31.4% from FY2021 to FY2023.", *INTEL_CITATIONS, calculation=future)
        self.assertEqual(check(relabelled, INTEL)["reason"], "claim_period_not_in_evidence")



class QualifierTests(unittest.TestCase):
    """Final live run: B12 (organic/non-GAAP), D07 (noncontrolling interests), B07 (excluding)."""

    def test_dropped_non_gaap_qualifier_is_rejected(self):
        text = ("Organic Net Revenue, a non-GAAP financial measure, increased 14.7% to $35.6 billion in 2023 "
                "and increased 12.3% to $31.7 billion in 2022.")
        source = make_bundle(unit("E1", text, ticker="MDLZ", year=2023))
        relabelled = claim("Mondelez's net revenues increased from $31.7 billion in 2022 to $35.6 billion in 2023, "
                           "with organic growth of 12.3% and 14.7%.", ("E1", text))
        result = check(relabelled, source)
        self.assertEqual(result["reason"], "claim_qualifier_mismatch")
        kept = claim("Mondelez's Organic Net Revenue, a non-GAAP measure, increased 14.7% to $35.6 billion in 2023.",
                     ("E1", text))
        self.assertTrue(check(kept, source)["ok"])

    def test_dropped_noncontrolling_qualifier_in_a_row_label_is_rejected(self):
        row = "Net earnings including noncontrolling interests | $ 1,856.7 | $ 3,762.3"
        source = make_bundle(unit("E1", row, kind="row", ticker="SBUX", year=2025))
        self.assertEqual(check(claim("Starbucks' net earnings in fiscal 2025 were $1,856.7 million.", ("E1", row)),
                               source)["reason"], "claim_qualifier_mismatch")
        self.assertTrue(check(claim("Starbucks' net earnings including noncontrolling interests in fiscal 2025 "
                                    "were $1,856.7 million.", ("E1", row)), source)["ok"])
        # Live D07 retry: a year between the qualifier and the figure must not cut the context.
        for text in ("Starbucks' reported net earnings including noncontrolling interests in fiscal year 2025 "
                     "amounted to $1,856.7 million.",
                     "Net earnings including noncontrolling interests for fiscal 2025 totaled $1,856.7 million."):
            self.assertTrue(check(claim(text, ("E1", row)), source)["ok"], text)

    def test_added_scope_qualifier_must_come_from_the_sources(self):
        text = "Total sales and revenues in 2023 were $67.060 billion."
        source = make_bundle(unit("E1", text, ticker="CAT", year=2023))
        relabelled = claim("Caterpillar's sales excluding financing were $67.060 billion in 2023.", ("E1", text))
        self.assertEqual(check(relabelled, source)["reason"], "claim_qualifier_mismatch")

    def test_single_digit_percentages_must_be_grounded(self):
        row = "Total revenue | 105,000 | 100,000"
        source = make_bundle(unit("E1", row, kind="row", ticker="COST", year=2024))
        derived = claim("Costco revenue grew 5% in 2024.", ("E1", row))
        self.assertEqual(check(derived, source)["reason"], "claim_number_not_in_evidence")
        stated = make_bundle(unit("E2", "Workforce reductions of 4% to 5% are expected in 2024.", ticker="KMB", year=2024))
        self.assertTrue(check(claim("Kimberly-Clark expects 4% to 5% workforce reductions in 2024.",
                                    ("E2", "Workforce reductions of 4% to 5% are expected in 2024.")), stated)["ok"])

    def test_dates_do_not_carry_amount_qualifiers(self):
        text = ("On May 16, 2022, the Company completed the combination. The Company recorded a noncontrolling "
                "interest of $5.9 billion for the 45 percent ownership.")
        source = make_bundle(unit("E1", text, ticker="EMR", year=2022))
        stated = claim("On May 16, 2022, Emerson recorded a noncontrolling interest of $5.9 billion.", ("E1", text))
        self.assertTrue(check(stated, source)["ok"])  # live C01: the date is not a qualified amount

    def test_neighbouring_adjusted_figure_does_not_taint_a_gaap_figure(self):
        text = "Adjusted diluted EPS of $4.52 and diluted EPS of $4.31 in fiscal 2024."
        source = make_bundle(unit("E1", text, ticker="GIS", year=2024))
        self.assertTrue(check(claim("General Mills' diluted EPS was $4.31 in fiscal 2024.", ("E1", text)), source)["ok"])



def year_row(identifier, text, cells, **options):
    entry = unit(identifier, text, kind="row", **options)
    entry["column_years"] = cells
    return entry


# dev-017: the caption printed "2023 2024" over columns 2024 | 2023; the store binds years by position.
ASSETS = make_bundle(year_row("E1", "Total Assets | $ 87,764 | $ 87,476", [["$ 87,764", 2024], ["$ 87,476", 2023]],
                              ticker="CAT", year=2024),
                     headers={"c1": "December 31, (Millions of dollars) 2023 2024"})


class ColumnYearTests(unittest.TestCase):
    def test_figure_from_the_prior_year_column_is_rejected(self):
        result = check(claim("Caterpillar's total assets were $87,476 million at the end of fiscal 2024.",
                             ("E1", "Total Assets | $ 87,764 | $ 87,476")), ASSETS)
        self.assertEqual(result["reason"], "claim_column_year_mismatch")
        self.assertIn("2023 column", result["detail"])

    def test_right_column_prior_year_named_and_multi_year_claims_pass(self):
        for text in ("Total assets were $87,764 million at the end of fiscal 2024.",
                     "Total assets were $87,476 million at the end of fiscal 2023.",
                     "Total assets were $87,764 million.",  # no year: the cited report's year
                     "Total assets were $87,476 million on December 31, 2023.",  # year named only by a date
                     "Total assets rose from $87,476 million in 2023 to $87,764 million in 2024."):
            with self.subTest(text=text):
                self.assertTrue(check(claim(text, ("E1", "Total Assets | $ 87,764 | $ 87,476")), ASSETS)["ok"])

    def test_a_december_date_does_not_accept_the_prior_year_column(self):
        # dev-017 replay: the prior-year figure was released 'as of December 31, 2024'.
        result = check(claim("Caterpillar reported total assets of $87,476 million as of December 31, 2024.",
                             ("E1", "Total Assets | $ 87,764 | $ 87,476")), ASSETS)
        self.assertEqual(result.get("reason"), "claim_column_year_mismatch")

    def test_a_fiscal_year_ending_early_in_the_next_calendar_year_is_accepted(self):
        # Cadence FY2021 ended January 1, 2022; Home Depot FY2024 ended February 2, 2025.
        for company, year, value, other, date in (("CDNS", 2021, "2,988.2", "2,682.9", "January 1, 2022"),
                                                  ("HD", 2024, "159,514", "152,669", "February 2, 2025")):
            bundle = make_bundle(year_row("E1", f"Total revenue | $ {value} | $ {other}",
                                          [[f"$ {value}", year], [f"$ {other}", year - 1]], ticker=company, year=year),
                                 headers={"c1": f"Fiscal year ended {date} {year} {year - 1}"})
            with self.subTest(company=company):
                result = check(claim(f"Total revenue was ${value} million for the fiscal year ended {date}.",
                                     ("E1", f"Total revenue | $ {value} | $ {other}")), bundle)
                self.assertNotEqual(result.get("reason"), "claim_column_year_mismatch")

    def test_calculation_operands_use_their_label_year(self):
        calculation = {"expression": "a / 1", "result": "87476", "unit": "USD millions",
                       "operands": [{"name": "a", "evidence_id": "E1", "value": "87,476",
                                     "label": "Total assets FY2024 (USD millions)"}]}
        result = check(claim("Total assets were $87,476 million.", ("E1", "Total Assets | $ 87,764 | $ 87,476"),
                             calculation=calculation), ASSETS)
        self.assertEqual(result["reason"], "claim_column_year_mismatch")

    def test_footnoted_row_is_rejected_when_the_plain_row_is_in_context(self):
        bundle = make_bundle(unit("E1", "Operating income (1) | 6,690", kind="row"),
                             unit("E2", "Operating income | 8,915", kind="row"))
        result = check(claim("Operating income was $6,690 million in 2025.", ("E1", "Operating income (1) | 6,690")),
                       bundle)
        self.assertEqual(result["reason"], "claim_footnoted_row")
        alone = make_bundle(unit("E1", "Operating income (1) | 6,690", kind="row"))
        self.assertTrue(check(claim("Operating income was $6,690 million in 2025.",
                                    ("E1", "Operating income (1) | 6,690")), alone)["ok"])


PAIR = make_bundle(unit("E1", "Net Revenue | $ 91,854 | $ 91,471", kind="row", chunks=("p",), ticker="PEP", year=2024),
                   unit("E2", "Net Operating Revenues | $ 47,061 | $ 45,754", kind="row", chunks=("k",), ticker="KO",
                        year=2024))
for chunk, company in zip(PAIR["chunks"], ("PepsiCo", "Coca-Cola")):
    chunk["company"] = company
PEP_CLAIM = claim("PepsiCo's FY2024 net revenue was $91,854 million.", ("E1", "Net Revenue | $ 91,854 | $ 91,471"))
KO_CLAIM = claim("Coca-Cola's FY2024 net operating revenues were $47,061 million.",
                 ("E2", "Net Operating Revenues | $ 47,061 | $ 45,754"))


class ComparisonVerdictTests(unittest.TestCase):
    QUESTION = "Which had the higher FY2024 revenue: Coca-Cola or PepsiCo?"

    def test_states_the_higher_company_and_cites_both_sides(self):
        added = comparison_verdict(self.QUESTION, ["KO", "PEP"], [PEP_CLAIM, KO_CLAIM], PAIR)
        self.assertEqual(added["text"], "PepsiCo reported the higher amount: $91,854 million, against $47,061 million "
                                        "for Coca-Cola.")
        self.assertEqual({c["evidence_id"] for c in added["citations"]}, {"E1", "E2"})

    def test_lower_questions_and_one_claim_with_both_companies(self):
        both = claim("PepsiCo reported $91,854 million and Coca-Cola reported $47,061 million in FY2024.",
                     ("E1", "Net Revenue | $ 91,854 | $ 91,471"), ("E2", "Net Operating Revenues | $ 47,061 | $ 45,754"))
        added = comparison_verdict("Which company had lower revenue in FY2024, Coca-Cola or PepsiCo?",
                                   ["KO", "PEP"], [both], PAIR)
        self.assertEqual(added["text"], "Coca-Cola reported the lower amount: $47,061 million, against $91,854 million "
                                        "for PepsiCo.")

    def test_does_not_fire_without_a_clear_two_company_amount_comparison(self):
        stated = claim("PepsiCo's revenue was higher than Coca-Cola's in FY2024.", ("E1", "Net Revenue | $ 91,854 | $ 91,471"))
        margin = [claim("PepsiCo's margin was 13.3%.", ("E1", "Net Revenue | $ 91,854 | $ 91,471")),
                  claim("Coca-Cola's margin was 25.4%.", ("E2", "Net Operating Revenues | $ 47,061 | $ 45,754"))]
        cases = {"verdict already stated": (self.QUESTION, ["KO", "PEP"], [PEP_CLAIM, KO_CLAIM, stated]),
                 "percentages": (self.QUESTION, ["KO", "PEP"], margin),
                 "one side missing": (self.QUESTION, ["KO", "PEP"], [PEP_CLAIM]),
                 "three companies": (self.QUESTION, ["KO", "PEP", "MDLZ"], [PEP_CLAIM, KO_CLAIM]),
                 "not a which question": ("Summarize Coca-Cola's and PepsiCo's FY2024 revenue.", ["KO", "PEP"],
                                          [PEP_CLAIM, KO_CLAIM])}
        for name, (question, tickers, claims) in cases.items():
            with self.subTest(name):
                self.assertIsNone(comparison_verdict(question, tickers, claims, PAIR))

if __name__ == "__main__":
    unittest.main()
