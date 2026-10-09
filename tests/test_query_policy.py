"""Query hints must preserve financial distinctions and source-scope semantics."""

import copy
import unittest

from report_search.query_policy import analyze_query, plan_query


INVENTORY = [
    {"company": name, "ticker": ticker, "fiscal_year": year, "status": "validated"}
    for ticker, name in (("NVDA", "NVIDIA"), ("AMD", "Advanced Micro Devices"),
                         ("MSFT", "Microsoft"), ("NOW", "ServiceNow"), ("TGT", "Target"))
    for year in range(2021, 2026)
]


class QueryPolicyTests(unittest.TestCase):
    def test_revenue_hints_preserve_original_basis_currency_and_period(self):
        query = "According to NVIDIA's FY2025 annual report, what was GAAP revenue in USD?"
        plan = plan_query(query, INVENTORY)
        self.assertEqual(plan["lexical_hints"], ["total revenue", "net sales"])
        self.assertEqual(plan["constraints"]["accounting_bases"], ["GAAP"])
        self.assertEqual(plan["constraints"]["currencies"], ["USD"])
        self.assertEqual(plan["constraints"]["requested_report_years"], [2025])
        for field in ("original_query", "dense_query", "reranker_query", "generation_query"):
            self.assertEqual(plan[field], query)
        self.assertTrue(all(value.startswith(query) for value in plan["lexical_queries"]))
        self.assertLessEqual(len(plan["lexical_queries"]), 3)

    def test_revenue_never_expands_to_income_or_profit(self):
        plan = plan_query("NVIDIA net sales FY2025", INVENTORY)
        self.assertEqual(plan["constraints"]["metric_concepts"], ["revenue"])
        self.assertEqual(plan["lexical_hints"], ["total revenue"])
        self.assertNotIn("net income", " ".join(plan["lexical_hints"]))

    def test_gross_margin_is_not_gross_profit_amount(self):
        plan = plan_query("What was AMD's consolidated GAAP gross profit margin in FY2024?", INVENTORY)
        self.assertEqual(plan["constraints"]["metric_concepts"], ["gross_margin"])
        self.assertNotIn("gross profit", plan["lexical_hints"])
        amount = plan_query("AMD gross profit in USD", INVENTORY)
        self.assertEqual(amount["constraints"]["metric_concepts"], ["gross_profit"])
        self.assertNotIn("gross margin", amount["lexical_hints"])

    def test_operating_income_is_not_operating_cash_flow(self):
        income = plan_query("Microsoft operating income", INVENTORY)
        cash = plan_query("Microsoft operating cash flow", INVENTORY)
        self.assertEqual(income["constraints"]["metric_concepts"], ["operating_income"])
        self.assertEqual(cash["constraints"]["metric_concepts"], ["operating_cash_flow"])
        self.assertFalse(any("cash" in hint for hint in income["lexical_hints"]))
        self.assertFalse(any("income" in hint for hint in cash["lexical_hints"]))

    def test_negated_metric_is_not_a_lexical_hint(self):
        plan = plan_query("Microsoft operating income, not operating cash flow", INVENTORY)
        self.assertEqual(plan["constraints"]["metric_concepts"], ["operating_income"])
        self.assertFalse(any("cash" in hint for hint in plan["lexical_hints"]))

    def test_non_gaap_is_not_silently_changed_to_gaap(self):
        query = "Compare NVIDIA and AMD non-GAAP operating income, FY2024, in EUR"
        plan = plan_query(query, INVENTORY)
        self.assertEqual(plan["constraints"]["accounting_bases"], ["NON-GAAP"])
        self.assertEqual(plan["constraints"]["currencies"], ["EUR"])
        self.assertEqual(plan["constraints"]["requested_tickers"], ["NVDA", "AMD"])
        self.assertTrue(all(value.startswith(query) for value in plan["lexical_queries"]))

    def test_apple_failure_is_missing_corpus_company(self):
        plan = plan_query("What was Apple's gross profit margin in FY2024?", INVENTORY)
        self.assertEqual(plan["coverage"]["status"], "unavailable")
        self.assertEqual(plan["coverage"]["missing_tickers"], ["AAPL"])
        self.assertEqual(plan["coverage"]["reason_codes"], ["company_not_in_corpus"])

    def test_tesla_forecast_failure_is_company_gap_not_future_year(self):
        plan = plan_query("What is Tesla's projected revenue for 2027?", INVENTORY)
        self.assertTrue(plan["constraints"]["forecast_intent"])
        self.assertEqual(plan["coverage"]["missing_tickers"], ["TSLA"])
        self.assertEqual(plan["coverage"]["missing_reports"], [])

    def test_forecast_intent_does_not_refuse_supported_company(self):
        plan = plan_query("What is NVIDIA's projected revenue in FY2027?", INVENTORY)
        self.assertTrue(plan["constraints"]["forecast_intent"])
        self.assertEqual(plan["constraints"]["explicit_fiscal_years"], [2027])
        self.assertEqual(plan["constraints"]["requested_report_years"], [])
        self.assertEqual(plan["coverage"]["status"], "covered")

    def test_future_disclosure_in_existing_filing_retains_source_scope(self):
        query = "According to NVIDIA’s FY2025 annual report, when is the CC expected to certify achievement and payouts for MY PSUs granted in Fiscal 2025 by March 2027?"
        plan = plan_query(query, INVENTORY)
        self.assertTrue(plan["constraints"]["forecast_intent"])
        self.assertEqual(plan["constraints"]["requested_report_years"], [2025])
        self.assertEqual(plan["coverage"]["status"], "covered")

    def test_future_source_report_is_unavailable_even_for_forecast(self):
        plan = plan_query("NVIDIA FY2027 annual report projected revenue", INVENTORY)
        self.assertEqual(plan["coverage"]["reason_codes"], ["requested_report_not_in_corpus"])
        self.assertEqual(plan["coverage"]["missing_reports"], [{"ticker": "NVDA", "fiscal_year": 2027}])

    def test_older_metric_may_appear_in_later_comparative_report(self):
        plan = plan_query("What was NVIDIA revenue in FY2020?", INVENTORY)
        self.assertEqual(plan["constraints"]["explicit_fiscal_years"], [2020])
        self.assertEqual(plan["constraints"]["requested_report_years"], [])
        self.assertEqual(plan["coverage"]["status"], "covered")

    def test_explicit_missing_older_report_is_distinct(self):
        plan = plan_query("NVIDIA annual report for fiscal year 2020 revenue", INVENTORY)
        self.assertEqual(plan["coverage"]["missing_reports"], [{"ticker": "NVDA", "fiscal_year": 2020}])

    def test_10k_and_shared_company_year_scope(self):
        plan = plan_query("Compare NVIDIA and AMD according to their 2024 Form 10-K reports", INVENTORY)
        self.assertEqual(plan["constraints"]["requested_reports"], [
            {"ticker": "AMD", "fiscal_year": 2024}, {"ticker": "NVDA", "fiscal_year": 2024}])
        self.assertEqual(plan["coverage"]["status"], "covered")

    def test_separate_company_report_year_pairs_do_not_form_cross_product(self):
        inventory = [INVENTORY[0], {"company": "Advanced Micro Devices", "ticker": "AMD", "fiscal_year": 2025}]
        plan = plan_query("Compare NVIDIA FY2021 annual report with AMD FY2025 annual report", inventory)
        self.assertEqual(plan["constraints"]["requested_reports"], [
            {"ticker": "AMD", "fiscal_year": 2025}, {"ticker": "NVDA", "fiscal_year": 2021}])
        self.assertEqual(plan["coverage"]["status"], "covered")

    def test_explicit_report_year_list_and_range(self):
        for phrase, years in (("FY2024 and FY2025 annual reports", [2024, 2025]),
                              ("2023-2025 annual reports", [2023, 2024, 2025]),
                              ("2023 – 2025 annual reports", [2023, 2024, 2025])):
            with self.subTest(phrase=phrase):
                plan = plan_query("NVIDIA " + phrase, INVENTORY)
                self.assertEqual(plan["constraints"]["requested_report_years"], years)

    def test_shared_company_and_year_lists_require_all_requested_reports(self):
        for query in (
            "Compare NVIDIA and AMD FY2023 and FY2024 annual reports.",
            "According to NVIDIA and AMD annual reports for FY2023 and FY2024, compare revenue.",
            "Compare NVIDIA and AMD 2023 annual reports and 2024 annual reports.",
            "Compare 2023 annual reports and 2024 annual reports of NVIDIA and AMD.",
        ):
            with self.subTest(query=query):
                plan = plan_query(query, INVENTORY)
                self.assertEqual(plan["constraints"]["requested_reports"], [
                    {"ticker": ticker, "fiscal_year": year}
                    for ticker in ("AMD", "NVDA") for year in (2023, 2024)])

    def test_missing_report_is_not_hidden_by_shared_company_year_list(self):
        inventory = [row for row in INVENTORY if (row["ticker"], row["fiscal_year"]) != ("NVDA", 2023)]
        plan = plan_query("Compare NVIDIA and AMD FY2023 and FY2024 annual reports", inventory)
        self.assertEqual(plan["coverage"]["missing_reports"], [{"ticker": "NVDA", "fiscal_year": 2023}])

    def test_common_word_ticker_not_inferred_without_symbol_marker(self):
        plan = plan_query("NOW compare operating income and revenue", INVENTORY)
        self.assertEqual(plan["constraints"]["recognized_tickers"], [])
        self.assertEqual(plan["coverage"]["status"], "unspecified")
        for symbol in ("$NOW", "NYSE:NOW", "Ticker NOW", "ServiceNow"):
            with self.subTest(symbol=symbol):
                self.assertEqual(plan_query(symbol + " revenue", INVENTORY)["constraints"]["requested_tickers"], ["NOW"])

    def test_explicit_unknown_ticker_is_a_known_inventory_gap(self):
        self.assertEqual(plan_query("$XYZ revenue", INVENTORY)["coverage"]["missing_tickers"], ["XYZ"])
        self.assertEqual(plan_query("$USD revenue", INVENTORY)["coverage"]["missing_tickers"], [])

    def test_incidental_apple_mention_does_not_require_apple_filing(self):
        plan = plan_query("What does NVIDIA's FY2025 annual report say about Apple?", INVENTORY)
        self.assertEqual(plan["constraints"]["recognized_tickers"], ["NVDA", "AAPL"])
        self.assertEqual(plan["constraints"]["requested_tickers"], ["NVDA"])
        self.assertEqual(plan["coverage"]["status"], "covered")

    def test_comparison_needs_both_companies(self):
        plan = plan_query("Compare NVIDIA and Apple FY2024 revenue", INVENTORY)
        self.assertEqual(plan["coverage"]["missing_tickers"], ["AAPL"])

    def test_target_common_word_and_exclusion(self):
        self.assertEqual(plan_query("What is the target gross margin?", INVENTORY)["constraints"]["requested_tickers"], [])
        self.assertEqual(plan_query("Target's FY2024 revenue", INVENTORY)["constraints"]["requested_tickers"], ["TGT"])
        self.assertEqual(plan_query("NVIDIA revenue, not Tesla", INVENTORY)["coverage"]["missing_tickers"], [])

    def test_inventory_is_dynamic_and_failed_rows_do_not_count(self):
        apple = {"company": "Apple", "ticker": "AAPL", "fiscal_year": 2024}
        self.assertEqual(plan_query("Apple FY2024 annual report", [*INVENTORY, apple])["coverage"]["status"], "covered")
        failed = {**apple, "status": "download_failed"}
        self.assertEqual(plan_query("Apple FY2024 annual report", [*INVENTORY, failed])["coverage"]["status"], "unavailable")

    def test_unknown_name_is_unspecified_not_a_guessed_company(self):
        self.assertEqual(plan_query("UnknownCo revenue", INVENTORY)["coverage"]["status"], "unspecified")

    def test_missing_global_report_year_is_detected(self):
        plan = plan_query("Summarize risks from the 2019 annual reports", INVENTORY)
        self.assertEqual(plan["coverage"]["missing_reports"], [{"ticker": None, "fiscal_year": 2019}])

    def test_no_input_mutation_and_deterministic_alias(self):
        original = copy.deepcopy(INVENTORY)
        self.assertEqual(plan_query("NVIDIA revenue", INVENTORY), analyze_query("NVIDIA revenue", INVENTORY))
        self.assertEqual(INVENTORY, original)

    def test_validation_rejects_empty_question_and_malformed_inventory(self):
        for value in ("", "  ", None, 17):
            with self.subTest(query=value), self.assertRaises(ValueError):
                plan_query(value, INVENTORY)
        for inventory in ({"NVDA": [2024]}, "data/manifest.json", [{"ticker": "NVDA"}],
                          [{"company": "NVIDIA", "ticker": "NVDA", "fiscal_year": True}]):
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                plan_query("NVIDIA revenue", inventory)


if __name__ == "__main__":
    unittest.main()
