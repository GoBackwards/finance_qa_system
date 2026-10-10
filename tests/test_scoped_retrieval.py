import unittest

from report_search.scoped_retrieval import document_index, merge_pools, near_duplicate, plan_pools, select_candidates


INVENTORY = [{"ticker": ticker, "fiscal_year": year, "relative_path": f"raw/{ticker}/{ticker}_FY{year}_annual_report.pdf",
              "sha256": f"{ticker.lower():0<12}{year}" + "0" * 48, "status": "validated"}
             for ticker in ("KO", "PEP") for year in (2021, 2022, 2023)]
DOCUMENTS = document_index(INVENTORY)
REPEATED = " ".join(f"word{i}" for i in range(40))


def candidate(chunk_id, ticker, year, score, text=None, document=None):
    return {"chunk_id": chunk_id, "ticker": ticker, "fiscal_year": year, "reranker_score": score,
            "vector_rank": 1, "document_id": document or f"{ticker}-{year}", "text": text or chunk_id}


class PlanPoolsTests(unittest.TestCase):
    def test_document_index_uses_indexed_document_identity_and_skips_invalid_rows(self):
        rows = INVENTORY + [dict(INVENTORY[0], fiscal_year=2020, status="rejected")]
        index = document_index(rows)
        self.assertEqual(sorted(index["KO"]), [2021, 2022, 2023])
        self.assertTrue(index["KO"][2021].startswith("KO_FY2021_annual_report-"))
        self.assertEqual(len(index["KO"][2021].split("-")[-1]), 12)

    def test_no_resolved_company_keeps_global_search(self):
        self.assertEqual(plan_pools({"tickers": []}, DOCUMENTS), [])
        self.assertEqual(plan_pools({"tickers": ["AAPL"]}, DOCUMENTS), [])

    def test_single_company_uses_full_depth_and_requested_years(self):
        pools = plan_pools({"tickers": ["KO"], "report_years": {"KO": [2022]}}, DOCUMENTS)
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["top_k"], 20)
        self.assertEqual(pools[0]["fiscal_years"], [2022])
        self.assertEqual(pools[0]["document_ids"], [DOCUMENTS["KO"][2022]])

    def test_comparison_splits_depth_and_unavailable_years_fall_back_to_all_reports(self):
        pools = plan_pools({"tickers": ["KO", "PEP"], "report_years": {"KO": [2019], "PEP": []}}, DOCUMENTS)
        self.assertEqual([pool["top_k"] for pool in pools], [10, 10])
        self.assertEqual(pools[0]["fiscal_years"], [2021, 2022, 2023])
        self.assertEqual(len(pools[1]["document_ids"]), 3)

    def test_many_companies_are_capped(self):
        documents = {f"T{i}": {2024: f"doc{i}"} for i in range(8)}
        pools = plan_pools({"tickers": list(documents)}, documents)
        self.assertLessEqual(sum(pool["top_k"] for pool in pools), 30)


class SelectionTests(unittest.TestCase):
    def test_comparison_reserves_slots_for_each_company(self):
        reranked = [candidate(f"pep{i}", "PEP", 2022, 9 - i) for i in range(6)] + [candidate("ko1", "KO", 2022, 1.0),
                                                                                   candidate("ko2", "KO", 2022, 0.5)]
        selected, trace = select_candidates(reranked, {"tickers": ["KO", "PEP"], "period_mode": "explicit"})
        self.assertEqual(len(selected), 5)
        self.assertEqual(sum(row["ticker"] == "KO" for row in selected), 2)
        self.assertEqual(trace["quota"], {"KO": 2, "PEP": 2})
        self.assertEqual([row["rank"] for row in selected], [1, 2, 3, 4, 5])
        self.assertEqual(selected[0]["chunk_id"], "pep0")

    def test_cross_report_near_duplicates_are_suppressed_but_slots_are_filled(self):
        reranked = [candidate("a2022", "EMR", 2022, 5, REPEATED, "d22"), candidate("a2023", "EMR", 2023, 4.9, REPEATED, "d23"),
                    candidate("buyout", "EMR", 2025, 3.0, "Emerson acquired the remaining shares", "d25")]
        selected, trace = select_candidates(reranked, {"tickers": ["EMR"], "period_mode": "explicit"}, top_k=2)
        self.assertEqual([row["chunk_id"] for row in selected], ["a2022", "buyout"])
        self.assertEqual(trace["suppressed_duplicates"], ["a2023"])
        filled, _ = select_candidates(reranked[:2], {"tickers": ["EMR"], "period_mode": "explicit"}, top_k=2)
        self.assertEqual(len(filled), 2)

    def test_undated_single_company_question_reserves_latest_report(self):
        reranked = [candidate(f"old{i}", "HD", 2021, 5 - i * 0.1) for i in range(6)] + [candidate("new", "HD", 2025, 1.0)]
        scope = {"tickers": ["HD"], "period_mode": "unspecified", "latest_year": {"HD": 2025}}
        selected, trace = select_candidates(reranked, scope)
        self.assertIn("new", [row["chunk_id"] for row in selected])
        self.assertEqual(trace["reserved_latest"], ["new"])
        explicit, _ = select_candidates(reranked, dict(scope, period_mode="explicit"))
        self.assertNotIn("new", [row["chunk_id"] for row in explicit])

    def test_payloads_are_not_mutated(self):
        reranked = [candidate("a", "KO", 2022, 2.0)]
        selected, _ = select_candidates(reranked, {"tickers": []})
        selected[0]["text"] = "changed"
        self.assertEqual(reranked[0]["text"], "a")

    def test_short_passages_and_same_document_are_never_duplicates(self):
        self.assertFalse(near_duplicate({"document_id": "a", "text": "Revenue"}, {"document_id": "b", "text": "Revenue"}))
        self.assertFalse(near_duplicate({"document_id": "a", "text": REPEATED}, {"document_id": "a", "text": REPEATED}))
        self.assertTrue(near_duplicate({"document_id": "a", "text": REPEATED}, {"document_id": "b", "text": REPEATED}))



class ReviewFixTests(unittest.TestCase):
    def test_undated_questions_get_a_latest_report_pool(self):
        pools = plan_pools({"tickers": ["KO"], "period_mode": "unspecified"}, DOCUMENTS)
        self.assertEqual([(pool["purpose"], pool["fiscal_years"]) for pool in pools],
                         [("scope", [2021, 2022, 2023]), ("latest", [2023])])
        explicit = plan_pools({"tickers": ["KO"], "period_mode": "explicit", "report_years": {"KO": [2023]}}, DOCUMENTS)
        self.assertEqual([pool["purpose"] for pool in explicit], ["scope"])

    def test_merged_pools_are_unique_and_globally_ranked(self):
        first = [{"chunk_id": "a", "score": 0.5, "rank": 1}, {"chunk_id": "b", "score": 0.4, "rank": 2}]
        second = [{"chunk_id": "c", "score": 0.6, "rank": 1}, {"chunk_id": "a", "score": 0.5, "rank": 1}]
        merged = merge_pools([first, second])
        self.assertEqual([(row["chunk_id"], row["rank"]) for row in merged], [("c", 1), ("a", 2), ("b", 3)])

    def test_undated_comparison_reserves_each_companys_latest_report(self):
        reranked = ([candidate(f"pep{i}", "PEP", 2021, 9 - i) for i in range(4)] + [candidate("pepnew", "PEP", 2023, 1.0)]
                    + [candidate(f"ko{i}", "KO", 2021, 5 - i) for i in range(3)] + [candidate("konew", "KO", 2023, 0.5)])
        scope = {"tickers": ["KO", "PEP"], "period_mode": "unspecified", "latest_year": {"KO": 2023, "PEP": 2023}}
        selected, trace = select_candidates(reranked, scope)
        chosen = {row["chunk_id"] for row in selected}
        self.assertTrue({"pepnew", "konew"} <= chosen)
        self.assertEqual(trace["uncovered"], [])

    def test_repeated_paragraph_with_different_figures_is_not_a_duplicate(self):
        base = " ".join(f"word{i}" for i in range(40))
        fy21 = {"document_id": "a", "ticker": "KO", "text": base + " gross margin 60.3% in 2021"}
        fy22 = {"document_id": "b", "ticker": "KO", "text": base + " gross margin 58.1% in 2022"}
        self.assertFalse(near_duplicate(fy21, fy22))
        self.assertTrue(near_duplicate(fy21, dict(fy21, document_id="b")))
        self.assertFalse(near_duplicate(fy21, dict(fy21, document_id="b", ticker="PEP")))


if __name__ == "__main__":
    unittest.main()
