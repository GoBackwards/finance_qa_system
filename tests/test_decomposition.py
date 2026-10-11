import copy
import json
import unittest

from report_search.decomposition import (
    DECOMPOSER_SYSTEM, pool_depth, plan_subquestions, retrieve_decomposed, rule_subquestions,
    select_round_robin, validate_plan,
)
from report_search.query_scope import ScopeResolver


INVENTORY = [{"ticker": ticker, "company": company, "fiscal_year": year, "status": "validated"}
             for ticker, company in (("KO", "Coca-Cola"), ("PEP", "PepsiCo"), ("AVGO", "Broadcom"),
                                     ("TXN", "Texas Instruments"))
             for year in (2021, 2022, 2023)]
RESOLVER = ScopeResolver(INVENTORY)
QUESTION = "Using their FY2022 annual reports, which had higher consolidated revenue: Coca-Cola or PepsiCo?"
SCOPE = RESOLVER.resolve(QUESTION).to_dict()


class Model:
    def __init__(self, response):
        self.response, self.calls = response, []

    def complete_json(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return copy.deepcopy(self.response)


def plan(*items):
    return {"subquestions": [{"ticker": ticker, "question": question} for ticker, question in items]}


GOOD = plan(("KO", "What consolidated revenue did Coca-Cola report in its FY2022 annual report?"),
            ("PEP", "What consolidated revenue did PepsiCo report in its FY2022 annual report?"))


class ValidationTests(unittest.TestCase):
    def test_scope_resolves_a_two_company_comparison(self):
        self.assertEqual(SCOPE["tickers"], ["KO", "PEP"])
        self.assertTrue(SCOPE["comparison"])

    def test_valid_plan_is_accepted_and_normalized(self):
        noisy = plan(("KO", "  What consolidated revenue did Coca-Cola report\nin its FY2022 annual report?  "),
                     ("PEP", "What consolidated revenue did PepsiCo report in its FY2022 annual report?"))
        subquestions, reason = validate_plan(noisy, SCOPE, RESOLVER)
        self.assertIsNone(reason)
        self.assertEqual(subquestions[0]["question"],
                         "What consolidated revenue did Coca-Cola report in its FY2022 annual report?")

    def test_invalid_plans_are_rejected_with_a_reason(self):
        cases = {
            "invalid_schema": {"subquestions": GOOD["subquestions"], "extra": 1},
            "missing_company": plan(("KO", "What consolidated revenue did Coca-Cola report in FY2022?")),
            "mentions_other_company": plan(("KO", "Did Coca-Cola or PepsiCo report more revenue in FY2022?"),
                                           ("PEP", "What revenue did PepsiCo report in FY2022?")),
            "introduces_year": plan(("KO", "What revenue did Coca-Cola report in FY2027?"),
                                    ("PEP", "What revenue did PepsiCo report in FY2022?")),
            "invalid_item": plan(("AAPL", "What revenue did Apple report in FY2022?"),
                                 ("PEP", "What revenue did PepsiCo report in FY2022?")),
            "invalid_count": {"subquestions": GOOD["subquestions"] * 4},
        }
        for reason, candidate in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(validate_plan(candidate, SCOPE, RESOLVER), (None, reason))

    def test_duplicate_subquestions_collapse(self):
        doubled = {"subquestions": GOOD["subquestions"] + GOOD["subquestions"][:1]}
        subquestions, reason = validate_plan(doubled, SCOPE, RESOLVER)
        self.assertIsNone(reason)
        self.assertEqual(len(subquestions), 2)


class PlanningTests(unittest.TestCase):
    def test_model_plan_is_used_when_valid(self):
        model = Model(GOOD)
        trace = plan_subquestions(QUESTION, SCOPE, RESOLVER, RESOLVER.company_names, model=model)
        self.assertEqual(trace["source"], "model")
        self.assertEqual([item["ticker"] for item in trace["subquestions"]], ["KO", "PEP"])
        context = json.loads(model.calls[0][1]["content"])
        self.assertEqual([company["ticker"] for company in context["companies"]], ["KO", "PEP"])
        self.assertTrue(model.calls[0][0]["content"].startswith("You split a comparison"))

    def test_invalid_or_failing_model_falls_back_to_the_rule(self):
        for response in (plan(("KO", "What revenue did Coca-Cola report in FY2022?")), RuntimeError("down")):
            with self.subTest(response=type(response).__name__):
                trace = plan_subquestions(QUESTION, SCOPE, RESOLVER, RESOLVER.company_names, model=Model(response))
                self.assertEqual(trace["source"], "rule")
                self.assertTrue(trace["rejected"])
                self.assertEqual(len(trace["subquestions"]), 2)

    def test_rule_removes_the_other_companies(self):
        subquestions = rule_subquestions(QUESTION, SCOPE, RESOLVER, RESOLVER.company_names)
        self.assertEqual(subquestions[0]["question"],
                         "Coca-Cola: Using their FY2022 annual reports, which had higher consolidated revenue: Coca-Cola ?")
        for item in subquestions:
            self.assertEqual(RESOLVER.companies_in_text(item["question"]), {item["ticker"]})

    def test_prompt_examples_are_fictional(self):
        self.assertEqual(RESOLVER.companies_in_text(DECOMPOSER_SYSTEM), frozenset())
        self.assertIn("Example Corp", DECOMPOSER_SYSTEM)


def ranked(prefix, scores):
    return [{"chunk_id": f"{prefix}{index}", "reranker_score": score} for index, score in enumerate(scores)]


class SelectionTests(unittest.TestCase):
    def test_round_robin_gives_each_subquestion_slots_and_skips_shared_passages(self):
        first, second = ranked("a", [9, 8, 7, 6]), ranked("b", [2, 1, 0.5])
        second.insert(1, dict(first[0]))
        selected, trace = select_round_robin([first, second], top_k=5)
        ids = [row["chunk_id"] for row in selected]
        self.assertEqual(sorted(ids), ["a0", "a1", "a2", "b0", "b1"])
        self.assertEqual(trace["per_subquestion_selected"], [3, 2])
        self.assertEqual(trace["company_quota"], 2)
        self.assertEqual([row["rank"] for row in selected], [1, 2, 3, 4, 5])

    def test_company_share_comes_first_so_auxiliary_subquestions_cannot_crowd_out_facts(self):
        # Two sub-questions per company (a fact and an auxiliary one), as in the v1 failure.
        lists = [ranked("kf", [9, 8, 7]), ranked("kd", [6, 5]), ranked("pf", [3, 2, 1]), ranked("pd", [0.5])]
        selected, trace = select_round_robin(lists, top_k=5, groups=["KO", "KO", "PEP", "PEP"])
        ids = {row["chunk_id"] for row in selected}
        self.assertEqual(trace["company_quota"], 2)
        self.assertTrue({"kf0", "kd0", "pf0", "pd0"} <= ids)  # each company keeps two slots
        self.assertEqual(len(selected), 5)

    def test_short_lists_are_exhausted_without_error(self):
        selected, _ = select_round_robin([ranked("a", [1]), []], top_k=5)
        self.assertEqual([row["chunk_id"] for row in selected], ["a0"])

    def test_pool_depth_keeps_reranking_bounded(self):
        self.assertEqual([pool_depth(n) for n in (1, 2, 4, 6, 8)], [10, 10, 10, 6, 5])


class RetrieveTests(unittest.TestCase):
    def test_each_subquestion_is_searched_in_its_company_and_reranked_with_its_own_text(self):
        calls = {"encode": [], "vector": [], "rerank": []}

        class Encoder:
            def encode_queries(self, texts):
                calls["encode"].append(list(texts))
                return [[float(index), 1.0] for index, _ in enumerate(texts)]

        class Engine:
            def vector(self, vector, top_k, document_ids):
                calls["vector"].append((top_k, tuple(document_ids)))
                ticker = document_ids[0].split("-")[0]
                return [{"chunk_id": f"{ticker}{i}", "score": 1 - i / 10, "rank": i + 1} for i in range(top_k)]

        class Reranker:
            def rank_candidates(self, question, rows, top_k):
                calls["rerank"].append(question)
                return [dict(row, reranker_score=10 - i) for i, row in enumerate(rows)][:top_k]

        pools = [{"ticker": "KO", "document_ids": ["KO-2022"]}, {"ticker": "PEP", "document_ids": ["PEP-2022"]}]
        result = retrieve_decomposed(encoder=Encoder(), retrieval_engine=Engine(), reranker=Reranker(),
                                     resolver=RESOLVER, query=QUESTION, scope=SCOPE, pools=pools, model=Model(GOOD))
        self.assertEqual(calls["vector"], [(10, ("KO-2022",)), (10, ("PEP-2022",))])
        self.assertEqual(calls["rerank"], [item["question"] for item in GOOD["subquestions"]])
        tickers = [row["chunk_id"][:2] for row in result["selected"]]
        self.assertEqual(len(result["selected"]), 5)
        self.assertGreaterEqual(tickers.count("PE"), 2)
        self.assertEqual(result["trace"]["decomposition"]["source"], "model")


if __name__ == "__main__":
    unittest.main()
