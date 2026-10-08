"""Small, hand-computed retrieval and evidence-recall fixtures."""

import json
from pathlib import Path
import tempfile
import unittest

from report_chunker.retrieval import (
    build_index, evaluate, index_stats, judge_retrieval, query_expression, search,
)


def citation(document="doc", page=1):
    return {"document_id": document, "pdf_page": page, "sha256": "abc123"}


def atom(text, start=0, element="element", document="doc", page=1):
    return {"kind": "prose", "element_id": element, "char_start": start,
            "char_end": start + len(text), "text": text, "citation": citation(document, page)}


def chunk(chunk_id, text, evidence=None, ticker="ACME", year=2025, document="doc"):
    return {"chunk_id": chunk_id, "document_id": document, "ticker": ticker,
            "fiscal_year": year, "company": "Acme Corporation", "kind": "prose",
            "text": text, "token_count": len(text.split()), "character_count": len(text),
            "evidence": evidence if evidence is not None else [atom(text, document=document)]}


def gold(text, evidence_id="g1", start=0, element="element", document="doc", page=1):
    return {"evidence_id": evidence_id, "document_id": document, "pdf_page": page,
            "source_sha256": "abc123", "element_id": element, "char_start": start,
            "char_end": start + len(text), "quote": text}


class EvidenceRecallTests(unittest.TestCase):
    def test_recall_is_fraction_of_gold_units_not_hit_rate(self):
        question = {"evidence": [gold("alpha", "one"), gold("beta", "two", element="other")]}
        result = judge_retrieval(question, [chunk("c1", "alpha")])
        self.assertEqual(result["recall_at_5"], 0.5)
        self.assertEqual(result["hit_at_5"], 1)
        self.assertEqual(result["all_evidence_hit_at_5"], 0)

    def test_union_of_partial_quote_spans_can_recover_one_unit(self):
        question = {"evidence": [gold("abcdefghi")]}
        results = [chunk("c1", "abcde", [atom("abcde")]),
                   chunk("c2", "defghi", [atom("defghi", 3)])]
        result = judge_retrieval(question, results)
        self.assertEqual(result["recall_at_5"], 1)
        self.assertEqual(result["evidence_judgments"][0]["covered_intervals"], [[0, 9]])

    def test_uncovered_gap_and_duplicate_overlap_do_not_gain_recall(self):
        question = {"evidence": [gold("abcdefghi")]}
        result = judge_retrieval(question, [chunk("c1", "abcd"), chunk("c1", "abcd"),
                                          chunk("c2", "fghi", [atom("fghi", 5)])])
        self.assertEqual(result["recall_at_5"], 0)
        self.assertEqual(result["evidence_judgments"][0]["covered_characters"], 8)

    def test_wrong_source_hash_page_or_text_is_not_relevant(self):
        for field, value in [("sha256", "different"), ("pdf_page", 2), ("document_id", "other")]:
            candidate = chunk("c", "alpha")
            candidate["evidence"][0]["citation"][field] = value
            self.assertEqual(judge_retrieval({"evidence": [gold("alpha")]}, [candidate])["recall_at_5"], 0)
        candidate = chunk("c", "xxxxx")
        self.assertEqual(judge_retrieval({"evidence": [gold("alpha")]}, [candidate])["recall_at_5"], 0)

    def test_exact_table_row_match(self):
        row = "Revenue | 100 | 90"
        evidence = {"evidence_id": "r1", "document_id": "doc", "pdf_page": 1,
                    "source_sha256": "abc123", "table_id": "t1", "row_index": 2, "row_text": row}
        candidate = chunk("c", row, [{"kind": "row", "table_id": "t1", "row_index": 2,
                                       "row_text": row, "citation": citation()}])
        self.assertEqual(judge_retrieval({"evidence": [evidence]}, [candidate])["recall_at_5"], 1)
        candidate["evidence"][0]["row_index"] = 3
        self.assertEqual(judge_retrieval({"evidence": [evidence]}, [candidate])["recall_at_5"], 0)

    def test_duplicate_gold_units_do_not_change_denominator(self):
        result = judge_retrieval({"evidence": [gold("alpha", "a"), gold("alpha", "b")]},
                                [chunk("c", "alpha")])
        self.assertEqual(result["gold_units"], 1)
        self.assertEqual(result["recall_at_5"], 1)

    def test_alternative_is_another_location_for_same_unit(self):
        target = gold("alpha")
        target["alternatives"] = [gold("alpha", document="other", page=7)]
        candidate = chunk("c", "alpha", [atom("alpha", document="other", page=7)], document="other")
        result = judge_retrieval({"evidence": [target]}, [candidate])
        self.assertEqual(result["gold_units"], 1)
        self.assertEqual(result["recall_at_5"], 1)
        self.assertEqual(result["evidence_judgments"][0]["matched_location"], 1)

    def test_top_five_is_hard_cutoff(self):
        results = [chunk(f"c{i}", "other") for i in range(5)] + [chunk("c6", "alpha")]
        self.assertEqual(judge_retrieval({"evidence": [gold("alpha")]}, results)["recall_at_5"], 0)

    def test_invalid_or_absent_gold_fails_loudly(self):
        with self.assertRaises(ValueError):
            judge_retrieval({"evidence": []}, [])
        invalid = gold("alpha")
        invalid["char_end"] += 1
        with self.assertRaises(ValueError):
            judge_retrieval({"evidence": [invalid]}, [])


class PersistedRetrievalTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp" / "retrieval_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.source = self.folder / "chunks.jsonl"
        self.index = self.folder / "index.sqlite"

    def write_chunks(self, chunks):
        self.source.write_text("".join(json.dumps(value) + "\n" for value in chunks), encoding="utf-8")

    def test_streamed_index_search_stemming_metadata_and_persistence(self):
        self.write_chunks([chunk("c1", "The factories manufacture tractors."),
                           chunk("c2", "Cloud services subscriptions.", ticker="CLOUD", document="d2")])
        stats = build_index(self.source, self.index)
        self.assertEqual(stats["chunk_count"], 2)
        self.assertEqual(index_stats(self.index)["chunks_sha256"], stats["chunks_sha256"])
        self.assertEqual(search(self.index, "What tractors are manufactured?")[0]["chunk_id"], "c1")
        self.assertEqual(search(self.index, "CLOUD FY2025 subscriptions")[0]["chunk_id"], "c2")
        self.assertEqual(search(self.index, "subscriptions", ticker="ACME"), [])
        self.assertEqual(search(self.index, "the and of?"), [])

    def test_ties_break_on_chunk_id_and_query_syntax_is_literal(self):
        self.write_chunks([chunk("z", "revenue margin near"), chunk("a", "revenue margin near")])
        build_index(self.source, self.index)
        self.assertEqual([row["chunk_id"] for row in search(self.index, 'revenue " OR NOT body:margin* NEAR(')], ["a", "z"])
        self.assertEqual(query_expression('revenue " OR NOT body:margin* NEAR('),
                         '"revenue" OR "not" OR "body" OR "margin" OR "near"')

    def test_failed_rebuild_preserves_existing_index(self):
        self.write_chunks([chunk("old", "revenue")])
        build_index(self.source, self.index)
        self.write_chunks([chunk("dup", "revenue"), chunk("dup", "expense")])
        with self.assertRaises(Exception):
            build_index(self.source, self.index)
        self.assertEqual(search(self.index, "revenue")[0]["chunk_id"], "old")

    def test_evaluate_macro_average_and_distinct_filtered_label(self):
        self.write_chunks([chunk("c1", "alpha"), chunk("c2", "beta", [atom("beta", element="other")])])
        build_index(self.source, self.index)
        questions = [{"question_id": "q1", "question": "alpha", "type": "prose", "ticker": "ACME", "fiscal_year": 2025,
                      "evidence": [gold("alpha", "a"), gold("beta", "b", element="other")]},
                     {"question_id": "q2", "question": "beta", "type": "prose", "ticker": "ACME", "fiscal_year": 2025,
                      "evidence": [gold("beta", element="other")]}]
        measured = evaluate(self.index, {"questions": questions})
        self.assertEqual(measured["aggregate"]["recall_at_5"], 0.75)
        self.assertEqual(measured["aggregate"]["hit_at_5"], 1)
        self.assertEqual(measured["aggregate"]["all_evidence_hit_at_5"], 0.5)
        self.assertEqual(measured["scope"], "full_corpus_unfiltered")
        self.assertIsNone(measured["per_query"][0]["filters"])
        self.assertEqual(evaluate(self.index, questions, filtered=True)["scope"],
                         "explicit_company_fiscal_year_filtered_diagnostic")


if __name__ == "__main__":
    unittest.main()
