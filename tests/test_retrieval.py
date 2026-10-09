"""Small, hand-computed retrieval and evidence-recall fixtures."""

import copy
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


def table_source_gold(text, start=0, table="t1", evidence_id="ts1"):
    return {"kind": "table_source", "evidence_id": evidence_id, "document_id": "doc", "pdf_page": 1,
            "source_sha256": "abc123", "table_id": table, "char_start": start,
            "char_end": start + len(text), "quote": text}


def table_source_atom(text, start=0, table="t1"):
    return {"kind": "table_source", "table_id": table, "char_start": start,
            "char_end": start + len(text), "text": text, "citation": citation()}


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

    def test_fallback_table_full_text_and_subspan_match(self):
        text = "Revenue\t100\t90\nMargin\t20%\t18%"
        candidate = chunk("fallback", text, [table_source_atom(text)])
        for target in [table_source_gold(text), table_source_gold("100\t90", start=8)]:
            with self.subTest(quote=target["quote"]):
                result = judge_retrieval({"evidence": [target]}, [candidate])
                self.assertEqual(result["recall_at_5"], 1)
                self.assertEqual(result["gold_units"], 1)

    def test_fallback_spans_need_complete_union_and_exact_text(self):
        target = table_source_gold("abcdefghi")
        first = chunk("a", "abcde", [table_source_atom("abcde")])
        last = chunk("b", "defghi", [table_source_atom("defghi", 3)])
        self.assertEqual(judge_retrieval({"evidence": [target]}, [first, last])["recall_at_5"], 1)
        self.assertEqual(judge_retrieval({"evidence": [target]}, [first])["recall_at_5"], 0)
        last["evidence"][0]["text"] = "xxxghi"
        self.assertEqual(judge_retrieval({"evidence": [target]}, [first, last])["recall_at_5"], 0)

    def test_fallback_rejects_wrong_source_row_kind_and_malformed_spans(self):
        target = table_source_gold("alpha")
        original = chunk("c", "alpha", [table_source_atom("alpha")])
        for key, value in [("sha256", "wrong"), ("pdf_page", 2), ("document_id", "other")]:
            candidate = copy.deepcopy(original)
            candidate["evidence"][0]["citation"][key] = value
            self.assertEqual(judge_retrieval({"evidence": [target]}, [candidate])["recall_at_5"], 0)
        for key, value in [("text", "xxxxx"), ("text", None), ("text", "alph"),
                           ("kind", "row"), ("kind", "prose"), ("table_id", "other"),
                           ("char_start", True), ("char_start", -1), ("char_end", 100)]:
            candidate = copy.deepcopy(original)
            candidate["evidence"][0][key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(judge_retrieval({"evidence": [target]}, [candidate])["recall_at_5"], 0)

    def test_fallback_requires_nonempty_string_quote_and_table_id(self):
        for field in ("quote", "table_id"):
            for value in (None, "", " ", 42, [], {}):
                target = table_source_gold("alpha")
                target[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    judge_retrieval({"evidence": [target]}, [])
            target = table_source_gold("alpha")
            del target[field]
            with self.assertRaises(ValueError):
                judge_retrieval({"evidence": [target]}, [])

    def test_fallback_canonical_kind_is_distinct_from_structured_row(self):
        fallback = table_source_gold("alpha")
        fallback["row_index"] = 0
        row = dict(fallback, kind="table", evidence_id="row1", row_text="alpha")
        source = table_source_atom("alpha")
        source["row_index"] = 0
        result = judge_retrieval({"evidence": [fallback, row]}, [chunk("c", "alpha", [source])])
        self.assertEqual(result["gold_units"], 2)
        self.assertEqual(result["recovered_units"], 1)
        self.assertEqual(result["recall_at_5"], .5)

    def test_fallback_alternative_and_duplicate_labels_do_not_inflate_recall(self):
        target = table_source_gold("alpha")
        target["alternatives"] = [table_source_gold("alpha", table="t2")]
        duplicate = dict(target, evidence_id="other-label")
        candidate = chunk("c", "alpha", [table_source_atom("alpha", table="t2")])
        result = judge_retrieval({"evidence": [target, duplicate]}, [candidate, candidate])
        self.assertEqual(result["gold_units"], 1)
        self.assertEqual(result["recall_at_5"], 1)
        self.assertEqual(result["evidence_judgments"][0]["matched_location"], 1)


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

    def test_chunking_benchmark_validator_checks_fallback_source_text(self):
        from report_chunker.experiment import validate_benchmark
        (self.folder / "document.json").write_text(json.dumps({"document_id": "doc", "sha256": "abc123"}), encoding="utf-8")
        (self.folder / "pages.jsonl").write_text("", encoding="utf-8")
        table = {"table_id": "t1", "source_text": "alpha", "structured_reliable": False,
                 "citation": citation(), "rows": [], "candidate_rows": [["WRONG"]]}
        table_path = self.folder / "tables.jsonl"
        table_path.write_text(json.dumps(table) + "\n", encoding="utf-8")
        benchmark = {"questions": [{"question_id": "fallback", "question": "What is alpha?",
                                    "evidence": [table_source_gold("alpha")]}]}
        self.assertTrue(validate_benchmark(benchmark, [self.folder])["valid"])
        for field, value in [("quote", None), ("quote", "WRONG"), ("table_id", ""),
                             ("char_start", -1), ("pdf_page", 2), ("source_sha256", "wrong")]:
            changed = copy.deepcopy(benchmark)
            changed["questions"][0]["evidence"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_benchmark(changed, [self.folder])
        table["structured_reliable"] = True
        table_path.write_text(json.dumps(table) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unstructured fallback"):
            validate_benchmark(benchmark, [self.folder])

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
