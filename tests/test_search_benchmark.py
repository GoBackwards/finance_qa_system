"""Source-reference and development/test leakage checks on a small corpus."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from report_search.benchmark import merge_benchmarks, validate_benchmark


class SearchBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder_by_doc = {}
        for document, ticker in [("doc-a", "AAA"), ("doc-b", "BBB"), ("doc-c", "CCC")]:
            folder = self.root / ticker / "FY2024"
            folder.mkdir(parents=True)
            self.folder_by_doc[document] = folder
            citation = {"document_id": document, "sha256": document + "-hash", "pdf_page": 1}
            page = {"page_number": 1, "elements": [
                {"element_id": document + ":e1", "text": "Supplier costs increased.", "citation": citation},
                {"element_id": document + ":e2", "text": "Factory outages affect production.", "citation": citation},
                {"element_id": document + ":ignored", "text": "Unselected source text", "citation": citation},
            ]}
            tables = [{"table_id": document + ":t1", "pdf_page": 1, "citation": citation,
                       "structured_reliable": True, "rows": [["Revenue", "100", "90"]]},
                      {"table_id": document + ":t2", "pdf_page": 1, "citation": citation,
                       "structured_reliable": False, "rows": [], "source_text": "Margin\t20%\t18%",
                       "candidate_rows": [["WRONG", "40%", "38%"]]}]
            (folder / "pages.jsonl").write_text(json.dumps(page) + "\n", encoding="utf-8")
            (folder / "tables.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tables), encoding="utf-8")
            metadata = {"document_id": document, "ticker": ticker, "fiscal_year": 2024,
                        "source_file": document + ".pdf", "sha256": document + "-hash", "complete": True,
                        "artifacts": {name: {"sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()}
                                      for name in ("pages.jsonl", "tables.jsonl")}}
            (folder / "document.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.benchmark = {"questions": [
            self.question("factual-a", "factual", "dev", [self.prose("doc-a")]),
            self.question("comparative-ab", "comparative", "dev", [self.row("doc-a"), self.row("doc-b")]),
            self.question("summary-c", "summary", "test", [self.prose("doc-c"), self.prose("doc-c", "e2")]),
        ], "provenance": {"human_reviewed": False}}

    def prose(self, document, element="e1"):
        text = "Supplier costs increased." if element == "e1" else "Factory outages affect production."
        return {"kind": "prose", "evidence_id": document + ":" + element,
                "document_id": document, "source_sha256": document + "-hash", "pdf_page": 1,
                "element_id": document + ":" + element, "char_start": 0, "char_end": len(text), "quote": text}

    def row(self, document):
        return {"kind": "table", "evidence_id": document + ":t1:row:0", "document_id": document,
                "source_sha256": document + "-hash", "pdf_page": 1, "table_id": document + ":t1",
                "row_index": 0, "row_cells": ["Revenue", "100", "90"], "row_text": "Revenue | 100 | 90"}

    def fallback(self, document):
        quote = "Margin\t20%\t18%"
        return {"kind": "table_source", "evidence_id": document + ":t2:source", "document_id": document,
                "source_sha256": document + "-hash", "pdf_page": 1, "table_id": document + ":t2",
                "char_start": 0, "char_end": len(quote), "quote": quote}

    def question(self, question_id, category, split, units):
        points = [{"claim": "Claim supported by " + unit["evidence_id"], "evidence_ids": [unit["evidence_id"]]}
                  for unit in units]
        if category == "summary":
            points.append({"claim": "A third distinct point combines the risks.",
                           "evidence_ids": [unit["evidence_id"] for unit in units]})
        return {"question_id": question_id, "query": "Question " + question_id,
                "question": "Question " + question_id, "category": category, "type": category,
                "split": split, "answer": "Source-grounded answer.", "answer_points": points,
                "evidence": units, "ground_truth_doc_ids": sorted({u["document_id"] for u in units})}

    def validate(self, benchmark=None):
        return validate_benchmark(benchmark or self.benchmark, self.root)

    def test_valid_report_counts_disjoint_split_and_agent_provenance(self):
        before = copy.deepcopy(self.benchmark)
        report = self.validate()
        self.assertEqual(report["by_category"], {"comparative": 1, "factual": 1, "summary": 1})
        self.assertEqual(report["by_split"], {"dev": 2, "test": 1})
        self.assertEqual(report["required_evidence_units"], 5)
        self.assertEqual(report["companies_by_split"], {"dev": ["AAA", "BBB"], "test": ["CCC"]})
        self.assertEqual(report["by_fiscal_year"], {"2024": 3})
        self.assertEqual(self.benchmark, before)

    def test_unstructured_fallback_uses_original_text(self):
        self.benchmark["questions"].append(self.question("fallback-a", "factual", "dev", [self.fallback("doc-a")]))
        self.assertTrue(self.validate()["valid"])
        unit = self.benchmark["questions"][-1]["evidence"][0]
        unit["quote"] = "Margin\t40%\t38%"
        with self.assertRaisesRegex(ValueError, "quote/character"):
            self.validate()

    def test_structured_and_fallback_modes_cannot_be_swapped(self):
        unit = self.fallback("doc-a")
        unit["table_id"] = "doc-a:t1"
        with self.assertRaisesRegex(ValueError, "only valid for an unstructured"):
            self.validate({"questions": [self.question("x", "factual", "dev", [unit])]})
        unit = self.row("doc-a")
        unit["table_id"] = "doc-a:t2"
        with self.assertRaisesRegex(ValueError, "unverified table cells"):
            self.validate({"questions": [self.question("x", "factual", "dev", [unit])]})

    def test_quote_row_page_hash_and_document_identity_are_exact(self):
        changes = [("quote", "Supplier costs decreased."), ("char_start", -1),
                   ("char_end", 999), ("pdf_page", 2), ("source_sha256", "wrong"),
                   ("element_id", "doc-a:missing")]
        for field, value in changes:
            changed = copy.deepcopy(self.benchmark)
            changed["questions"][0]["evidence"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(changed)
        for field, value in [("row_index", -1), ("row_index", 3),
                             ("row_cells", ["Revenue", "200", "90"]),
                             ("row_text", "Revenue | 200 | 90")]:
            changed = copy.deepcopy(self.benchmark)
            changed["questions"][1]["evidence"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(changed)

    def test_question_and_claim_contracts(self):
        for field, value in [("query", ""), ("question", "different"), ("category", "invalid"),
                             ("type", "table"), ("split", "train"), ("answer", ""),
                             ("ground_truth_doc_ids", ["doc-a", "doc-b"]), ("answer_points", [])]:
            changed = copy.deepcopy(self.benchmark)
            changed["questions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(changed)
        changed = copy.deepcopy(self.benchmark)
        changed["questions"][0]["answer_points"][0]["evidence_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "invalid evidence IDs"):
            self.validate(changed)

    def test_duplicate_ids_queries_units_and_unlinked_claims_rejected(self):
        changed = copy.deepcopy(self.benchmark)
        changed["questions"].append(copy.deepcopy(changed["questions"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate question_id"):
            self.validate(changed)
        changed["questions"][-1]["question_id"] = "another-id"
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            self.validate(changed)
        changed = copy.deepcopy(self.benchmark)
        unit = copy.deepcopy(changed["questions"][0]["evidence"][0])
        unit["evidence_id"] = "new-name-same-source"
        changed["questions"][0]["evidence"].append(unit)
        with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
            self.validate(changed)
        changed = copy.deepcopy(self.benchmark)
        changed["questions"][1]["answer_points"].pop()
        with self.assertRaisesRegex(ValueError, "Every required evidence"):
            self.validate(changed)

    def test_summary_and_comparison_need_sufficient_sources(self):
        for question in [self.question("x", "comparative", "dev", [self.row("doc-a")]),
                         self.question("x", "summary", "dev", [self.prose("doc-a")])]:
            with self.assertRaises(ValueError):
                self.validate({"questions": [question]})
        changed = copy.deepcopy(self.benchmark)
        changed["questions"][2]["answer_points"].pop()
        with self.assertRaisesRegex(ValueError, "at least 3 answer points"):
            self.validate(changed)

    def test_cross_split_source_reuse_and_company_leakage_rejected(self):
        changed = copy.deepcopy(self.benchmark)
        changed["questions"].append(self.question("test-reuse", "factual", "test", [self.prose("doc-a")]))
        with self.assertRaisesRegex(ValueError, "Canonical evidence overlaps"):
            self.validate(changed)
        changed["questions"][-1] = self.question("test-new-span", "factual", "test", [self.prose("doc-a", "e2")])
        with self.assertRaisesRegex(ValueError, "companies overlap"):
            self.validate(changed)

    def test_artifact_corruption_is_detected_even_in_unselected_sources(self):
        path = self.folder_by_doc["doc-a"] / "pages.jsonl"
        path.write_text(path.read_text().replace("Unselected source text", "Corrupted source text"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.validate()

    def test_missing_sources_fail_clearly(self):
        changed = copy.deepcopy(self.benchmark)
        question = changed["questions"][0]
        question["evidence"][0]["document_id"] = "missing"
        question["ground_truth_doc_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "Missing benchmark source documents"):
            self.validate(changed)

    def test_merge_preserves_parts_and_never_claims_human_verification(self):
        files = [self.root / "part1.json", self.root / "part2.json"]
        for path, questions in zip(files, [self.benchmark["questions"][:2], self.benchmark["questions"][2:]]):
            path.write_text(json.dumps({"questions": questions, "provenance": {"creator": "agent"}}), encoding="utf-8")
        merged = merge_benchmarks(files)
        self.assertEqual(len(merged["questions"]), 3)
        self.assertFalse(merged["provenance"]["independent_human_review"])
        self.assertEqual(len(merged["provenance"]["source_parts"]), 2)
        self.assertTrue(self.validate(merged)["valid"])


if __name__ == "__main__":
    unittest.main()
