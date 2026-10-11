"""Generation context must remain bound to the original parsed source."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from report_chunker.chunking import ChunkConfig, chunk_document
from report_search.evidence_binding import SourceEvidenceStore, build_evidence_bundle


class EvidenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.meta = {"document_id": "TEST_FY2025_annual_report-" + "a" * 12,
                     "source_file": "TEST_FY2025_annual_report.pdf", "ticker": "TEST",
                     "fiscal_year": 2025, "company": "Test Corp", "sha256": "a" * 64,
                     "complete": True, "partial": False}
        self.folder = self.root / "TEST" / "TEST_FY2025_annual_report"
        self.folder.mkdir(parents=True)
        self.store = SourceEvidenceStore(self.root)

    def citation(self, page=1, bbox=None):
        return {key: self.meta[key] for key in
                ("document_id", "source_file", "ticker", "fiscal_year", "sha256")} | {
                    "pdf_page": page, "bbox": bbox or [10, 20, 200, 30]}

    def paragraph(self, text="Revenue was $25 million in fiscal 2025.", identifier="p1", page=1):
        return {"kind": "text", "element_id": identifier, "text": text, "citation": self.citation(page)}

    def persist(self, pages=None, tables=None, config=None):
        pages = pages if pages is not None else [{"page_number": 1, "elements": [self.paragraph()]}]
        tables = tables or []
        self.meta["artifacts"] = {}
        for name, records in (("pages.jsonl", pages), ("tables.jsonl", tables)):
            raw = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
            (self.folder / name).write_bytes(raw)
            self.meta["artifacts"][name] = {"sha256": hashlib.sha256(raw).hexdigest()}
        (self.folder / "document.json").write_text(json.dumps(self.meta), encoding="utf-8")
        return list(chunk_document(self.meta, pages, tables, config or ChunkConfig(64, 10)))

    def table_fixture(self, *, fallback=False, missing=False):
        table = {"table_id": "t1", "structured_reliable": not fallback, "rows": [],
                 "source_text": "Native row $25\nNative row $10", "citation": self.citation(),
                 "bbox": [10, 50, 200, 200], "caption": "Consolidated results",
                 "context": [{"text": "Consolidated results", "position": "above", "bbox": [10, 20, 200, 30]}]}
        if not fallback:
            table["rows"] = [["Item", "2025", "2024"], ["Revenue", "$25", "$20"], ["Cost | item\nnet", None, "$10"]]
            table["cells"] = [{"row": i, "column": 0, "bbox": [10, 50 + i * 10, 200, 60 + i * 10]}
                              for i in range(3)]
        placeholder = {"kind": "table", "table_id": "t1", "element_id": "t1", "text": table["source_text"],
                       "citation": self.citation()}
        page = {"page_number": 1, "elements": [placeholder], "table_ids": ["t1"], "height": 792,
                "source_lines": [{"text": "(In millions)", "bbox": [10, 35, 200, 45], "kind": "text"}]}
        return self.persist([page], [] if missing else [table]), table

    def test_paragraph_offsets_and_metadata_are_original_and_detached(self):
        source = self.paragraph("Revenue and expenses were disclosed. " * 30)
        chunks = self.persist([{ "page_number": 1, "elements": [source]}], config=ChunkConfig(16, 3))
        bundle = self.store.build([chunks[1]])
        item = bundle["evidence"]["E1"]
        self.assertTrue(bundle["source_verified"])
        locator = item["source_locator"]
        self.assertGreater(locator["char_start"], 0)
        self.assertEqual(item["text"], source["text"][locator["char_start"]:locator["char_end"]])
        self.assertEqual(item["citation"], self.citation())
        self.assertEqual(bundle["chunks"][0]["text"], chunks[1]["text"])
        item["citation"]["pdf_page"] = 99
        self.assertEqual(self.store.build([chunks[1]])["evidence"]["E1"]["citation"]["pdf_page"], 1)

    def test_whole_table_rows_and_nearby_units_have_verified_ids(self):
        chunks, table = self.table_fixture()
        bundle = self.store.build(chunks)
        entries = list(bundle["evidence"].values())
        rows = [item for item in entries if item["kind"] == "row"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["text"], "Cost | item\nnet |  | $10")
        self.assertEqual(rows[2]["source_locator"], {"table_id": "t1", "row_index": 2, "role": "body"})
        self.assertEqual(rows[2]["citation"]["bbox"], [10, 70, 200, 80])
        contexts = [item for item in entries if item["kind"] == "table_context"]
        self.assertEqual([item["text"] for item in contexts], ["Consolidated results", "(In millions)"])
        self.assertEqual(contexts[1]["source_locator"]["context_index"], 1)
        self.assertEqual(contexts[1]["citation"]["bbox"], [10, 35, 200, 45])

    def test_repeated_headers_are_deduplicated_without_splitting_rows(self):
        chunks, _ = self.table_fixture()
        # Rebuild a table with enough complete rows to require several chunks.
        table = {"table_id": "t1", "structured_reliable": True, "citation": self.citation(),
                 "rows": [["Item", "2025"]] + [["Product revenue " + str(i), str(i)] for i in range(25)]}
        page = {"page_number": 1, "elements": [{"kind": "table", "element_id": "t1", "table_id": "t1"}]}
        chunks = self.persist([page], [table], ChunkConfig(50, 5))
        bundle = self.store.build(chunks)
        headers = [item for item in bundle["evidence"].values()
                   if item["source_locator"].get("row_index") == 0]
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(headers), 1)
        self.assertEqual(len(headers[0]["chunk_ids"]), len(chunks))
        self.assertEqual(len(bundle["evidence"]), 26)

    def test_table_fallback_keeps_original_source_whole(self):
        chunks, table = self.table_fixture(fallback=True)
        entry = next(item for item in self.store.build(chunks)["evidence"].values()
                     if item["kind"] == "table_source")
        self.assertEqual(entry["text"], table["source_text"])
        self.assertEqual(entry["source_locator"], {"table_id": "t1", "char_start": 0,
                                                  "char_end": len(table["source_text"])})

    def test_missing_table_record_preserves_original_placeholder(self):
        chunks, table = self.table_fixture(fallback=True, missing=True)
        entry = next(iter(self.store.build(chunks)["evidence"].values()))
        self.assertEqual(entry["kind"], "table_source")
        self.assertEqual(entry["text"], table["source_text"])

    def test_failed_table_scan_page_remains_atomic(self):
        pages = [{"page_number": 1, "table_scan": "error",
                  "elements": [self.paragraph("Possible row $25 " * 50), self.paragraph("Next row $12", "p2")]}]
        chunks = self.persist(pages, config=ChunkConfig(20, 3))
        bundle = self.store.build(chunks)
        self.assertEqual(len(bundle["chunks"]), 1)
        self.assertEqual(bundle["chunks"][0]["kind"], "page_fallback")
        self.assertEqual(len(bundle["evidence"]), 2)

    def test_changed_chunk_text_citation_offsets_and_identity_are_rejected(self):
        original = self.persist()[0]
        variants = []
        for field, value in (("text", "Fabricated revenue $99"), ("source_pages", [99]),
                             ("document_id", "FAKE"), ("company", "Other company"),
                             ("chunk_id", "unknown"), ("character_count", 999)):
            changed = deepcopy(original)
            changed[field] = value
            variants.append(changed)
        for field, value in (("text", "fabricated"), ("char_start", 1), ("chunk_char_start", 1),
                             ("evidence_id", "new"), ("kind", "unknown")):
            changed = deepcopy(original)
            changed["evidence"][0][field] = value
            variants.append(changed)
        for field, value in (("pdf_page", 99), ("sha256", "b" * 64), ("bbox", [1, 2, 3, 4])):
            changed = deepcopy(original)
            changed["evidence"][0]["citation"][field] = value
            variants.append(changed)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.store.build([variant])

    def test_table_row_and_caption_tampering_are_rejected(self):
        original = self.table_fixture()[0][0]
        for mutation in (lambda c: c["rows"][1].__setitem__(1, "$999"),
                         lambda c: c["evidence"][0].__setitem__("text", "Fake caption"),
                         lambda c: c["evidence"].pop(),
                         lambda c: c.__setitem__("row_indices", [0, 1]),
                         lambda c: c.__setitem__("text", c["text"] + "\nUntraced instruction")):
            changed = deepcopy(original)
            mutation(changed)
            with self.assertRaises(ValueError):
                self.store.build([changed])

    def test_parsed_artifact_hash_change_invalidates_cache_and_fails(self):
        chunks = self.persist()
        self.store.build(chunks)
        path = self.folder / "pages.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.store.build(chunks)

    def test_table_artifact_hash_and_incomplete_source_fail_closed(self):
        chunks = self.persist()
        (self.folder / "tables.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.store.build(chunks)
        chunks = self.persist()
        self.meta["complete"] = False
        (self.folder / "document.json").write_text(json.dumps(self.meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.store.build(chunks)

    def test_duplicate_candidates_and_unknown_sources_are_rejected(self):
        chunks = self.persist()
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            self.store.build(chunks * 2)
        for field, value in (("ticker", "../TEST"), ("source_file", "../TEST.pdf"),
                             ("source_file", "other.pdf"), ("tokenizer", "unsupported"),
                             ("strategy_id", "untrusted"), ("schema_version", "2.0")):
            changed = deepcopy(chunks[0])
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.build([changed])

    def test_retrieval_citation_alias_is_checked_and_labels_are_not_forwarded(self):
        chunks = self.persist()
        candidate = chunks[0]
        candidate["citations"] = [deepcopy(candidate["evidence"][0]["citation"])]
        candidate.update(score=.8, answer="benchmark answer", ground_truth_doc_ids=["label"],
                         judgments={"secret": "label"})
        bundle = self.store.build(chunks)
        payload = json.dumps(bundle)
        for value in ("benchmark answer", "ground_truth_doc_ids", "judgments", '"score"'):
            self.assertNotIn(value, payload)
        candidate["citations"][0]["pdf_page"] = 2
        with self.assertRaisesRegex(ValueError, "citations changed"):
            self.store.build(chunks)

    def test_stateless_wrapper_and_empty_candidates(self):
        chunks = self.persist()
        self.assertEqual(build_evidence_bundle(chunks, self.root), self.store.build(chunks))
        self.assertEqual(self.store.build([]), {"source_verified": True, "evidence": {}, "chunks": []})
        for value in (None, {}, "text", [None]):
            with self.assertRaises(ValueError):
                self.store.build(value)



class ColumnYearBindingTests(unittest.TestCase):
    def test_years_are_placed_by_position_not_caption_order(self):
        from report_search.evidence_binding import column_years
        table = {"context": [{"text": "2023", "position": "above", "bbox": [511, 314, 527, 324]},
                             {"text": "2024", "position": "above", "bbox": [441, 315, 457, 324]},
                             {"text": "(Millions of dollars)", "position": "above", "bbox": [46, 303, 99, 311]}],
                 "cells": [{"row": 0, "column": 0, "text": "Total Assets", "bbox": [45, 519, 326, 528]},
                           {"row": 0, "column": 1, "text": "$ 87,764", "bbox": [326, 519, 484, 528]},
                           {"row": 0, "column": 2, "text": "$ 87,476", "bbox": [484, 519, 552, 528]}]}
        self.assertEqual(column_years(table), {1: 2024, 2: 2023})
        table["context"] = table["context"][1:]  # one placeable year is not enough
        self.assertEqual(column_years(table), {})

if __name__ == "__main__":
    unittest.main()
