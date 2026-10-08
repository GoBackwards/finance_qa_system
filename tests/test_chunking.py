"""Source conservation and row integrity are the chunker's primary contracts."""
import copy
import unittest

from report_chunker.chunking import ChunkConfig, chunk_document, get_tokenizer, header_row_indices, token_count


META = {"document_id": "TEST-hash", "source_file": "TEST.pdf", "ticker": "TEST", "fiscal_year": 2025,
        "company": "Test Corporation"}


def citation(page=1, bbox=None):
    return {"document_id": META["document_id"], "source_file": "TEST.pdf", "sha256": "a" * 64,
            "ticker": "TEST", "fiscal_year": 2025, "pdf_page": page, "bbox": bbox or [10, 20, 500, 600]}


def element(text, identifier="e1", page=1):
    return {"element_id": identifier, "kind": "text", "text": text, "citation": citation(page)}


def table(rows, **kwargs):
    return {"table_id": "t1", "rows": rows, "structured_reliable": True, "citation": citation(),
            "cells": [{"row": r, "column": c, "text": value, "bbox": [10 + c * 30, 20 + r * 12, 40 + c * 30, 32 + r * 12]}
                      for r, row in enumerate(rows) for c, value in enumerate(row)], **kwargs}


def table_element(identifier="t1"):
    return {"kind": "table", "table_id": identifier, "element_id": "table-element", "text": "UNUSED PLACEHOLDER",
            "citation": citation()}


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_tokenizer()

    def run_chunks(self, elements=None, tables=None, config=None, pages=None):
        return list(chunk_document(META, pages or [{"page_number": 1, "elements": elements or []}],
                                   tables or [], config or ChunkConfig(64, 10), self.tokenizer))

    def assert_source_conservation(self, chunks, elements):
        for source in elements:
            coverage = bytearray(len(source["text"]))
            for chunk in chunks:
                for evidence in chunk["evidence"]:
                    if evidence.get("element_id") != source["element_id"]:
                        continue
                    lo, hi = evidence["char_start"], evidence["char_end"]
                    self.assertEqual(source["text"][lo:hi], evidence["text"])
                    self.assertEqual(evidence["text"], chunk["text"][evidence["chunk_char_start"]:evidence["chunk_char_end"]])
                    coverage[lo:hi] = b"\1" * (hi - lo)
            self.assertTrue(all(coverage), source["element_id"])

    def test_all_nine_experiments_conserve_prose_and_measure_true_tokens(self):
        elements = [element(("Revenue increased by 17% in fiscal 2025. Expenses declined. " * 40) + " END", "e1"),
                    element("Risk factors include new competitors and currency fluctuations. " * 60, "e2", 2)]
        for size in (512, 1024, 2048):
            for overlap in (50, 100, 200):
                chunks = self.run_chunks(elements, config=ChunkConfig(size, overlap))
                self.assert_source_conservation(chunks, elements)
                for chunk in chunks:
                    self.assertEqual(chunk["token_count"], token_count(chunk["text"], self.tokenizer))
                    self.assertLessEqual(chunk["token_count"], size)
                    self.assertLessEqual(chunk["overlap_tokens"], overlap)

    def test_unicode_and_special_token_literals_are_never_corrupted(self):
        text = "Lowe’s 日本語の売上 💰€—café <|endoftext|> 表格\n" * 30
        source = element(text)
        chunks = self.run_chunks([source], config=ChunkConfig(17, 5))
        self.assert_source_conservation(chunks, [source])
        self.assertNotIn("\ufffd", "".join(chunk["text"] for chunk in chunks))

    def test_one_character_larger_than_budget_remains_whole(self):
        source = element("💰💸")
        chunks = self.run_chunks([source], config=ChunkConfig(1, 0))
        self.assert_source_conservation(chunks, [source])
        self.assertTrue(all(chunk["oversized"] for chunk in chunks))
        self.assertTrue(all(chunk["oversized_reason"] == "single_unicode_character" for chunk in chunks))

    def test_prose_crosses_pages_with_per_fragment_citations(self):
        first, second = element("First page paragraph."), element("Second page paragraph.", "e2", 2)
        chunks = self.run_chunks(pages=[{"elements": [first]}, {"elements": [second]}])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source_pages"], [1, 2])
        self.assertEqual([item["citation"]["pdf_page"] for item in chunks[0]["evidence"]], [1, 2])

    def test_tables_are_barriers_and_never_duplicate_placeholder_prose(self):
        rows = [["Income", "123"], ["Costs", "99"]]
        chunks = self.run_chunks([element("Before."), table_element(), element("After.", "e2")], [table(rows)])
        self.assertEqual([c["kind"] for c in chunks], ["prose", "table", "prose"])
        self.assertEqual(chunks[2]["overlap_tokens"], 0)
        self.assertNotIn("UNUSED PLACEHOLDER", "".join(c["text"] for c in chunks))

    def test_rows_are_whole_and_repeated_headers_preserve_original_cells(self):
        rows = [["Item", "2025", "2024"]] + [[f"Product {i} revenue", f"${i:,}", f"({i * 2:,})"] for i in range(15)]
        source = table(rows, caption="Consolidated statements (in millions)")
        chunks = self.run_chunks([table_element()], [source], ChunkConfig(90, 10))
        self.assertGreater(len(chunks), 1)
        body = []
        for chunk in chunks:
            self.assertEqual(chunk["rows"][0], rows[0])
            self.assertEqual(chunk["header_row_indices"], [0])
            self.assertIn(source["caption"], chunk["text"])
            self.assertEqual(chunk["rows"], [rows[i] for i in chunk["row_indices"]])
            body.extend(i for i in chunk["row_indices"] if i != 0)
            self.assertEqual(chunk["overlap_tokens"], 0)
        self.assertEqual(body, list(range(1, len(rows))))

    def test_first_data_row_is_not_assumed_to_be_a_header(self):
        rows = [["Revenue", "2025", "2024"], ["Costs", "500", "400"]]
        self.assertEqual(header_row_indices(table(rows)), [])
        chunks = self.run_chunks([table_element()], [table(rows)], ChunkConfig(25, 0))
        self.assertEqual(sum(0 in c["row_indices"] for c in chunks), 1)

    def test_source_year_headers_and_explicit_multirow_headers(self):
        self.assertEqual(header_row_indices(table([[None, "2025", "2024"], ["Income", "21", "20"]])), [0])
        self.assertEqual(header_row_indices(table([["X", "Y"], ["Units", "Dollars"], ["Income", "25"]],
                                                 header_row_indices=[0, 1])), [0, 1])
        with self.assertRaises(ValueError):
            header_row_indices(table([["Revenue", "1"]], header_row_indices=[2]))

    def test_empty_null_multiline_and_pipe_cells_preserved_as_source_rows(self):
        rows = [["North | South\nRegion", None, ""], ["Amounts", "−12.5%", "$ 1,234.00"]]
        chunks = self.run_chunks([table_element()], [table(rows)])
        self.assertEqual([row for chunk in chunks for row in chunk["rows"]], rows)
        self.assertIn("North \\| South<br>Region", chunks[0]["text"])

    def test_oversized_single_row_is_explicit_and_atomic(self):
        long_row = ["Long commitment " * 200, "123"]
        chunks = self.run_chunks([table_element()], [table([long_row, ["Short", "1"]])], ChunkConfig(32, 5))
        containing = [chunk for chunk in chunks if 0 in chunk["row_indices"]]
        self.assertEqual(len(containing), 1)
        self.assertEqual(containing[0]["rows"], [long_row])
        self.assertTrue(containing[0]["oversized"])
        self.assertEqual(containing[0]["oversized_reason"], "atomic_table_row")

    def test_fallback_table_ignores_all_untrusted_candidate_structures(self):
        source_text = "Native exact $10,001\n" * 200
        source = {
            "table_id": "t1", "structured_reliable": False, "rows": [], "source_text": source_text,
            "candidate_rows": [["FABRICATED", "99999"]], "candidate_cells": [{"text": "FABRICATED"}],
            "raw_rows": [["FABRICATED"]], "citation": citation()}
        chunks = self.run_chunks([table_element()], [source], ChunkConfig(32, 5))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], source_text)
        self.assertNotIn("FABRICATED", str(chunks))
        self.assertTrue(chunks[0]["oversized"])
        self.assertEqual(chunks[0]["evidence"][0]["text"], source_text)

    def test_unit_context_repeated_with_exact_row_citations(self):
        source = table([["Revenue", "30"], ["Costs", "20"]], context=[{
            "text": "in millions, except per-share amounts", "position": "below", "bbox": [10, 65, 400, 75]}])
        chunks = self.run_chunks([table_element()], [source], ChunkConfig(28, 0))
        for chunk in chunks:
            self.assertIn(source["context"][0]["text"], chunk["text"])
            for evidence in chunk["evidence"]:
                if evidence["kind"] == "row":
                    self.assertEqual(evidence["evidence_id"], f"t1:row:{evidence['row_index']}")
                    self.assertEqual(evidence["citation"]["sha256"], "a" * 64)
                    self.assertEqual(len(evidence["citation"]["bbox"]), 4)

    def test_no_input_mutations_and_deterministic_ids(self):
        pages = [{"elements": [element("Original."), table_element()]}]
        tables = [table([["Revenue", "123"], ["Costs", "1"]])]
        before = copy.deepcopy((pages, tables))
        first = self.run_chunks(pages=pages, tables=tables)
        self.assertEqual(first, self.run_chunks(pages=pages, tables=tables))
        self.assertEqual((pages, tables), before)

    def test_orphan_table_records_fail_instead_of_silently_disappearing(self):
        with self.assertRaisesRegex(ValueError, "no page element"):
            self.run_chunks([element("No table marker.")], [table([["Revenue", "1"]])])

    def test_missing_table_record_is_retained_whole_and_flagged(self):
        chunks = self.run_chunks([table_element()])
        self.assertEqual(chunks[0]["text"], "UNUSED PLACEHOLDER")
        self.assertEqual(chunks[0]["kind"], "table_fallback")
        self.assertTrue(any("missing_table_record" in w for w in chunks[0]["warnings"]))

    def test_failed_table_scan_page_remains_atomic_and_carries_prose_evidence(self):
        source = element("Unknown table rows revenue 100 200.\n" * 100, "failed-e", 2)
        pages = [{"elements": [element("Before.")]},
                 {"elements": [source], "table_scan": "error", "warnings": ["table_extraction_error: vector budget exceeded"]},
                 {"elements": [element("After.", "e3", 3)]}]
        chunks = self.run_chunks(pages=pages, config=ChunkConfig(32, 5))
        self.assertEqual([c["kind"] for c in chunks], ["prose", "page_fallback", "prose"])
        self.assertEqual(chunks[1]["text"], source["text"])
        self.assertTrue(chunks[1]["oversized"])
        self.assert_source_conservation(chunks, [source])
        self.assertIn("table_extraction_error: vector budget exceeded", chunks[1]["source_quality_flags"])

    def test_nearest_unit_heading_search_stops_at_preceding_table(self):
        source = table([["Revenue", "100"]])
        source["bbox"] = [10, 300, 500, 500]
        pages = [{"height": 792, "elements": [table_element()], "table_ids": ["t1"],
                  "source_lines": [{"text": "Amounts in millions", "bbox": [20, 120, 400, 130]}]}]
        chunks = self.run_chunks(pages=pages, tables=[source])
        self.assertIn("Amounts in millions", chunks[0]["text"])
        earlier = table([["Other", "5"]]) | {"table_id": "t0", "bbox": [10, 180, 500, 200]}
        pages[0]["elements"].insert(0, table_element("t0"))
        pages[0]["table_ids"].insert(0, "t0")
        chunks = self.run_chunks(pages=pages, tables=[earlier, source])
        later = [c for c in chunks if c.get("table_id") == "t1"]
        self.assertNotIn("Amounts in millions", later[0]["text"])

    def test_native_ocr_warning_is_propagated_without_normalizing_text(self):
        pages = [{"elements": [element("Native sparse content.")], "warnings": ["low_native_text: OCR/review may be needed"]}]
        chunk = self.run_chunks(pages=pages)[0]
        self.assertEqual(chunk["source_quality_flags"], pages[0]["warnings"])

    def test_invalid_configurations(self):
        for size, overlap in [(0, 0), (10, -1), (10, 10), (10, 11), (True, 0)]:
            with self.assertRaises(ValueError):
                ChunkConfig(size, overlap)


if __name__ == "__main__":
    unittest.main()
