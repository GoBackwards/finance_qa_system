import copy
import unittest

from report_chunker.chunking import ChunkConfig, chunk_document, get_tokenizer
from report_chunker.validation import validate_document_chunks


class ChunkValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_tokenizer()

    def setUp(self):
        self.metadata = {"document_id": "DOC", "sha256": "a" * 64, "company": "Acme",
                         "ticker": "ACME", "source_file": "report.pdf", "fiscal_year": 2025}
        citation = {**self.metadata, "pdf_page": 1, "bbox": [10, 20, 500, 600]}
        self.table = {"table_id": "t1", "structured_reliable": True, "citation": citation,
                      "header_row_indices": [0], "rows": [["Item", "2025", "2024"]]
                      + [[f"Revenue source {index}", str(index + 100), str(index + 90)] for index in range(12)]}
        self.pages = [{"page_number": 1, "elements": [
            {"element_id": "e1", "kind": "text", "text": "Revenue rose. Expenses fell. " * 60,
             "citation": citation},
            {"element_id": "te1", "kind": "table", "table_id": "t1", "text": "table", "citation": citation}]}]
        self.tables = [self.table]

    def chunks(self):
        return list(chunk_document(self.metadata, self.pages, self.tables, ChunkConfig(90, 10), self.tokenizer))

    def validate(self, chunks):
        return validate_document_chunks(self.metadata, self.pages, self.tables, chunks)

    def test_real_engine_output_covers_text_and_rows(self):
        result = self.validate(self.chunks())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["table_body_rows"], 12)
        self.assertEqual(result["table_header_rows"], 1)
        self.assertEqual(result["source_prose_characters"], len(self.pages[0]["elements"][0]["text"]))

    def test_missing_text_is_detected(self):
        chunks = self.chunks()
        chunks = [chunk for chunk in chunks if chunk["kind"] != "prose"]
        with self.assertRaisesRegex(ValueError, "characters missing"):
            self.validate(chunks)

    def test_altered_offsets_or_actual_chunk_text_fail(self):
        for mutation in ("offset", "text"):
            chunks = self.chunks()
            if mutation == "offset":
                chunks[0]["evidence"][0]["char_start"] += 1
            else:
                chunks[0]["text"] = "X" + chunks[0]["text"][1:]
            with self.assertRaises(ValueError):
                self.validate(chunks)

    def test_corrupted_or_split_table_row_fails(self):
        chunks = self.chunks()
        target = next(chunk for chunk in chunks if chunk["kind"] == "table")
        target["rows"] = copy.deepcopy(target["rows"])
        target["rows"][-1][-1] = "WRONG"
        with self.assertRaisesRegex(ValueError, "row cells changed or split"):
            self.validate(chunks)

    def test_missing_or_duplicate_body_row_fails(self):
        chunks = self.chunks()
        target = next(chunk for chunk in chunks if chunk["kind"] == "table")
        duplicate = copy.deepcopy(target)
        duplicate["chunk_id"] += "duplicate"
        with self.assertRaisesRegex(ValueError, "row .* missing or duplicated"):
            self.validate(chunks + [duplicate])
        with self.assertRaises(ValueError):
            self.validate([chunk for chunk in chunks if chunk is not target])

    def test_row_payload_without_rendered_text_is_rejected(self):
        chunks = self.chunks()
        target = next(chunk for chunk in chunks if chunk["kind"] == "table")
        target["text"] = "Source data omitted"
        target["character_count"] = len(target["text"])
        with self.assertRaisesRegex(ValueError, "rendered row missing"):
            self.validate(chunks)

    def test_bad_citations_or_source_pages_fail(self):
        for change in ("hash", "bbox", "pages", "duplicate_id"):
            chunks = self.chunks()
            if change == "hash":
                chunks[0]["evidence"][0]["citation"]["sha256"] = "b" * 64
            elif change == "bbox":
                chunks[0]["evidence"][0]["citation"]["bbox"] = [0, 0, float("inf"), 1]
            elif change == "pages":
                chunks[0]["source_pages"] = [999]
            else:
                chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
            with self.assertRaises(ValueError):
                self.validate(chunks)

    def test_unreliable_table_is_atomic_and_ignores_candidates(self):
        self.table.update(structured_reliable=False, source_text="Raw financial row 100 90\n" * 100,
                          candidate_rows=[["Bogus", "999"]])
        chunks = self.chunks()
        self.assertEqual(self.validate(chunks)["atomic_fallback_tables"], 1)
        target = next(chunk for chunk in chunks if chunk["kind"] == "table_fallback")
        target["evidence"][-1]["text"] = target["evidence"][-1]["text"][:-1]
        with self.assertRaisesRegex(ValueError, "fallback source text changed"):
            self.validate(chunks)

    def test_failed_table_scan_page_stays_one_atomic_chunk(self):
        self.pages[0]["elements"] = self.pages[0]["elements"][:1]
        self.pages[0]["table_scan"] = "error"
        self.tables = []
        chunks = self.chunks()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(self.validate(chunks)["atomic_fallback_pages"], 1)
        chunks[0]["kind"] = "prose"
        with self.assertRaisesRegex(ValueError, "failed table-scan page was split"):
            self.validate(chunks)


if __name__ == "__main__":
    unittest.main()
