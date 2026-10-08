"""Regression tests use real PDFs, including borderless financial statements."""

import io
import unittest
from unittest.mock import patch

import pymupdf as fitz
import pdfplumber

from report_parser.tables import extract_tables


def fixture(ruled=False, narrative=False, separated_currency=False, contents=False,
            tight_currency=False, spanning_label=False, split_numeric_cells=False,
            missing_numeric_cells=False, hidden_content=False):
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((50, 60), "Consolidated Statements of Operations", fontsize=12)
    page.insert_text((50, 80), "In millions, except per-share amounts", fontsize=10)
    if narrative:
        for index in range(9):
            y = 120 + index * 16
            page.insert_text((50, y), "Revenue increased in fiscal 2025.", fontsize=10)
            page.insert_text((330, y), "Operations continued to grow in 2024.", fontsize=10)
    else:
        rows = [("", "2025", "2024"), ("Revenue", "$1,234", "$1,105"),
                ("Costs", "(456)", "(490)"), ("Operating margin", "15.2%", "14.1%"),
                ("Earnings per share", "-1.05", "0.95")]
        if contents:
            rows = [("", "", "Page"), ("Business", "", "12"), ("Risk Factors", "", "20"),
                    ("Properties", "", "32"), ("Selected Financial Data", "", "44")]
        if spanning_label:
            rows.insert(1, ("A long financial description crossing the inferred numeric column boundary", "", ""))
        for index, row in enumerate(rows):
            y = 112 + index * 25
            page.insert_text((55, y), row[0], fontsize=10)
            for right, value in zip((450, 500) if tight_currency else (400, 545), row[1:]):
                if separated_currency and value.startswith("$"):
                    page.insert_text((right - (45 if tight_currency else 65), y), "$", fontsize=10)
                    value = value[1:]
                x = right - fitz.get_text_length(value, fontsize=10)
                page.insert_text((x, y), value, fontsize=10)
        if ruled:
            for x in (50, 320, 440, 550):
                bottom = 195 if missing_numeric_cells and x in (440, 550) else 220
                page.draw_line((x, 95), (x, bottom))
            for y in (95, 120, 145, 170, 195, 220):
                right = 320 if missing_numeric_cells and y == 220 else 550
                page.draw_line((50, y), (right, y))
            if split_numeric_cells:
                page.draw_line((385, 95), (385, 220))
    if hidden_content:
        for index in range(5):
            y = 330 + index * 25
            page.insert_text((-500, y), "Hidden financial amount", fontsize=10)
            page.insert_text((-150, y), "987,654", fontsize=10)
            page.insert_text((-70, y), "876,543", fontsize=10)
        for x in (-510, -200, -100, -20):
            page.draw_line((x, 312), (x, 440))
        for y in (312, 337, 362, 387, 412, 440):
            page.draw_line((-510, y), (-20, y))
    lines = [{"bbox": line["bbox"], "text": "".join(span["text"] for span in line["spans"])}
             for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"]]
    result = document.tobytes()
    document.close()
    return result, lines


class TableExtractionTests(unittest.TestCase):
    def extract(self, **kwargs):
        pdf, lines = fixture(**kwargs)
        with pdfplumber.open(io.BytesIO(pdf)) as document:
            return extract_tables(document.pages[0], lines, 612, 792)

    def test_ruled_table_keeps_financial_strings_and_cell_coordinates(self):
        tables = self.extract(ruled=True)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table["strategy"], "lines")
        self.assertTrue(table["structured_reliable"])
        self.assertEqual(table["rows"][0], ["", "2025", "2024"])
        self.assertEqual(table["rows"][1], ["Revenue", "$1,234", "$1,105"])
        self.assertEqual(table["rows"][2], ["Costs", "(456)", "(490)"])
        self.assertEqual(table["rows"][3][1], "15.2%")
        self.assertEqual(table["rows"][4][1], "-1.05")
        self.assertEqual(table["cells"][4]["bbox"], [320.0, 120.0, 440.0, 145.0])
        self.assertIn("In millions", table["caption"])
        self.assertIn("2025", table["markdown"])
        self.assertIn("Revenue", table["source_text"])

    def test_borderless_statement_is_cropped_and_preserves_header(self):
        tables = self.extract()
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table["strategy"], "numeric_alignment_text")
        self.assertGreater(table["bbox"][1], 90)
        rows = [r for r in table["rows"] if any(r)]
        self.assertEqual(rows[0], ["", "2025", "2024"])
        self.assertIn(["Costs", "(456)", "(490)"], rows)
        self.assertIn(["Operating margin", "15.2%", "14.1%"], rows)
        self.assertTrue(table["warnings"])

    def test_narrative_columns_with_repeated_years_are_not_tables(self):
        self.assertEqual(self.extract(narrative=True), [])

    def test_detached_currency_stays_in_the_correct_value_column(self):
        tables = self.extract(separated_currency=True)
        self.assertEqual(len(tables), 1)
        self.assertIn(["Revenue", "$ 1,234", "$ 1,105"], tables[0]["rows"])

    def test_currency_closer_to_previous_amount_still_starts_next_cell(self):
        tables = self.extract(separated_currency=True, tight_currency=True)
        self.assertEqual(len(tables), 1)
        self.assertIn(["Revenue", "$ 1,234", "$ 1,105"], tables[0]["rows"])

    def test_spanning_text_label_preserves_full_words_and_original_cells(self):
        tables = self.extract(spanning_label=True)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        label = "A long financial description crossing the inferred numeric column boundary"
        self.assertIn([label, None, None], table["rows"])
        self.assertIn("raw_rows", table)
        self.assertIn(label, table["row_source_texts"])
        self.assertTrue(any("Spanning text-only" in warning for warning in table["warnings"]))

    def test_contents_page_is_not_financial_data(self):
        self.assertEqual(self.extract(contents=True), [])

    def test_ruled_grid_splitting_digits_uses_native_text_instead(self):
        tables = self.extract(ruled=True, split_numeric_cells=True)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertFalse(table["structured_reliable"])
        self.assertEqual(table["rows"], [])
        self.assertEqual(table["cells"], [])
        self.assertTrue(table["candidate_rows"])
        self.assertTrue(table["candidate_cells"])
        self.assertIn("1,234", table["numeric_integrity"]["missing_from_cells"])
        self.assertIn("$1,234", table["markdown"])
        self.assertIn("```text", table["markdown"])
        self.assertNotIn("| Column", table["markdown"])

    def test_ruled_grid_omitting_numeric_cells_preserves_source_values(self):
        tables = self.extract(ruled=True, missing_numeric_cells=True)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertFalse(table["structured_reliable"])
        self.assertEqual(table["rows"], [])
        self.assertEqual(table["cells"], [])
        self.assertIn("1.05", table["numeric_integrity"]["missing_from_cells"])
        self.assertIn("0.95", table["numeric_integrity"]["missing_from_cells"])
        self.assertIn("-1.05", table["markdown"])
        self.assertIn("0.95", table["markdown"])
        self.assertTrue(any("numeric_integrity_failed" in warning for warning in table["warnings"]))

    def test_mismatched_coordinate_spaces_fail_explicitly(self):
        pdf, lines = fixture()
        with pdfplumber.open(io.BytesIO(pdf)) as document:
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                extract_tables(document.pages[0], lines, 792, 612)

    def test_hidden_offpage_text_and_rules_do_not_create_phantom_tables(self):
        pdf, lines = fixture(ruled=True, hidden_content=True)
        with pdfplumber.open(io.BytesIO(pdf)) as document:
            page = document.pages[0]
            self.assertTrue(any(word["x1"] < 0 for word in page.extract_words()))
            tables = extract_tables(page, lines, 612, 792)
        self.assertEqual(len(tables), 1)
        self.assertTrue(tables[0]["structured_reliable"])
        self.assertNotIn("987,654", tables[0]["markdown"])
        self.assertNotIn("Hidden", tables[0]["source_text"])
        self.assertEqual(tables[0]["bbox"], [50.0, 95.0, 550.0, 220.0])

    def test_invalid_candidate_bounds_are_rejected_without_page_failure(self):
        pdf, lines = fixture()
        invalid = [{"bbox": [0, 100, -50, 200], "cuts": [0, -25, -50]},
                   {"bbox": [0, 100, 550, 200], "cuts": [0, -25, 550]}]
        with pdfplumber.open(io.BytesIO(pdf)) as document:
            with patch("report_parser.tables._borderless_candidates", return_value=invalid):
                self.assertEqual(extract_tables(document.pages[0], lines, 612, 792), [])

    def test_cropbox_is_translated_to_pymupdf_page_coordinates(self):
        pdf, _ = fixture(ruled=True)
        with fitz.open(stream=pdf, filetype="pdf") as document:
            page = document[0]
            page.set_cropbox(fitz.Rect(9, 9, 603, 783))
            lines = [{"bbox": line["bbox"], "text": "".join(s["text"] for s in line["spans"])}
                     for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"]]
            cropped_pdf = document.tobytes()
        with pdfplumber.open(io.BytesIO(cropped_pdf)) as document:
            table = extract_tables(document.pages[0], lines, 594, 774)[0]
        self.assertEqual(table["bbox"], [41.0, 86.0, 541.0, 211.0])
        self.assertEqual(table["cells"][4]["bbox"], [311.0, 111.0, 431.0, 136.0])
        self.assertIn("In millions", table["caption"])

    def test_small_cropbox_offset_is_not_ignored_by_dimension_tolerance(self):
        pdf, _ = fixture(ruled=True)
        with fitz.open(stream=pdf, filetype="pdf") as document:
            page = document[0]
            page.set_cropbox(fitz.Rect(0.5, 0.5, 611.5, 791.5))
            lines = [{"bbox": line["bbox"], "text": "".join(s["text"] for s in line["spans"])}
                     for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"]]
            cropped_pdf = document.tobytes()
        with pdfplumber.open(io.BytesIO(cropped_pdf)) as document:
            table = extract_tables(document.pages[0], lines, 611, 791)[0]
        self.assertEqual(table["bbox"], [49.5, 94.5, 549.5, 219.5])
        self.assertTrue(table["structured_reliable"])

    def test_rotation_fails_explicitly_instead_of_returning_wrong_cells(self):
        pdf, lines = fixture(ruled=True)
        with fitz.open(stream=pdf, filetype="pdf") as document:
            document[0].set_rotation(90)
            rotated_pdf = document.tobytes()
        with pdfplumber.open(io.BytesIO(rotated_pdf)) as document:
            with self.assertRaisesRegex(ValueError, "orientation normalization"):
                extract_tables(document.pages[0], lines, 792, 612)


if __name__ == "__main__":
    unittest.main()
