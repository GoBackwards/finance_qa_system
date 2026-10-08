"""Behavioral geometry tests; the rectangles model PDF text line bounds."""

import unittest

from report_parser.layout import annotate_margins, lines_to_blocks, order_lines


def line(text, x0, y0, x1, y1, block=0, number=0, font=10):
    return {"text": text, "bbox": [x0, y0, x1, y1], "font_size": font,
            "block_id": block, "line_id": number}


class ReadingOrderTests(unittest.TestCase):
    def test_two_columns_read_down_before_across(self):
        items = [line("R2", 330, 112, 550, 122), line("L1", 50, 100, 270, 110),
                 line("R1", 330, 100, 550, 110), line("L2", 50, 112, 270, 122)]
        ordered = order_lines(items, 600, 800)
        self.assertEqual([x["text"] for x in ordered], ["L1", "L2", "R1", "R2"])
        self.assertNotEqual(ordered[0]["column"], ordered[2]["column"])
        self.assertNotIn("reading_order", items[0])

    def test_three_columns_with_spanning_title_and_midpage_heading(self):
        items = [line("TITLE", 50, 20, 550, 35)]
        for y, suffix in ((60, "1"), (72, "2"), (125, "3"), (137, "4")):
            for index, (left, right) in enumerate(((50, 190), (230, 370), (410, 550))):
                items.append(line(f"{chr(65 + index)}{suffix}", left, y, right, y + 10, block=index + 1))
        items.append(line("SECTION", 50, 100, 550, 115))
        ordered = order_lines(list(reversed(items)), 600, 800)
        self.assertEqual([x["text"] for x in ordered],
                         ["TITLE", "A1", "A2", "B1", "B2", "C1", "C2", "SECTION",
                          "A3", "A4", "B3", "B4", "C3", "C4"])

    def test_close_spanning_title_does_not_force_row_order(self):
        items = [line("TITLE", 50, 80, 550, 90)]
        for n, y in enumerate((91, 103, 115)):
            items.extend([line(f"L{n}", 50, y, 270, y + 10), line(f"R{n}", 330, y, 550, y + 10)])
        self.assertEqual([x["text"] for x in order_lines(items, 600, 800)],
                         ["TITLE", "L0", "L1", "L2", "R0", "R1", "R2"])

    def test_one_column_and_empty_page(self):
        self.assertEqual(order_lines([], 600, 800), [])
        items = [line("Second", 50, 112, 500, 122), line("First", 50, 100, 500, 110)]
        self.assertEqual([x["text"] for x in order_lines(items, 600, 800)], ["First", "Second"])

    def test_paragraph_blocks_preserve_refs_without_merging_columns(self):
        items = [line("left first", 50, 100, 270, 110, 1, 0), line("left second", 50, 112, 270, 122, 1, 1),
                 line("right first", 330, 100, 550, 110, 2, 0), line("right second", 330, 112, 550, 122, 2, 1)]
        blocks = lines_to_blocks(order_lines(items, 600, 800))
        self.assertEqual([x["text"] for x in blocks], ["left first left second", "right first right second"])
        self.assertEqual(blocks[0]["line_ids"], [{"block_id": 1, "line_id": 0}, {"block_id": 1, "line_id": 1}])
        self.assertEqual(blocks[0]["bbox"], [50, 100, 270, 122])
        self.assertEqual(sum(len(x["lines"]) for x in blocks), len(items))

    def test_paragraphs_never_merge_across_pages(self):
        items = [dict(line("Page one", 50, 100, 270, 110), page_number=1),
                 dict(line("Page two", 50, 112, 270, 122), page_number=2)]
        self.assertEqual(len(lines_to_blocks(items)), 2)

    def test_table_placeholder_is_isolated_and_keeps_identifier(self):
        items = [line("Before", 50, 100, 500, 110),
                 dict(line("| Revenue | 123 |", 50, 112, 500, 212), kind="table", table_id="table-0",
                      source_line_id="table:0"), line("After", 50, 214, 500, 224)]
        blocks = lines_to_blocks(order_lines(items, 600, 800))
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[1]["kind"], "table")
        self.assertEqual(blocks[1]["table_id"], "table-0")
        self.assertEqual(blocks[1]["line_ids"][0]["source_line_id"], "table:0")

    def test_narrow_gutters_in_six_column_spread(self):
        # Geometry modeled on NVDA FY2025 PDF p11: the title spans two of
        # three columns on the left printed page; three more are on the right.
        items = [line("TITLE", 36, 49, 247, 108, font=24)]
        for column, x in enumerate((36, 205, 374, 657, 826, 995), 1):
            for row, y in enumerate((130, 144, 158), 1):
                items.append(line(f"C{column}R{row}", x, y, x + 160, y + 10, column, row, font=8))
        ordered = order_lines(items, 1188, 774)
        self.assertEqual([x["text"] for x in ordered],
                         ["TITLE"] + [f"C{column}R{row}" for column in range(1, 7) for row in range(1, 4)])

    def test_vertically_stacked_centered_heading_stays_before_subheading(self):
        # MSFT FY2025 p19 has disjoint x ranges, but no vertical overlap.
        items = [line("RESEARCH AND DEVELOPMENT", 226.25, 127.26, 367.788, 137.26),
                 line("Product and Service Development", 39, 149.152, 198.687, 159.152),
                 line("Our success is based on our ability", 39, 171.06, 555.126, 181.06)]
        self.assertEqual([x["text"] for x in order_lines(items, 594, 774)], [x["text"] for x in items])

    def test_detached_bullets_follow_their_rows_not_a_separate_column(self):
        # MSFT FY2025 p29: bullet glyphs are separate native PDF text lines.
        items = []
        for i, y in enumerate((76.41, 118.16, 135.91), 1):
            items.extend([line("•", 64, y, 68.06, y + 10, 2*i),
                          line(f"Bullet paragraph {i}", 78.994, y - .008, 555, y + 9.992, 2*i+1)])
        ordered = order_lines(items, 594, 774)
        self.assertEqual([x["text"] for x in ordered],
                         [text for i in range(1, 4) for text in ("•", f"Bullet paragraph {i}")])
        self.assertEqual([x["text"] for x in lines_to_blocks(ordered)],
                         [f"• Bullet paragraph {i}" for i in range(1, 4)])


class MarginTests(unittest.TestCase):
    def pages(self, count=4):
        return [{"page_number": n, "width": 600, "height": 800,
                 "lines": [line("ACME 2025 Annual Report", 40, 24, 250, 34, 0),
                           line("Repeated body text", 50, 300, 350, 310, 1),
                           line(str(n), 540, 772, 550, 782, 2)]} for n in range(1, count + 1)]

    def test_recurring_headers_and_page_numbers_retained_but_labeled(self):
        pages = self.pages()
        annotate_margins(pages)
        for page in pages:
            self.assertEqual([x["kind"] for x in page["lines"]], ["header", "text", "footer"])
            self.assertEqual(page["lines"][2]["removal_reason"], "recurring_margin_text")
            self.assertEqual(len(page["lines"]), 3)

    def test_unique_cover_title_and_body_margin_paragraph_survive(self):
        pages = self.pages()
        pages[0]["lines"].append(line("A bold future", 40, 8, 500, 28, 8, font=24))
        for page in pages:
            page["lines"].extend([line("The Company operates across global markets", 50, 55, 500, 65, 6, 0),
                                  line("and supports customers through its subsidiaries.", 50, 67, 500, 77, 6, 1)])
        annotate_margins(pages)
        self.assertEqual(pages[0]["lines"][3]["kind"], "text")
        for page in pages:
            self.assertTrue(all(x["kind"] == "text" for x in page["lines"] if x["block_id"] == 6))

    def test_same_text_at_inconsistent_margin_positions_is_preserved(self):
        pages = self.pages(3)
        for page, y in zip(pages, (2, 25, 50)):
            page["lines"][0]["bbox"] = [40, y, 250, y + 10]
        annotate_margins(pages)
        self.assertTrue(all(page["lines"][0]["kind"] == "text" for page in pages))

    def test_prominent_repeated_body_title_is_not_a_running_header(self):
        pages = self.pages()
        for page in pages:
            page["lines"][0] = line("Financial statements", 40, 24, 300, 44, 0, font=20)
            page["lines"][1]["text"] = "This is a normal body paragraph in the annual report."
        annotate_margins(pages)
        self.assertTrue(all(page["lines"][0]["kind"] == "text" for page in pages))

    def test_bold_same_size_body_section_titles_are_preserved(self):
        pages = self.pages()
        for page in pages:
            page["lines"][0].update(text="Opposing Statement of the Board", bold=True)
        annotate_margins(pages)
        self.assertTrue(all(page["lines"][0]["kind"] == "text" for page in pages))

    def test_arbitrary_financial_values_in_footer_are_not_folios(self):
        pages = self.pages(3)
        for page, value in zip(pages, (500, 600, 900)):
            page["lines"][2]["text"] = str(value)
        annotate_margins(pages)
        self.assertTrue(all(page["lines"][2]["kind"] == "text" for page in pages))

    def test_roman_folios_following_pdf_page_sequence_are_removed(self):
        pages = self.pages(3)
        for page, value in zip(pages, ("iii", "iv", "v")):
            page["lines"][2]["text"] = value
        annotate_margins(pages)
        self.assertTrue(all(page["lines"][2]["kind"] == "footer" for page in pages))

    def test_printed_folio_resets_can_have_different_offsets(self):
        pages = self.pages(6)
        for page, value in zip(pages, (1, 2, 3, 1, 2, 3)):
            page["lines"][2]["text"] = f"Page {value} of 3"
        annotate_margins(pages)
        self.assertTrue(all(page["lines"][2]["kind"] == "footer" for page in pages))

    def test_one_page_alone_is_not_evidence_of_recurrence(self):
        page = self.pages(1)
        annotate_margins(page)
        self.assertTrue(all(x["kind"] == "text" for x in page[0]["lines"]))

    def test_alternating_page_headers_are_detected_independently(self):
        pages = self.pages(6)
        for page in pages:
            page["lines"][0]["text"] = "Annual Report" if page["page_number"] % 2 else "ACME Corporation"
        annotate_margins(pages)
        self.assertTrue(all(page["lines"][0]["kind"] == "header" for page in pages))


if __name__ == "__main__":
    unittest.main()
