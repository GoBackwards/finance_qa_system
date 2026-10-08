"""Conservative, geometry-based layout handling for born-digital reports.

This is a transparent heuristic baseline, not DeepDoc's learned layout model.
Coordinates are PDF points with a top-left origin.  Removed margin lines remain
in the page's ``lines`` array, and all ordering functions retain source fields.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from typing import Any

Line = dict[str, Any]
Page = dict[str, Any]


def _font(line: Line) -> float:
    return float(line.get("font_size") or max(1, line["bbox"][3] - line["bbox"][1]))


def _margin_signature(text: str) -> str:
    text = re.sub(r"\s+", " ", text.casefold()).strip(" |•·–—-")
    if re.fullmatch(r"(?:page\s+)?[ivxlcdm]+", text):
        return "<page-number>"
    text = re.sub(r"\b\d+\b", "#", text)
    if re.fullmatch(r"(?:page\s+)?#(?:\s+(?:of|/)\s+#)?", text):
        return "<page-number>"
    return text


def _folio_number(text: str) -> int | None:
    """Read a bare/page-prefixed folio, excluding signs, currency and dates."""
    value = re.sub(r"^page\s+", "", text.strip().casefold())
    match = re.fullmatch(r"(\d+)(?:\s*(?:of|/)\s*\d+)?", value)
    if match:
        return int(match.group(1))
    if not value or not re.fullmatch(r"m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})", value):
        return None
    numbers = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    return sum(-numbers[char] if i + 1 < len(value) and numbers[char] < numbers[value[i + 1]]
               else numbers[char] for i, char in enumerate(value))


def _paragraph_like(line: Line, lines: list[Line], side: str, height: float) -> bool:
    """Protect ordinary paragraphs extending into the margin candidate band."""
    x0, y0, x1, y1 = line["bbox"]
    font = _font(line)
    if len(line.get("text", "").split()) > 18 or len(line.get("text", "")) > 120:
        return True
    for other in lines:
        if other is line or not other.get("text", "").strip():
            continue
        ox0, oy0, ox1, oy1 = other["bbox"]
        gap = oy0 - y1 if side == "header" else y0 - oy1
        same_block = line.get("block_id") is not None and line.get("block_id") == other.get("block_id")
        aligned = abs(ox0 - x0) < max(5, font) and min(x1, ox1) > max(x0, ox0)
        bodyward = oy1 > height * 0.08 if side == "header" else oy0 < height * 0.92
        similar_font = abs(_font(other) - font) <= max(1.5, font * 0.15)
        if -1 <= gap <= font * 0.75 and aligned and similar_font:
            if same_block or bodyward:
                return True
    return False


def annotate_margins(pages: list[Page]) -> list[Page]:
    """Label recurring marginal text in place, without discarding any lines.

    A signature must recur on at least three distinct pages (two for a two-page
    document), at a stable relative vertical position and comparable font size.
    Recurrence need not cover the whole report, so section headers and alternating
    recto/verso headers work.  Numbers are normalized, but body-like paragraphs,
    unique titles, and isolated currency/numeric data are protected.
    """
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    required_pages = min(3, max(2, len(pages)))
    for page in pages:
        height = float(page["height"])
        body_fonts = [_font(line) for line in page["lines"]
                      if height * 0.12 < line["bbox"][1] < height * 0.88
                      and len(line.get("text", "").strip()) >= 25]
        body_font = statistics.median(body_fonts) if body_fonts else None
        for line in page["lines"]:
            # Preserve types supplied by the extractor, e.g. isolated table regions.
            if line.get("kind") in {"header", "footer"}:
                line["kind"] = "text"
            line.setdefault("kind", "text")
            line.pop("removal_reason", None)
            line.pop("margin_signature", None)
            line.pop("margin_recurrence_pages", None)
            if line["kind"] != "text":
                continue
            text = line.get("text", "").strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            side = "header" if y1 <= height * 0.085 else "footer" if y0 >= height * 0.915 else None
            if side is None or _paragraph_like(line, page["lines"], side, height):
                continue
            if body_font is not None and _font(line) > max(body_font * 1.4, body_font + 3):
                # Prominent section/cover titles are content even if repeated.
                continue
            signature = _margin_signature(text)
            if not signature or (not any(c.isalpha() for c in text) and signature != "<page-number>"):
                continue
            if line.get("bold") and signature != "<page-number>":
                # Repeated section titles can occupy exactly the same margin
                # geometry as running headers. Retaining bold text is safer.
                continue
            candidates[(side, signature)].append({
                "line": line, "page": page["page_number"],
                "y": (y0 + y1) / (2 * height), "font": _font(line),
                "folio": _folio_number(text) if signature == "<page-number>" else None,
            })

    for (side, signature), matches in candidates.items():
        # Cluster by geometry instead of requiring identical integer coordinates.
        clusters: list[list[dict[str, Any]]] = []
        for match in sorted(matches, key=lambda item: item["y"]):
            cluster = next((group for group in clusters
                            if abs(match["y"] - statistics.median(x["y"] for x in group)) <= 0.012
                            and abs(match["font"] - statistics.median(x["font"] for x in group))
                            <= max(1.5, match["font"] * 0.25)), None)
            if cluster is None:
                clusters.append([match])
            else:
                cluster.append(match)
        qualifying_clusters = []
        for cluster in clusters:
            if signature == "<page-number>":
                # A real folio advances with physical PDF pages, even if the
                # printed numbering starts later or resets in a new section.
                # Geometry and normalized digits alone would erase table values.
                offsets: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for match in cluster:
                    if match["folio"] is not None:
                        offsets[match["folio"] - match["page"]].append(match)
                qualifying_clusters.extend(offsets.values())
            else:
                qualifying_clusters.append(cluster)
        for cluster in qualifying_clusters:
            occurrences = len({match["page"] for match in cluster})
            if occurrences < required_pages:
                continue
            for match in cluster:
                line = match["line"]
                line["kind"] = side
                line["removal_reason"] = "recurring_margin_text"
                line["margin_signature"] = signature
                line["margin_recurrence_pages"] = occurrences
    return pages


def _axis_gaps(lines: list[Line], axis: int) -> list[tuple[float, float]]:
    intervals = sorted((float(line["bbox"][axis]), float(line["bbox"][axis + 2])) for line in lines)
    if not intervals:
        return []
    end = intervals[0][1]
    gaps = []
    for start, next_end in intervals[1:]:
        if start > end + 0.5:
            gaps.append((start - end, (start + end) / 2))
        end = max(end, next_end)
    return gaps


def _column_partition(left: list[Line], right: list[Line], font: float) -> bool:
    """A gutter must separate substantial, vertically overlapping content.

    Detached bullet glyphs and vertically stacked, differently aligned headings
    create whitespace too, but neither should become a separate reading column.
    """
    if not left or not right:
        return False
    if min(max(line["bbox"][2] - line["bbox"][0] for line in side)
           for side in (left, right)) < max(18, font * 3):
        return False
    tops = [min(line["bbox"][1] for line in side) for side in (left, right)]
    bottoms = [max(line["bbox"][3] for line in side) for side in (left, right)]
    overlap = min(bottoms) - max(tops)
    return overlap >= 0.2 * min(bottom - top for top, bottom in zip(tops, bottoms))


def _row_order(lines: list[Line]) -> list[Line]:
    """Sort a single region, allowing slight glyph-height/baseline differences."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        center = (line["bbox"][1] + line["bbox"][3]) / 2
        if rows:
            anchor = rows[-1][0]
            anchor_center = (anchor["bbox"][1] + anchor["bbox"][3]) / 2
            tolerance = min(line["bbox"][3] - line["bbox"][1], anchor["bbox"][3] - anchor["bbox"][1]) * 0.45
            if abs(center - anchor_center) <= tolerance:
                rows[-1].append(line)
                continue
        rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda item: item["bbox"][0])]


def _inferred_gutter(lines: list[Line], min_gap: float) -> tuple[float, list[Line]] | None:
    """Find a gutter interrupted by a small number of spanning lines."""
    if len(lines) < 5:
        return None
    edges = sorted({float(line["bbox"][axis]) for line in lines for axis in (0, 2)})
    region_width = edges[-1] - edges[0]
    best = None
    for start, end in zip(edges, edges[1:]):
        if end - start < min_gap:
            continue
        cut = (start + end) / 2
        left = [line for line in lines if line["bbox"][2] <= cut]
        right = [line for line in lines if line["bbox"][0] >= cut]
        spanning = [line for line in lines if line["bbox"][0] < cut < line["bbox"][2]]
        if min(len(left), len(right)) < 2 or not spanning or len(spanning) > max(2, len(lines) * 0.22):
            continue
        if min(max(x["bbox"][2] - x["bbox"][0] for x in left),
               max(x["bbox"][2] - x["bbox"][0] for x in right)) < region_width * 0.12:
            continue
        ltop, lbottom = min(x["bbox"][1] for x in left), max(x["bbox"][3] for x in left)
        rtop, rbottom = min(x["bbox"][1] for x in right), max(x["bbox"][3] for x in right)
        overlap = min(lbottom, rbottom) - max(ltop, rtop)
        if overlap < min(lbottom - ltop, rbottom - rtop) * 0.35:
            continue
        score = (2 * min(len(left), len(right)) - 2 * len(spanning), end - start)
        if best is None or score > best[0]:
            best = (score, cut, spanning)
    return (best[1], best[2]) if best is not None else None


def order_lines(lines: list[Line], width: float, height: float) -> list[Line]:
    """Return copied lines in region-aware reading order using recursive XY cuts.

    True vertical whitespace takes precedence over row sorting.  Large horizontal
    gaps divide sections; interrupted gutters isolate full-width titles.  Column
    IDs identify branches within the page, and ``layout_group`` identifies each
    terminal region.  No lines are dropped; callers should exclude margins first.
    """
    output: list[Line] = []
    next_column = 0
    next_group = 0
    # Spread PDFs can contain six narrow columns; page-width percentages would
    # demand larger gutters merely because two printed pages share one PDF page.
    text_fonts = [_font(line) for line in lines if line.get("kind", "text") == "text"]
    median_font = statistics.median(text_fonts) if text_fonts else 10.0
    min_gutter = max(6.0, min(12.0, median_font * 0.8))

    def emit(items: list[Line], column: int, warnings: list[str] | None = None) -> None:
        nonlocal next_group
        next_group += 1
        for source in _row_order(items):
            line = dict(source)
            line.update(column=column, layout_group=next_group, reading_order=len(output))
            line["layout_warnings"] = list(dict.fromkeys(list(source.get("layout_warnings", [])) + (warnings or [])))
            output.append(line)

    def visit(items: list[Line], column: int = 0, depth: int = 0) -> None:
        nonlocal next_column
        if not items:
            return
        if len(items) == 1 or depth > 40:
            emit(items, column, ["layout_recursion_limit"] if depth > 40 else None)
            return
        xgaps = [gap for gap in _axis_gaps(items, 0) if gap[0] >= min_gutter]
        for _, cut in sorted(xgaps, reverse=True):
            left = [line for line in items if line["bbox"][2] <= cut]
            right = [line for line in items if line["bbox"][0] >= cut]
            if _column_partition(left, right, median_font):
                next_column += 1
                left_column = next_column
                next_column += 1
                right_column = next_column
                visit(left, left_column, depth + 1)
                visit(right, right_column, depth + 1)
                return
        median_height = statistics.median(max(1, line["bbox"][3] - line["bbox"][1]) for line in items)
        ygaps = [gap for gap in _axis_gaps(items, 1) if gap[0] >= max(5, median_height * 0.7)]
        if ygaps:
            _, cut = max(ygaps)
            visit([line for line in items if line["bbox"][3] <= cut], column, depth + 1)
            visit([line for line in items if line["bbox"][1] >= cut], column, depth + 1)
            return
        inferred = _inferred_gutter(items, min_gutter)
        if inferred:
            _, spanning = inferred
            remaining = [line for line in items if line not in spanning]
            bands: list[list[Line]] = []
            for line in sorted(spanning, key=lambda item: item["bbox"][1]):
                if bands and line["bbox"][1] <= max(item["bbox"][3] for item in bands[-1]) + 1:
                    bands[-1].append(line)
                else:
                    bands.append([line])
            for band in bands:
                top = min(line["bbox"][1] for line in band)
                bottom = max(line["bbox"][3] for line in band)
                before = [line for line in remaining if (line["bbox"][1] + line["bbox"][3]) / 2 < top]
                visit(before, column, depth + 1)
                remaining = [line for line in remaining if line not in before]
                overlaps = [line for line in remaining if line["bbox"][1] < bottom]
                emit(band, 0, ["overlapping_spanning_region"] if overlaps else None)
            visit(remaining, column, depth + 1)
            return
        emit(items, column)

    visit(lines)
    return output


def lines_to_blocks(ordered_lines: list[Line]) -> list[dict[str, Any]]:
    """Group nearby lines only within one page, column, and layout region.

    Text is joined conservatively without changing hyphenation or numerals.
    ``lines`` retains the complete source dictionaries for citation/debugging;
    ``line_ids`` is a compact list of native (block_id, line_id) references.
    """
    blocks: list[dict[str, Any]] = []
    for line in ordered_lines:
        previous = blocks[-1]["lines"][-1] if blocks else None
        join = False
        if previous is not None:
            gap = line["bbox"][1] - previous["bbox"][3]
            font = max(_font(line), _font(previous))
            same_region = all(line.get(key) == previous.get(key)
                              for key in ("page_number", "column", "layout_group", "kind"))
            same_block = line.get("block_id") is not None and line.get("block_id") == previous.get("block_id")
            aligned = abs(line["bbox"][0] - previous["bbox"][0]) <= font * 1.8
            similar_font = abs(_font(line) - _font(previous)) <= max(1, font * 0.12)
            inline_marker = bool(re.fullmatch(r"[\u2022\u203a\u25b8\u25cf\u25aa\u2013\u2014*-]", previous.get("text", "").strip())) and \
                0 <= line["bbox"][0] - previous["bbox"][2] <= font * 3 and \
                abs(sum(line["bbox"][i] for i in (1, 3)) - sum(previous["bbox"][i] for i in (1, 3))) <= font
            join = same_region and line.get("kind", "text") == "text" and similar_font and \
                (inline_marker or (-font * 0.2 <= gap <= font * 0.8 and (same_block or aligned)))
        if not join:
            blocks.append({
                "kind": line.get("kind", "text"), "text": "", "bbox": list(line["bbox"]),
                "column": line.get("column", 0), "layout_group": line.get("layout_group", 0),
                "reading_order": len(blocks), "lines": [], "line_ids": [], "layout_warnings": [],
                **({"page_number": line["page_number"]} if "page_number" in line else {}),
                **({"table_id": line["table_id"]} if "table_id" in line else {}),
            })
        block = blocks[-1]
        block["lines"].append(line)
        block["line_ids"].append({"block_id": line.get("block_id"), "line_id": line.get("line_id"),
                                  **({"source_line_id": line["source_line_id"]} if "source_line_id" in line else {})})
        block["text"] = " ".join(item.get("text", "").strip() for item in block["lines"] if item.get("text", "").strip())
        block["bbox"] = [min(block["bbox"][0], line["bbox"][0]), min(block["bbox"][1], line["bbox"][1]),
                         max(block["bbox"][2], line["bbox"][2]), max(block["bbox"][3], line["bbox"][3])]
        block["layout_warnings"] = list(dict.fromkeys(block["layout_warnings"] + line.get("layout_warnings", [])))
    return blocks
