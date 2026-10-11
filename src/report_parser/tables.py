"""Conservative, coordinate-preserving table extraction for annual reports.

Ruled tables use pdfplumber's line strategy. Borderless candidates must have
repeated, aligned numeric columns and a separate label column before a small
crop is sent to pdfplumber. This deliberately prefers missed tables over turning
two newspaper-style columns into an invented financial statement.

Confidence values are ranking heuristics, not calibrated probabilities. Source
cell strings are never converted to numbers or assigned inferred units.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import median
from typing import Any, Iterable


_VALUE_RE = re.compile(
    r"^(?:[$€£¥])?\(?[+\-−–]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"%?\)?(?:%|[$€£¥])?(?:\([a-zA-Z0-9]+\))?$"
)
_EMPTY_VALUE_RE = re.compile(r"^[$€£¥]?\(?[—–−-]\)?%?$")
_UNIT_RE = re.compile(
    r"\b(?:in (?:thousands|millions|billions)|except per.share|"
    r"dollars in|shares in|unaudited|consolidated|balance sheets?|"
    r"statements? of|cash flows?|years? ended)\b", re.I
)
_MAX_WORDS = 4000
_MAX_CANDIDATES = 12
_DIGIT_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _number(value: str | None) -> bool:
    """Recognize a displayed value without changing its original spelling."""
    if value is None:
        return False
    compact = re.sub(r"\s+", "", value)
    return bool(_VALUE_RE.fullmatch(compact) or _EMPTY_VALUE_RE.fullmatch(compact))


def _area(box: Iterable[float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _overlap(a: Iterable[float], b: Iterable[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return _area((max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)))


def _bbox(words: list[dict]) -> tuple[float, float, float, float]:
    return (min(w["x0"] for w in words), min(w["top"] for w in words),
            max(w["x1"] for w in words), max(w["bottom"] for w in words))


def _word_rows(words: list[dict]) -> list[dict]:
    """Group baseline neighbours, preserving complete words and coordinates."""
    rows: list[dict] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        centre = (word["top"] + word["bottom"]) / 2
        height = word["bottom"] - word["top"]
        match = None
        for row in reversed(rows[-4:]):
            if abs(centre - row["centre"]) <= max(2.0, min(height, row["height"]) * 0.3):
                match = row
                break
        if match is None:
            rows.append({"centre": centre, "height": height, "words": [word]})
        else:
            match["words"].append(word)
    for row in rows:
        row["words"].sort(key=lambda w: w["x0"])
        row["bbox"] = _bbox(row["words"])
    return sorted(rows, key=lambda r: r["bbox"][1])


def _numeric_tail(row: dict) -> list[dict]:
    """Find numeric chunks after a label separated by a visible gutter."""
    words = row["words"]
    chunks: list[list[dict]] = []
    for word in words:
        # A prefix currency symbol starts the NEXT cell even if a tightly
        # spaced previous amount is closer to it than its own amount.
        currency_prefix = word["text"] in {"$", "€", "£", "¥"}
        if not chunks or currency_prefix or word["x0"] - chunks[-1][-1]["x1"] > max(9.0, row["height"] * 0.8):
            chunks.append([word])
        else:
            chunks[-1].append(word)
    # Financial statements often place a currency symbol at the left of a
    # numeric cell and right-align its amount. Keep that symbol with its value.
    joined: list[list[dict]] = []
    index = 0
    while index < len(chunks):
        text = " ".join(w["text"] for w in chunks[index])
        if text in {"$", "€", "£", "¥", "("} and index + 1 < len(chunks):
            following = " ".join(w["text"] for w in chunks[index + 1])
            if _number(text + following):
                joined.append(chunks[index] + chunks[index + 1])
                index += 2
                continue
        joined.append(chunks[index])
        index += 1
    chunks = joined
    tail = []
    for chunk in reversed(chunks):
        text = " ".join(w["text"] for w in chunk)
        if not _number(text):
            break
        tail.append({"text": text, "bbox": _bbox(chunk)})
    tail.reverse()
    if not tail or len(tail) == len(chunks):
        return []
    label_words = [w for w in words if w["x1"] <= tail[0]["bbox"][0]]
    if not any(re.search(r"[A-Za-z]", w["text"]) for w in label_words):
        return []
    # A number at the end of an ordinary sentence has no table-like gap.
    if tail[0]["bbox"][0] - max(w["x1"] for w in label_words) < 10:
        return []
    return tail


def _borderless_candidates(words: list[dict], width: float, height: float) -> list[dict]:
    rows = _word_rows(words)
    seeds = []
    anchors: list[dict] = []
    for row_index, row in enumerate(rows):
        values = _numeric_tail(row)
        if not values:
            continue
        row["values"] = values
        row["index"] = row_index
        seeds.append(row)
        for value in values:
            right = value["bbox"][2]
            near = [a for a in anchors if abs(a["x"] - right) <= 5]
            if near:
                anchor = min(near, key=lambda a: abs(a["x"] - right))
                anchor["positions"].append(right)
                anchor["x"] = median(anchor["positions"])
                anchor["rows"].add(row_index)
            else:
                anchors.append({"x": right, "positions": [right], "rows": {row_index}})
    anchors = [a for a in anchors if len(a["rows"]) >= 3]
    if not anchors:
        return []
    seeds = [r for r in seeds if any(r["index"] in a["rows"] for a in anchors)]
    if len(seeds) < 3:
        return []
    groups: list[list[dict]] = []
    typical_height = median(r["height"] for r in seeds)
    for row in seeds:
        if not groups or row["bbox"][1] - groups[-1][-1]["bbox"][3] > max(38.0, typical_height * 3.5):
            groups.append([row])
        else:
            groups[-1].append(row)
    candidates = []
    for group in groups:
        if len(group) < 3:
            continue
        indices = {r["index"] for r in group}
        local_anchors = sorted(
            [a for a in anchors if len(a["rows"] & indices) >= 3], key=lambda a: a["x"]
        )
        if not local_anchors:
            continue
        # At least half the candidate rows must carry a value in every inferred
        # column; sporadic narrative numbers do not form a numeric grid.
        local_anchors = [a for a in local_anchors if len(a["rows"] & indices) >= max(3, len(group) * 0.45)]
        if not local_anchors:
            continue
        x0 = min(r["bbox"][0] for r in group)
        x1 = max(v["bbox"][2] for r in group for v in r["values"])
        y0 = group[0]["bbox"][1]
        y1 = group[-1]["bbox"][3]
        columns = []
        for anchor in local_anchors:
            matched = [v["bbox"] for r in group for v in r["values"]
                       if abs(v["bbox"][2] - anchor["x"]) <= 5]
            columns.append((min(b[0] for b in matched), max(b[2] for b in matched)))
        first_value_x = columns[0][0]
        labels = [w for r in group for w in r["words"] if w["x1"] < first_value_x - 5]
        if not labels:
            continue
        label_right = max(w["x1"] for w in labels)
        if first_value_x - label_right < 7:
            continue
        # Include adjacent, possibly multirow headers and literal unit lines.
        for previous in reversed(rows[:group[0]["index"]]):
            if y0 - previous["bbox"][3] > max(21.0, typical_height * 1.8):
                break
            previous_text = " ".join(w["text"] for w in previous["words"])
            if _UNIT_RE.search(previous_text) and not re.search(r"\byears? ended\b", previous_text, re.I):
                break  # Captions and unit notes belong in context, not split cells.
            inside = [w for w in previous["words"] if w["x0"] >= x0 - 3 and w["x1"] <= x1 + 8]
            in_value_area = any(w["x0"] >= (label_right + first_value_x) / 2 for w in inside)
            if not inside or not (in_value_area or _UNIT_RE.search(previous_text)):
                break
            if group[0]["bbox"][1] - previous["bbox"][1] > 65:
                break
            y0 = previous["bbox"][1]
        # Explicit separators avoid text/text inventing columns inside labels.
        cuts = [x0 - 1, (label_right + first_value_x) / 2]
        cuts.extend((columns[i][1] + columns[i + 1][0]) / 2 for i in range(len(columns) - 1))
        cuts.append(x1 + 2)
        if any(b - a < 8 for a, b in zip(cuts, cuts[1:])):
            continue
        box = [max(0, x0 - 1), max(0, y0 - 2), min(width, x1 + 2), min(height, y1 + 2)]
        cuts[0], cuts[-1] = box[0], box[2]
        candidates.append({"bbox": box, "cuts": cuts})
    if len(candidates) > _MAX_CANDIDATES:
        raise ValueError(f"Table candidate budget exceeded ({len(candidates)} > {_MAX_CANDIDATES})")
    return candidates


def _is_financial_table(rows: list[list[str | None]]) -> bool:
    populated = [r for r in rows if any(v is not None and v.strip() for v in r)]
    if len(populated) < 3 or max((len(r) for r in populated), default=0) < 2:
        return False
    if any(v and v.strip().lower() in {"page", "pages", "table of contents", "contents"}
           for row in populated[:2] for v in row):
        return False
    data_rows = sum(any(_number(v) for v in r[1:]) and any(
        v and re.search(r"[A-Za-z]", v) for v in r) for r in populated)
    numeric_count = sum(_number(v) for r in populated for v in r)
    nonempty = [v for r in populated for v in r if v and v.strip()]
    long_cells = sum(len(v) > 160 for v in nonempty)
    return data_rows >= 2 and numeric_count >= 3 and long_cells <= max(1, len(nonempty) * 0.15)


def _markdown(rows: list[list[str | None]]) -> str:
    """Use neutral column headings so source header rows stay unmerged."""
    count = max((len(row) for row in rows), default=0)
    if not count:
        return ""
    def escape(value: str | None) -> str:
        return "" if value is None else value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
    output = ["| " + " | ".join(f"Column {i + 1}" for i in range(count)) + " |",
              "| " + " | ".join("---" for _ in range(count)) + " |"]
    output += ["| " + " | ".join(escape(v) for v in row + [None] * (count - len(row))) + " |" for row in rows]
    return "\n".join(output)


def _context(lines: list[dict], bbox: list[float]) -> tuple[str | None, list[dict]]:
    context = []
    for line in lines:
        box, text = line.get("bbox"), line.get("text", "")
        if not box or not text or len(box) != 4:
            continue
        horizontal_overlap = min(box[2], bbox[2]) - max(box[0], bbox[0])
        if horizontal_overlap <= 0:
            continue
        before = bbox[1] - 55 <= box[3] <= bbox[1] + 2
        after = bbox[3] - 2 <= box[1] <= bbox[3] + 28
        inside_unit = _UNIT_RE.search(text) and _overlap(box, bbox) > 0
        if before or after or inside_unit:
            context.append({"text": text, "bbox": [round(float(v), 3) for v in box],
                            "position": "above" if before else "below" if after else "inside"})
    # Read each visual line left to right: year headers over adjacent columns ('2024' at x=441, '2023' at
    # x=511) can differ by a fraction of a point in top edge, and a plain (top, x) sort reversed them.
    context.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    lines_ = []
    for item in context:
        middle = (item["bbox"][1] + item["bbox"][3]) / 2
        if lines_ and abs(middle - lines_[-1][0]) <= 2.0:
            lines_[-1][1].append(item)
        else:
            lines_.append((middle, [item]))
    context = [item for _, line in lines_ for item in sorted(line, key=lambda item: item["bbox"][0])]
    above = [item["text"] for item in context if item["position"] == "above"]
    return ("\n".join(above) if above else None), context


def _source_rows(lines: list[dict]) -> tuple[list[dict], str]:
    """Keep whole native lines while ordering neighbours by baseline and x."""
    words = [{"text": line["text"], "x0": line["bbox"][0], "top": line["bbox"][1],
              "x1": line["bbox"][2], "bottom": line["bbox"][3], "source": line}
             for line in lines]
    grouped = _word_rows(words)
    ordered = [word["source"] for row in grouped for word in row["words"]]
    text = "\n".join("\t".join(word["text"] for word in row["words"]) for row in grouped)
    return ordered, text


def _digit_tokens(texts: Iterable[str]) -> Counter:
    # A trailing comma can be prose punctuation ("June 30,"); internal commas
    # and decimal points remain significant. Process each cell separately so
    # split values cannot be accidentally recombined by joining the cells.
    return Counter(token.rstrip(",") for text in texts for token in _DIGIT_TOKEN_RE.findall(text))


def _record(table: Any, strategy: str, lines: list[dict], warnings: list[str]) -> dict | None:
    extracted_rows = table.extract(x_tolerance=2, y_tolerance=3)
    # Text-strategy edges produce empty interline spacer rows. Remove only
    # wholly empty rows; empty/null cells within actual rows remain untouched.
    row_indices = [i for i, row in enumerate(extracted_rows) if any(value is not None and value.strip() for value in row)]
    rows = [list(extracted_rows[i]) for i in row_indices]
    if not _is_financial_table(rows):
        return None
    bbox = [round(float(v), 3) for v in table.bbox]
    cells = []
    table_rows = table.rows
    raw_rows = [list(row) for row in rows]
    row_texts = []
    recovered_spans = False
    for row_index, source_row_index in enumerate(row_indices):
        row = table_rows[source_row_index]
        row_bbox = row.bbox
        row_lines = [line for line in lines if line.get("bbox") and line.get("text")
                     and min(line["bbox"][3], row_bbox[3]) - max(line["bbox"][1], row_bbox[1])
                     >= min(line["bbox"][3] - line["bbox"][1], row_bbox[3] - row_bbox[1]) * 0.6
                     and min(line["bbox"][2], row_bbox[2]) > max(line["bbox"][0], row_bbox[0])]
        row_lines.sort(key=lambda line: line["bbox"][0])
        row_texts.append(" ".join(line["text"] for line in row_lines))
        recovered_bbox = None
        if strategy != "lines" and len(row.cells) > 1 and row.cells[0] is not None:
            first_cut = row.cells[0][2]
            spanning = [line for line in row_lines if line["bbox"][0] < first_cut - 5
                        and line["bbox"][2] > first_cut + 1]
            # A single text-only source line crossing an inferred divider is a
            # spanning label, not financial values. Recover its exact source
            # text; keep the original cell extraction for audit. Never merge
            # numeric rows or adjacent physical header rows.
            if len(spanning) == 1 and len(row_lines) == 1 and not any(_number(v) for v in rows[row_index][1:]):
                rows[row_index] = [spanning[0]["text"]] + [None] * (len(rows[row_index]) - 1)
                recovered_bbox = [min(row_bbox[0], spanning[0]["bbox"][0]), row_bbox[1],
                                  max(first_cut, spanning[0]["bbox"][2]), row_bbox[3]]
                recovered_spans = True
        for column_index, cell in enumerate(row.cells):
            value = rows[row_index][column_index] if row_index < len(rows) and column_index < len(rows[row_index]) else None
            if recovered_bbox is not None:
                cell = recovered_bbox if column_index == 0 else None
            cells.append({"row": row_index, "column": column_index,
                          "bbox": None if cell is None else [round(float(v), 3) for v in cell],
                          "text": value})
    caption, context = _context(lines, bbox)
    source_lines = [{"text": line["text"], "bbox": [round(float(v), 3) for v in line["bbox"]]}
                    for line in lines if line.get("text") and line.get("bbox")
                    and _overlap(line["bbox"], bbox) / max(1, _area(line["bbox"])) >= 0.5]
    source_lines, source_text = _source_rows(source_lines)
    notes = list(warnings)
    if any(cell is None for row in rows for cell in row):
        notes.append("Missing or spanning cells are preserved as null; headers are not merged.")
    if strategy != "lines":
        notes.append("Borderless table boundaries are inferred from repeated numeric alignment; review complex headers.")
    if recovered_spans:
        notes.append("Spanning text-only rows retain the complete source line in the first cell; original extracted cells are in raw_rows.")
    result = {"bbox": bbox, "rows": rows, "cells": cells,
            "markdown": _markdown(rows), "strategy": strategy, "warnings": notes,
            "confidence": 0.88 if strategy == "lines" else 0.64,
            "caption": caption, "context": context, "source_lines": source_lines,
            "source_text": source_text, "row_source_texts": row_texts}
    if recovered_spans:
        result["raw_rows"] = raw_rows
    native_numbers = _digit_tokens(line["text"] for line in source_lines)
    cell_numbers = _digit_tokens(value for row in rows for value in row if value)
    missing = native_numbers - cell_numbers
    unexpected = cell_numbers - native_numbers
    reliable = bool(native_numbers) and not missing and not unexpected
    result["structured_reliable"] = reliable
    result["numeric_integrity"] = {"passed": reliable, "missing_from_cells": dict(missing),
                                   "unexpected_in_cells": dict(unexpected)}
    if not reliable:
        result["candidate_rows"] = result["rows"]
        result["candidate_cells"] = result["cells"]
        result["rows"], result["cells"] = [], []
        # Do not render suspect values as a table in the default reading stream.
        # Debug candidates remain explicitly segregated for later review.
        fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", source_text)), default=0))
        result["markdown"] = ("Table structure requires review: numeric values did not match the native source text. "
                              "Original source text follows.\n\n" + fence + "text\n" + source_text + "\n" + fence)
        result["warnings"].append("numeric_integrity_failed: using original source text; candidate_rows and candidate_cells are debug only.")
    return result


def extract_tables(
    plumber_page: Any, pymupdf_lines: list[dict], page_width: float, page_height: float,
) -> list[dict]:
    """Extract financial tables in top-left PDF page coordinates.

    The caller owns page-level exception/warning handling and closing pages.
    A matching CropBox is translated into PyMuPDF's crop-relative coordinates.
    Other dimension mismatches raise instead of producing misleading citations.
    """
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in (page_width, page_height)):
        raise ValueError("Page dimensions must be finite and positive")
    if getattr(plumber_page, "rotation", 0):
        raise ValueError("Table extraction on rotated PDF pages requires orientation normalization; retain source text")
    page_bbox = tuple(plumber_page.bbox)
    cropbox = getattr(plumber_page, "cropbox", None)
    matching_crop = (cropbox and abs(cropbox[2] - cropbox[0] - page_width) <= 2
                     and abs(cropbox[3] - cropbox[1] - page_height) <= 2
                     and cropbox[0] >= page_bbox[0] and cropbox[1] >= page_bbox[1]
                     and cropbox[2] <= page_bbox[2] and cropbox[3] <= page_bbox[3])
    if matching_crop:
        # Always crop, even when dimensions match or differ by less than the
        # tolerance. PDF content streams may include hidden neighbouring pages
        # outside the visible box; pdfplumber otherwise exposes their objects.
        plumber_page = plumber_page.crop(tuple(cropbox))
    elif abs(float(plumber_page.width) - page_width) > 2 or abs(float(plumber_page.height) - page_height) > 2:
        raise ValueError("pdfplumber/PyMuPDF page dimensions differ; table coordinates need transformation")
    else:
        plumber_page = plumber_page.crop(page_bbox)
    origin_x, origin_y = plumber_page.bbox[:2]
    native_lines = []
    for line in pymupdf_lines:
        shifted = dict(line)
        if line.get("bbox"):
            b = line["bbox"]
            shifted["bbox"] = [b[0] + origin_x, b[1] + origin_y, b[2] + origin_x, b[3] + origin_y]
        native_lines.append(shifted)
    records = []
    words = plumber_page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
    if not words:
        return []
    if len(words) > _MAX_WORDS:
        raise ValueError(f"Table extraction word budget exceeded ({len(words)} > {_MAX_WORDS})")
    # Cap pathological vector artwork before pdfplumber's pairwise intersections.
    if len(plumber_page.edges) > 2500:
        raise ValueError("Table extraction edge budget exceeded (>2500 edges)")
    settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines",
                "snap_tolerance": 3, "join_tolerance": 3,
                "intersection_tolerance": 4, "edge_min_length": 8}
    for table in plumber_page.find_tables(table_settings=settings):
        result = _record(table, "lines", native_lines, [])
        if result is not None:
            records.append(result)
    for candidate in _borderless_candidates(words, page_width + origin_x, page_height + origin_y):
        candidate["bbox"][0] = max(origin_x, candidate["bbox"][0])
        candidate["bbox"][1] = max(origin_y, candidate["bbox"][1])
        candidate["bbox"][2] = min(plumber_page.bbox[2], candidate["bbox"][2])
        candidate["bbox"][3] = min(plumber_page.bbox[3], candidate["bbox"][3])
        candidate["cuts"][0] = candidate["bbox"][0]
        candidate["cuts"][-1] = candidate["bbox"][2]
        box, cuts = candidate["bbox"], candidate["cuts"]
        if (not all(math.isfinite(float(v)) for v in [*box, *cuts])
                or box[2] <= box[0] or box[3] <= box[1]
                or any(b <= a for a, b in zip(cuts, cuts[1:]))
                or any(cut < box[0] or cut > box[2] for cut in cuts)):
            continue
        if any(_overlap(candidate["bbox"], r["bbox"]) / max(1, _area(candidate["bbox"])) > 0.85 for r in records):
            continue
        crop = plumber_page.crop(tuple(candidate["bbox"]))
        settings = {"vertical_strategy": "explicit", "horizontal_strategy": "text",
                    "explicit_vertical_lines": candidate["cuts"], "min_words_horizontal": 1,
                    "intersection_tolerance": 4, "text_x_tolerance": 2,
                    "text_y_tolerance": 3, "snap_tolerance": 2}
        for table in crop.find_tables(table_settings=settings):
            result = _record(table, "numeric_alignment_text", native_lines, [])
            if result is not None:
                records.append(result)
    # Prefer ruled evidence for duplicate detections; do not fuse unrelated tables.
    kept: list[dict] = []
    for record in sorted(records, key=lambda r: (r["confidence"], _area(r["bbox"])), reverse=True):
        if any(_overlap(record["bbox"], old["bbox"]) / max(1, min(_area(record["bbox"]), _area(old["bbox"]))) > 0.8 for old in kept):
            continue
        kept.append(record)
    if origin_x or origin_y:
        for record in kept:
            for item in [record, *record["cells"], *record.get("candidate_cells", []),
                         *record["context"], *record["source_lines"]]:
                if item["bbox"] is not None:
                    b = item["bbox"]
                    item["bbox"] = [round(b[0] - origin_x, 3), round(b[1] - origin_y, 3),
                                    round(b[2] - origin_x, 3), round(b[3] - origin_y, 3)]
    return sorted(kept, key=lambda r: (r["bbox"][1], r["bbox"][0]))
