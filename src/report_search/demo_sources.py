"""Resolve demo citations through the checked report manifest, never user paths.

The public link is a manifest location hint. On-demand local downloads and page
previews additionally verify the complete PDF hash and actual page count.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit


class SourceCatalogError(ValueError):
    """The requested source cannot be bound to the manifest or local PDF."""


def _positive_integer(value):
    return type(value) is int and value > 0


def _public_url(row):
    valid = []
    for name in ("resolved_url", "download_url", "url"):
        value = row.get(name)
        if not isinstance(value, str) or not value or "\\" in value:
            continue
        if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
            continue
        try:
            parts = urlsplit(value)
            if (parts.scheme not in {"http", "https"} or not parts.hostname
                    or parts.username is not None or parts.password is not None
                    or "%" in parts.hostname):
                continue
            parts.port  # Reject malformed port declarations as well.
            valid.append((parts.scheme, urlunsplit(parts._replace(fragment=""))))
        except ValueError:
            continue
    return next((url for scheme, url in valid if scheme == "https"), valid[0][1] if valid else None)


@dataclass(frozen=True)
class _Report:
    company: str
    ticker: str
    fiscal_year: int
    source_file: str
    document_id: str
    sha256: str
    relative_path: str
    page_count: int
    public_url: str | None


class SourceCatalog:
    """Serve only validated reports listed in ``root/data/manifest.json``.

    Methods accept either original citation metadata or an answer citation
    containing it under ``citation``. Page numbers are one-based PDF pages.
    ``describe`` does not read the PDF; ``local_available`` means it exists, not
    that its bytes were verified. ``pdf_bytes`` and ``page_png`` always verify.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        manifest = self.root / "data" / "manifest.json"
        if not manifest.resolve().is_relative_to(self.root):
            raise SourceCatalogError("Manifest escapes the project root")
        rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise SourceCatalogError("Report manifest must be a list")
        self._reports = {}
        for row in rows:
            if not isinstance(row, dict):
                raise SourceCatalogError("Invalid report manifest entry")
            if row.get("status") != "validated":
                continue
            ticker, year = row.get("ticker"), row.get("fiscal_year")
            sha, pages, company = row.get("sha256"), row.get("page_count"), row.get("company")
            if (not isinstance(ticker, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,15}", ticker)
                    or not _positive_integer(year) or year > 9999
                    or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                    or not _positive_integer(pages) or not isinstance(company, str) or not company.strip()):
                raise SourceCatalogError("Invalid validated report metadata")
            stem = f"{ticker}_FY{year}_annual_report"
            relative = f"data/raw/{ticker}/{stem}.pdf"
            if row.get("relative_path") != relative:
                raise SourceCatalogError("Manifest report path is not canonical")
            identifier = f"{stem}-{sha[:12]}"
            if identifier in self._reports:
                raise SourceCatalogError("Duplicate report manifest identity")
            report = _Report(company, ticker, year, stem + ".pdf", identifier, sha,
                             relative, pages, _public_url(row))
            self._path(report)  # Reject existing links outside the raw corpus.
            self._reports[identifier] = report

    def _path(self, report):
        raw_root = (self.root / "data" / "raw").resolve()
        path = (self.root / report.relative_path).resolve()
        if not raw_root.is_relative_to(self.root) or not path.is_relative_to(raw_root):
            raise SourceCatalogError("Report path escapes the raw corpus")
        return path

    def _resolve(self, citation):
        if not isinstance(citation, dict):
            raise SourceCatalogError("Citation must be an object")
        metadata = citation.get("citation", citation)
        if not isinstance(metadata, dict):
            raise SourceCatalogError("Citation metadata must be an object")
        identifier = metadata.get("document_id")
        if not isinstance(identifier, str) or identifier not in self._reports:
            raise SourceCatalogError("Citation document is not in the validated manifest")
        report = self._reports[identifier]
        expected = {"document_id": report.document_id, "source_file": report.source_file,
                    "sha256": report.sha256, "ticker": report.ticker, "fiscal_year": report.fiscal_year}
        if (type(metadata.get("fiscal_year")) is not int
                or any(metadata.get(key) != value for key, value in expected.items())):
            raise SourceCatalogError("Citation identity does not match the manifest")
        page = metadata.get("pdf_page")
        if not _positive_integer(page) or page > report.page_count:
            raise SourceCatalogError("Citation PDF page is outside the report")
        return report, page

    def describe(self, citation):
        report, page = self._resolve(citation)
        title = f"{report.company} ({report.ticker}) FY{report.fiscal_year} annual report"
        return {"document_id": report.document_id, "source_file": report.source_file,
                "sha256": report.sha256, "company": report.company, "ticker": report.ticker,
                "fiscal_year": report.fiscal_year, "pdf_page": page, "page_count": report.page_count,
                "title": title, "source_label": f"{title} — PDF page {page} of {report.page_count}",
                "public_url": f"{report.public_url}#page={page}" if report.public_url else None,
                "local_available": self._path(report).is_file()}

    def pdf_bytes(self, citation):
        report, _ = self._resolve(citation)
        data = self._path(report).read_bytes()
        if hashlib.sha256(data).hexdigest() != report.sha256:
            raise SourceCatalogError("Local PDF hash does not match the manifest")
        import pymupdf
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                if not document.is_pdf or document.needs_pass or document.page_count != report.page_count:
                    raise SourceCatalogError("Local PDF properties do not match the manifest")
        except SourceCatalogError:
            raise
        except (RuntimeError, ValueError) as error:
            raise SourceCatalogError("Local source is not a readable PDF") from error
        return data

    def page_png(self, citation, *, max_width=1200, highlight=None, report_highlight=False):
        """Render a bounded RGB preview from the same hash-verified PDF bytes.

        ``highlight`` is an optional [x0, y0, x1, y1] box in PDF points, such as
        an answer citation's bbox. It is drawn only on the in-memory copy, and
        only when it is a valid box on an unrotated page; otherwise it is ignored.
        With ``report_highlight`` the result is ``(png, drawn)``.
        """
        if type(max_width) is not int or not 256 <= max_width <= 2400:
            raise SourceCatalogError("Preview width must be between 256 and 2400 pixels")
        _, page = self._resolve(citation)
        data = self.pdf_bytes(citation)
        import pymupdf
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                source_page = document[page - 1]
                width, height = source_page.rect.width, source_page.rect.height
                if not all(math.isfinite(x) and x > 0 for x in (width, height)):
                    raise SourceCatalogError("Invalid PDF page dimensions")
                scale = min(max_width / width, 2400 / height, 2.0)
                box = _highlight_box(highlight, source_page)
                if box is not None:
                    source_page.draw_rect(box, color=(0.85, 0.45, 0.0), fill=(1.0, 0.85, 0.2), width=1.5,
                                          fill_opacity=0.25, overlay=True)
                png = source_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                            colorspace=pymupdf.csRGB, alpha=False).tobytes("png")
                return (png, box is not None) if report_highlight else png
        except SourceCatalogError:
            raise
        except (RuntimeError, ValueError) as error:
            raise SourceCatalogError("Unable to render source PDF page") from error


def _highlight_box(highlight, page):
    """A padded pymupdf.Rect for a finite, ordered box that meets an unrotated page, else None."""
    import pymupdf
    if highlight is None or page.rotation != 0:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in highlight)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)) or x0 >= x1 or y0 >= y1:
        return None
    box = pymupdf.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2)
    return box if box.intersects(page.rect) else None
