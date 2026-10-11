"""The demo only exposes PDFs bound to validated report citations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from report_search.demo_sources import SourceCatalog, SourceCatalogError


class SourceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "data/raw/TEST/TEST_FY2025_annual_report.pdf"
        self.raw.parent.mkdir(parents=True)
        with pymupdf.open() as doc:
            for text in ("First report page", "Second report page"):
                doc.new_page().insert_text((72, 72), text)
            self.data = doc.tobytes()
        self.raw.write_bytes(self.data)
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.row = {"company": "Test Company", "ticker": "TEST", "fiscal_year": 2025,
                    "relative_path": "data/raw/TEST/TEST_FY2025_annual_report.pdf",
                    "status": "validated", "sha256": self.sha, "page_count": 2,
                    "resolved_url": "https://reports.example/report.pdf?download=1#old"}
        self.citation = {"document_id": "TEST_FY2025_annual_report-" + self.sha[:12],
                         "source_file": self.raw.name, "sha256": self.sha,
                         "ticker": "TEST", "fiscal_year": 2025, "pdf_page": 2}
        self.persist()

    def persist(self, rows=None):
        (self.root / "data/manifest.json").write_text(json.dumps(rows or [self.row]), encoding="utf-8")

    def test_verified_download_and_requested_page_preview(self):
        catalog = SourceCatalog(self.root)
        description = catalog.describe({"citation": self.citation, "quote": "Second report page"})
        self.assertEqual(description["public_url"], "https://reports.example/report.pdf?download=1#page=2")
        self.assertIn("Test Company (TEST) FY2025", description["title"])
        self.assertIn("PDF page 2 of 2", description["source_label"])
        self.assertTrue(description["local_available"])
        self.assertEqual(catalog.pdf_bytes(self.citation), self.data)
        image = catalog.page_png(self.citation, max_width=600)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", image[16:24])
        self.assertLessEqual(width, 600)
        self.assertLessEqual(height, 2400)
        with pymupdf.open(stream=self.data, filetype="pdf") as doc:
            page = doc[1]
            scale = min(600 / page.rect.width, 2400 / page.rect.height, 2.0)
            expected = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                       colorspace=pymupdf.csRGB, alpha=False).tobytes("png")
        self.assertEqual(image, expected)

    def test_missing_local_report_retains_manifest_link(self):
        self.raw.unlink()
        catalog = SourceCatalog(self.root)
        self.assertFalse(catalog.describe(self.citation)["local_available"])
        self.assertTrue(catalog.describe(self.citation)["public_url"].endswith("#page=2"))
        with self.assertRaises(FileNotFoundError):
            catalog.pdf_bytes(self.citation)

    def test_citation_identity_tampering_rejected(self):
        catalog = SourceCatalog(self.root)
        for key, wrong in {"document_id": "other", "sha256": "f" * 64, "ticker": "OTHER",
                           "fiscal_year": 2024, "source_file": "../../secret.pdf"}.items():
            with self.subTest(field=key), self.assertRaises(SourceCatalogError):
                catalog.describe(self.citation | {key: wrong})

    def test_noninteger_and_out_of_range_pages_rejected(self):
        catalog = SourceCatalog(self.root)
        for page in (0, -1, 3, True, 1.0, "1", None):
            with self.subTest(page=page), self.assertRaises(SourceCatalogError):
                catalog.page_png(self.citation | {"pdf_page": page})

    def test_unvalidated_or_duplicate_manifest_identity_rejected(self):
        self.persist([self.row | {"status": "failed"}])
        with self.assertRaises(SourceCatalogError):
            SourceCatalog(self.root).describe(self.citation)
        self.persist([self.row, deepcopy(self.row)])
        with self.assertRaises(SourceCatalogError):
            SourceCatalog(self.root)

    def test_manifest_paths_must_be_canonical_and_confined(self):
        for path in ("../secret.pdf", "data/raw/TEST/../../../secret.pdf", str(self.raw),
                     "data/raw/OTHER/TEST_FY2025_annual_report.pdf", "data\\raw\\TEST\\secret.pdf"):
            with self.subTest(path=path), self.assertRaises(SourceCatalogError):
                self.persist([self.row | {"relative_path": path}])
                SourceCatalog(self.root)

    def test_symlink_escape_rejected(self):
        outside = self.root / "outside.pdf"
        outside.write_bytes(self.data)
        self.raw.unlink()
        try:
            self.raw.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Creating symlinks is unavailable for this user")
        with self.assertRaises(SourceCatalogError):
            SourceCatalog(self.root)

    def test_path_resolution_escape_rejected_without_symlink_privileges(self):
        original_resolve = Path.resolve
        outside = self.root / "outside.pdf"

        def resolved(path, *args, **kwargs):
            if path == self.raw:
                return outside
            return original_resolve(path, *args, **kwargs)

        # Models a junction/symlink resolution when Windows cannot create one.
        with patch.object(Path, "resolve", resolved), self.assertRaisesRegex(SourceCatalogError, "escapes"):
            SourceCatalog(self.root)

    def test_modified_pdf_rechecked_on_every_download(self):
        catalog = SourceCatalog(self.root)
        self.assertEqual(catalog.pdf_bytes(self.citation), self.data)
        self.raw.write_bytes(self.data.replace(b"First", b"False"))
        # Compression may hide the text. A trailing byte must still change SHA.
        self.raw.write_bytes(self.raw.read_bytes() + b"\n")
        with self.assertRaisesRegex(SourceCatalogError, "hash"):
            catalog.pdf_bytes(self.citation)
        with self.assertRaisesRegex(SourceCatalogError, "hash"):
            catalog.page_png(self.citation)

    def test_actual_page_count_must_match_manifest(self):
        self.persist([self.row | {"page_count": 3}])
        with self.assertRaisesRegex(SourceCatalogError, "properties"):
            SourceCatalog(self.root).pdf_bytes(self.citation)

    def test_matching_hash_does_not_make_non_pdf_content_valid(self):
        data = b"This is not a PDF document."
        sha = hashlib.sha256(data).hexdigest()
        self.raw.write_bytes(data)
        self.persist([self.row | {"sha256": sha}])
        citation = self.citation | {"sha256": sha, "document_id": "TEST_FY2025_annual_report-" + sha[:12]}
        with self.assertRaisesRegex(SourceCatalogError, "readable PDF"):
            SourceCatalog(self.root).pdf_bytes(citation)

    def test_unsafe_public_urls_never_exposed_and_https_preferred(self):
        for url in ("javascript:alert(1)", "file:///etc/passwd", "https://user:pass@reports.example/x",
                    "https://reports.example\\@evil.example/x", "https://reports.example/x\n", "https://reports.example:bad/x"):
            with self.subTest(url=url):
                self.persist([self.row | {"resolved_url": url}])
                self.assertIsNone(SourceCatalog(self.root).describe(self.citation)["public_url"])
        self.persist([self.row | {"resolved_url": "http://reports.example/old.pdf",
                                 "download_url": "https://reports.example/secure.pdf"}])
        self.assertEqual(SourceCatalog(self.root).describe(self.citation)["public_url"],
                         "https://reports.example/secure.pdf#page=2")

    def test_preview_highlights_only_a_valid_box_on_the_page(self):
        catalog = SourceCatalog(self.root)
        plain = catalog.page_png(self.citation)
        self.assertNotEqual(catalog.page_png(self.citation, highlight=[60, 50, 300, 90]), plain)
        for invalid in ([float("nan"), 0, 10, 10], [300, 50, 60, 90], [5000, 5000, 6000, 6000],
                        ["a", 1, 2, 3], [1, 2, 3], "not a box"):
            with self.subTest(box=invalid):
                self.assertEqual(catalog.page_png(self.citation, highlight=invalid), plain)
        self.assertEqual(hashlib.sha256(self.raw.read_bytes()).hexdigest(), self.sha)  # drawn in memory only
        self.assertTrue(catalog.page_png(self.citation, highlight=[60, 50, 300, 90], report_highlight=True)[1])
        self.assertEqual(catalog.page_png(self.citation, highlight=[300, 50, 60, 90], report_highlight=True),
                         (plain, False))

    def test_preview_size_is_bounded(self):
        catalog = SourceCatalog(self.root)
        for width in (0, 255, 2401, True, 500.0):
            with self.subTest(width=width), self.assertRaises(SourceCatalogError):
                catalog.page_png(self.citation, max_width=width)


if __name__ == "__main__":
    unittest.main()
