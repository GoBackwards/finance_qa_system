"""The parse wrapper never writes into corpus v1, which this parser cannot reproduce."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parse_reports_script", ROOT / "scripts/parse_reports.py")
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


class ParseWrapperGuardTests(unittest.TestCase):
    def test_v1_folder_and_missing_output_are_refused(self):
        for argv in ([], ["--workers", "4"], ["--output", "data/processed"], ["--output=data/processed/"],
                     ["--output", str(ROOT / "data/processed")], ["--tickers", "NVDA", "--output", "./data/processed"]):
            with self.subTest(argv=argv):
                self.assertIsNotNone(script.refusal(argv))

    def test_other_outputs_and_help_are_allowed(self):
        for argv in (["--output", "data/processed_v2"], ["--output=tmp/parsed_pages", "--pages", "1-2"], ["--help"]):
            with self.subTest(argv=argv):
                self.assertIsNone(script.refusal(argv))


if __name__ == "__main__":
    unittest.main()
