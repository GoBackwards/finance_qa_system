"""CLI preconditions must refuse without starting expensive or external work."""
import io
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import answer_reliably as cli


class ReliableCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps([{"company": "NVIDIA", "ticker": "NVDA", "fiscal_year": 2025}]), encoding="utf-8")

    def invoke(self, arguments):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli.main([*arguments, "--manifest", str(self.manifest)])
        return code, json.loads(stdout.getvalue())

    def test_help_documents_session_and_startup(self):
        output = io.StringIO()
        with patch.object(cli, "_build_pipeline") as build, redirect_stdout(output), self.assertRaises(SystemExit) as result:
            cli.main(["--help"])
        self.assertEqual(result.exception.code, 0)
        help_text = " ".join(output.getvalue().split())
        self.assertIn("outside per-request deadlines", help_text)
        self.assertIn("process only", help_text)
        build.assert_not_called()

    def test_default_threshold_refuses_without_provider_or_models(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "_build_pipeline") as build:
            code, report = self.invoke(["NVIDIA revenue"])
        self.assertEqual(code, 2)
        self.assertEqual(report["results"][0]["reason"], "unconfigured_threshold")
        self.assertEqual(report["startup_seconds"], 0)
        build.assert_not_called()

    def test_missing_provider_refuses_before_models(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "_build_pipeline") as build:
            code, report = self.invoke(["NVIDIA revenue", "--threshold", "6"])
        self.assertEqual(code, 2)
        self.assertEqual(report["results"][0]["reason"], "model_not_configured")
        self.assertEqual(report["results"][0]["answer"], "I'm not sure")
        build.assert_not_called()

    def test_absent_company_and_report_refuse_before_configuration(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "_build_pipeline") as build:
            code, report = self.invoke(["Apple FY2024 gross margin", "NVIDIA FY2027 annual report"])
        self.assertEqual(code, 2)
        self.assertEqual([row["reason"] for row in report["results"]], ["company_not_in_corpus", "report_not_in_corpus"])
        build.assert_not_called()

    def test_empty_query_refuses_without_models(self):
        with patch.object(cli, "_build_pipeline") as build:
            code, report = self.invoke([""])
        self.assertEqual(code, 2)
        self.assertEqual(report["results"][0]["reason"], "invalid_query")
        build.assert_not_called()

    def test_oversized_query_refuses_before_provider_or_model_startup(self):
        with patch.object(cli.ModelConfig, "from_env") as provider, patch.object(cli, "_build_pipeline") as build:
            code, report = self.invoke(["NVIDIA " + "x" * 9994, "--threshold", "6"])
        self.assertEqual(code, 2)
        self.assertEqual(report["results"][0]["reason"], "invalid_query")
        provider.assert_not_called()
        build.assert_not_called()
        _, boundary = self.invoke(["NVIDIA " + "x" * 9993])
        self.assertEqual(boundary["results"][0]["reason"], "unconfigured_threshold")

    def test_invalid_question_does_not_discard_other_session_questions(self):
        pipeline = Mock(cache=SimpleNamespace(stats={"entries": 0, "hits": 0}))
        pipeline.answer.return_value = {"status": "answer", "answer": "Verified fixture", "audit": {}}
        engine = Mock()
        with patch.dict(os.environ, {"FINANCE_QA_BASE_URL": "http://localhost:1234/v1", "FINANCE_QA_MODEL": "fixture"}, clear=True), patch.object(cli, "_build_pipeline", return_value=(pipeline, engine)):
            code, report = self.invoke(["", "NVIDIA revenue", "x" * 10001, "--threshold", "6"])
        self.assertEqual(code, 2)
        self.assertEqual([row["status"] for row in report["results"]], ["abstain", "answer", "abstain"])
        pipeline.answer.assert_called_once_with("NVIDIA revenue")

    def test_repeated_questions_share_one_pipeline(self):
        pipeline = Mock(cache=SimpleNamespace(stats={"entries": 1, "hits": 1}))
        pipeline.answer.return_value = {"status": "answer", "answer": "Verified fixture", "audit": {}}
        engine = Mock()
        with patch.dict(os.environ, {"FINANCE_QA_BASE_URL": "http://localhost:1234/v1", "FINANCE_QA_MODEL": "fixture"}, clear=True), patch.object(cli, "_build_pipeline", return_value=(pipeline, engine)) as build:
            code, report = self.invoke(["NVIDIA revenue", "NVIDIA revenue", "--threshold", "6"])
        self.assertEqual(code, 0)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(pipeline.answer.call_count, 2)
        engine.close.assert_called_once()
        self.assertEqual(report["cache_stats"]["hits"], 1)

    def test_expansion_requires_separate_threshold_and_invalid_options_reject(self):
        for extra in (["--enable-expansion"], ["--threshold", "nan"], ["--deadline-seconds", "0"], ["--device", "cuda"]):
            with self.subTest(extra=extra), patch.object(cli, "_build_pipeline") as build, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                cli.main(["NVIDIA revenue", *extra])
            self.assertEqual(result.exception.code, 2)
            build.assert_not_called()

    def test_output_writes_new_json_and_preserves_existing_file(self):
        output = self.root / "answer.json"
        with redirect_stdout(io.StringIO()), patch.object(cli, "_build_pipeline") as build:
            self.assertEqual(cli.main(["NVIDIA revenue", "--manifest", str(self.manifest), "--output", str(output)]), 2)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["results"][0]["answer"], "I'm not sure")
        before = output.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["NVIDIA revenue", "--output", str(output)])
        self.assertEqual(output.read_bytes(), before)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
