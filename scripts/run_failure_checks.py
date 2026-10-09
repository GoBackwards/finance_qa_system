"""Record deterministic failure-handling regression results; no live LLM calls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
import hashlib
import json
import time
import unittest

from report_search.experiment import write_json


PATTERNS = ("test_query_policy.py", "test_runtime_support.py", "test_reliable_pipeline.py",
            "test_reliable_cli.py",
            "test_generation.py", "test_answer_verification.py", "test_evidence_binding.py", "test_llm_client.py")


class RecordedResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.records = []

    def startTest(self, test):
        super().startTest(test)
        self.started = time.perf_counter()
        self.prior_errors = len(self.errors) + len(self.failures)

    def stopTest(self, test):
        self.records.append({"test": test.id(), "passed": len(self.errors) + len(self.failures) == self.prior_errors,
                             "duration_ms": (time.perf_counter() - self.started) * 1000})
        super().stopTest(test)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/failure_handling_checks/results.json"))
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args(argv)
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern)
                               for pattern in PATTERNS)
    result = RecordedResult()
    suite.run(result)
    files = [ROOT / "src/report_search" / name for name in
             ("query_policy.py", "runtime_support.py", "reliable_pipeline.py", "generation.py",
              "answer_verification.py", "evidence_binding.py", "llm_client.py")]
    files += [ROOT / "tests" / pattern for pattern in PATTERNS]
    files.append(ROOT / "scripts/answer_reliably.py")
    files.append(Path(__file__).resolve())
    artifact = {"schema_version": 1, "mode": "controlled_failure_regression", "live_llm_evaluated": False,
                "production_latency_measured": False, "tests_run": result.testsRun,
                "passed": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
                "cases": result.records,
                "implementation_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "limitation": "Controlled inputs/model verdicts check behavior, not hallucination rates or semantic verifier reliability. Case durations are test execution times, not serving latency."}
    write_json(args.output, artifact)
    if args.publish:
        write_json(args.publish, artifact)
    print(json.dumps({key: artifact[key] for key in ("mode", "tests_run", "passed", "failures", "errors")}))
    if not result.wasSuccessful():
        for test, details in result.failures + result.errors:
            print(test.id(), details, file=sys.stderr)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
