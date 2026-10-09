"""Replay classic financial QA faults with real sources and controlled model output.

The retrieval probe is a separate GPU experiment. This CPU audit does not send
provider requests and does not measure a real semantic reviewer's accuracy.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import copy
import json
import sqlite3
import subprocess
import time
import types

from report_index.faiss_index import sha256_file
from report_search.answer_verification import validate_draft
from report_search.evidence_binding import SourceEvidenceStore
from report_search.experiment import write_json
from report_search.generation import generate_answer


BASELINE_COMMIT = "ba6fc6e99e55ff69c96542e6f44c84ac55f10322"


class FixtureModel:
    identity = {"backend": "prescribed_response_fixture", "live_model": False}

    def __init__(self, response, clock=None, delay=0):
        self.response, self.clock, self.delay = response, clock, delay
        self.calls = 0

    def complete_json(self, messages):
        self.calls += 1
        if self.clock is not None:
            self.clock.value += self.delay
        return copy.deepcopy(self.response)


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def find_candidate(connection, anchor, *, row_index=None, needle=None):
    """Get full existing source payloads; supplied scores only open fixture gates."""
    for payload, body in connection.execute("SELECT payload,body FROM chunks WHERE document_id=?", (anchor["document_id"],)):
        chunk = json.loads(payload)
        chunk["text"] = body
        for atom in chunk["evidence"]:
            same = (atom.get("table_id") == anchor["table_id"] and atom.get("row_index") == row_index
                    if row_index is not None else atom.get("element_id") == anchor.get("element_id"))
            text = atom.get("text", atom.get("row_text", ""))
            if same and (needle is None or needle in text):
                chunk.update(rank=1, score=10.0, reranker_score=10.0, retrieval_approach="vector_reranker")
                return chunk
    raise ValueError("Missing verified diagnostic source candidate")


def entry_for(bundle, anchor, *, row_index=None, needle=None):
    for entry in bundle["evidence"].values():
        locator = entry["source_locator"]
        same = (locator.get("table_id") == anchor["table_id"] and locator.get("row_index") == row_index
                if row_index is not None else locator.get("element_id") == anchor.get("element_id"))
        if same and (needle is None or needle in entry["text"]):
            return entry
    raise ValueError("Missing registered diagnostic evidence")


def draft_for(entry, claim):
    return {"status": "answer", "claims": [{"text": claim,
            "citations": [{"evidence_id": entry["evidence_id"], "quote": entry["text"]}]}]}


def verdict(supported=True, answers_question=True):
    return {"answers_question": answers_question,
            "claim_verdicts": [{"claim_index": 0, "supported": supported,
                                "reason": "Prescribed source-grounded regression verdict, not a live model judgment."}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/classic_bad_cases"))
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)
    sources_path = Path("data/evaluation/classic_bad_case_sources.json")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    findings = sources["findings"]
    retrieval_path = Path("data/evaluation/bad_case_retrieval_results.json")
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    provider_path = Path("data/evaluation/provider_fault_results.json")
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    connection = sqlite3.connect(Path("data/indexes/bm25_s512_o50/retrieval.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    store = SourceEvidenceStore(args.processed)
    controls = []

    def check(name, query, candidate, draft, review, *, expected_reason, reference_expected=True):
        generator, verifier = FixtureModel(draft), FixtureModel(review)
        result = generate_answer(query, [candidate], threshold=0, source_store=store,
                                 generator=generator, verifier=verifier)
        assert result["status"] == "abstain" and result["answer"] == "I'm not sure", name
        assert result["reason"] == expected_reason, (name, result["reason"])
        assert result["reference_verified"] is reference_expected, name
        assert result["citations"] == [], name
        controls.append({"name": name, "passed": True, "query": query, "bad_claim": draft["claims"][0]["text"],
                         "expected_reason": expected_reason, "reference_verified": result["reference_verified"],
                         "generator_calls": generator.calls, "verifier_calls": verifier.calls,
                         "evidence_status": "controlled_response_with_real_source", "result": result})

    cloud = findings["consolidated_vs_cloud"]["cloud_passage"]
    cloud_chunk = find_candidate(connection, cloud, needle="71%")
    cloud_entry = entry_for(store.build([cloud_chunk]), cloud, needle="71%")
    check("company_scope_substitution", findings["consolidated_vs_cloud"]["query"], cloud_chunk,
          draft_for(cloud_entry, "Microsoft's company-wide gross margin was 71% in FY2024."), verdict(False),
          expected_reason="unsupported_claim")

    historic = findings["unsupported_forecast"]["historical_revenue_anchor"]
    historic_chunk = find_candidate(connection, historic, row_index=2)
    historic_entry = entry_for(store.build([historic_chunk]), historic, row_index=2)
    check("historical_revenue_as_forecast", findings["unsupported_forecast"]["query"], historic_chunk,
          draft_for(historic_entry, "NVIDIA forecasts FY2027 revenue of $500 billion."), verdict(False),
          expected_reason="unsupported_claim")

    income = findings["wrong_year_column"]["anchor"]
    income_chunk = find_candidate(connection, income, row_index=13)
    income_entry = entry_for(store.build([income_chunk]), income, row_index=13)
    for name, claim in (("wrong_fiscal_column", "Microsoft FY2024 operating income was $88,523 million."),
                        ("wrong_unit_scale", "Microsoft FY2024 operating income was $109,433 billion.")):
        check(name, findings["wrong_year_column"]["query"], income_chunk,
              draft_for(income_entry, claim), verdict(False), expected_reason="unsupported_claim")

    amd = findings["incomplete_comparison"]["amd_anchor"]
    amd_chunk = find_candidate(connection, amd, row_index=4)
    amd_entry = entry_for(store.build([amd_chunk]), amd, row_index=4)
    check("comparison_missing_intel", findings["incomplete_comparison"]["query"], amd_chunk,
          draft_for(amd_entry, "AMD's FY2024 gross margin was 49%."), verdict(True, False),
          expected_reason="question_not_answered")

    good_draft = draft_for(income_entry, "Microsoft FY2024 operating income was $109,433 million.")
    for name, field, value, reason in (("invented_citation", "evidence_id", "E999999", "unknown_evidence_id"),
                                       ("altered_quote", "quote", "Operating income 999999", "quote_not_in_source")):
        bad = copy.deepcopy(good_draft)
        bad["claims"][0]["citations"][0][field] = value
        check(name, findings["wrong_year_column"]["query"], income_chunk, bad, verdict(),
              expected_reason=reason, reference_expected=False)

    broken = findings["numeric_parse_corruption_contained"]
    chunk_id = broken["indexed_fallback_chunks"][0]["chunk_id"]
    payload, body = connection.execute("SELECT payload,body FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
    fallback = json.loads(payload)
    fallback.update(text=body, rank=1, score=10.0, reranker_score=10.0)
    preserved = store.build([fallback])
    assert preserved["chunks"][0]["structured_reliable"] is False
    damaged = copy.deepcopy(fallback)
    damaged["rows"] = broken["candidate_rows_debug_only"]
    generator, verifier = FixtureModel(good_draft), FixtureModel(verdict())
    result = generate_answer("What was Intel FY2024 revenue?", [damaged], threshold=0,
                             source_store=store, generator=generator, verifier=verifier)
    assert result["reason"] == "source_verification_failed" and generator.calls == verifier.calls == 0
    controls.append({"name": "rejected_table_cells_reintroduced", "passed": True,
                     "accepted_structured_rows": len(broken["accepted_rows"]), "source_fallback_preserved": True,
                     "generator_calls": generator.calls, "result": result})

    # A direct reported-value answer is a positive control for reference rendering.
    good = generate_answer(findings["wrong_year_column"]["query"], [income_chunk], threshold=0, source_store=store,
                           generator=FixtureModel(good_draft), verifier=FixtureModel(verdict()))
    assert good["status"] == "answer" and good["citations"][0]["citation"]["pdf_page"] == 48
    controls.append({"name": "supported_reported_value", "passed": True, "result": good})

    future = findings["unsupported_forecast"]["supported_future_control"]
    future_chunk = find_candidate(connection, future)
    future_entry = entry_for(store.build([future_chunk]), future)
    future_result = generate_answer("What does NVIDIA's FY2025 report say about FY2026 capital expenditures?",
        [future_chunk], threshold=0, source_store=store,
        generator=FixtureModel(draft_for(future_entry, "NVIDIA expects higher capital expenditures in FY2026.")),
        verifier=FixtureModel(verdict()))
    assert future_result["status"] == "answer"
    controls.append({"name": "supported_future_statement", "passed": True, "result": future_result})

    # Compare old/new control flow under the same 4-second simulated generator.
    baseline_source = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:src/report_search/generation.py"])
    baseline = types.ModuleType("baseline_generation_fault_audit")
    exec(compile(baseline_source, "baseline_generation.py", "exec"), baseline.__dict__)
    clock = Clock()
    before_verifier = FixtureModel(verdict())
    before = baseline.generate_answer(findings["wrong_year_column"]["query"], [income_chunk], threshold=0,
        source_store=store, generator=FixtureModel(good_draft, clock, 4), verifier=before_verifier)
    clock = Clock()
    after_verifier = FixtureModel(verdict())
    after = generate_answer(findings["wrong_year_column"]["query"], [income_chunk], threshold=0,
        source_store=store, generator=FixtureModel(good_draft, clock, 4), verifier=after_verifier,
        deadline_seconds=3, clock=clock)
    assert before["status"] == "answer" and before_verifier.calls == 1
    assert after["reason"] == "deadline_exceeded" and after_verifier.calls == 0
    controls.append({"name": "late_generator_with_three_second_example_budget", "passed": True,
        "baseline_commit": BASELINE_COMMIT, "before": {"status": before["status"], "verifier_calls": before_verifier.calls},
        "after": after, "clock": "simulated 4-second active generation; not a production latency measurement"})

    cold_store = SourceEvidenceStore(args.processed)
    start = time.perf_counter()
    cold_store.build([income_chunk])
    cold_ms = (time.perf_counter() - start) * 1000
    warm = []
    for _ in range(5):
        start = time.perf_counter()
        cold_store.build([income_chunk])
        warm.append((time.perf_counter() - start) * 1000)
    connection.close()
    case_types = ["retrieval_expansion_regression", "metric_scope_confusion", "unsupported_forecast",
                  "fiscal_column_and_unit_confusion", "incomplete_comparison", "table_parse_corruption",
                  "fabricated_citation", "partial_provider_response", "latency_budget", "incomplete_gold_labels"]
    case_controls = [[], [0], [1, 9], [2, 3], [4], [7], [5, 6], [], [10], []]
    report = {"schema_version": 1, "case_count": 10, "live_generation_evaluated": False,
        "case_types": case_types, "cases": [{"case_id": f"B{n+1:02}", "failure_type": kind,
            "controlled_checks": [controls[i]["name"] for i in indices]} for n, (kind, indices) in enumerate(zip(case_types, case_controls))],
        "controlled_check_count": len(controls), "controlled_checks_passed": sum(c["passed"] for c in controls),
        "controls": controls, "retrieval_probe": {"artifact": str(retrieval_path), "sha256": sha256_file(retrieval_path),
            "case_count": retrieval["case_count"], "generation_calls": retrieval["generation_calls"]},
        "source_audit": {"artifact": str(sources_path), "sha256": sha256_file(sources_path)},
        "provider_faults": {"artifact": str(provider_path), "sha256": sha256_file(provider_path), "results": provider["results"]},
        "source_binding_latency": {"case": "MSFT FY2024 p48 single statement chunk", "cold_ms": cold_ms,
            "warm_ms": warm, "scope": "source reconstruction/integrity checking only; same store reused; no model work"},
        "implementation_sha256": {name: sha256_file(name) for name in
            ["src/report_search/generation.py", "src/report_search/llm_client.py", "src/report_search/answer_cli.py",
             "src/report_search/evidence_binding.py", "src/report_search/answer_verification.py", "scripts/run_bad_case_audit.py"]},
        "limitations": ["Source anchors and alternative support are agent-reviewed, not independently human labeled.",
            "Negative semantic verdicts are prescribed fixtures; passing them does not measure a real model's detection rate.",
            "The three-second budget is illustrative and cooperative; active operations are not interrupted.",
            "No production end-to-end latency, generated-answer accuracy or hallucination reduction was measured.",
            "Frozen benchmark labels and default retrieval/gate policies are unchanged."]}
    write_json(args.output / "results.json", report)
    if args.publish:
        write_json(args.publish, report)
    print(json.dumps({key: report[key] for key in ("case_count", "controlled_check_count", "controlled_checks_passed", "live_generation_evaluated")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
