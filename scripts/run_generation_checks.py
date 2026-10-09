"""Exercise real sources and HTTP JSON integration with controlled model fixtures.

This is an offline guardrail regression, not a generated-answer quality test.
The loopback server returns prescribed drafts/verdicts; it is not an LLM.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from collections import deque
import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from report_index.faiss_index import sha256_file
from report_search.answer_cli import load_cached_context
from report_search.evidence_binding import SourceEvidenceStore
from report_search.experiment import write_json
from report_search.generation import generate_answer
from report_search.llm_client import ChatCompletionsClient, ModelConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranker-run", type=Path, default=Path("data/experiments/reranker"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/generation_checks"))
    args = parser.parse_args(argv)
    retrieval_query, candidates, _ = load_cached_context(args.reranker_run, "test-001")
    store = SourceEvidenceStore(args.processed)
    bundle = store.build(candidates)
    entry = next(value for value in bundle["evidence"].values() if value["kind"] == "prose" and 40 < len(value["text"]) < 1500)
    quote = entry["text"]
    query = "Quote the source paragraph beginning with: " + quote[:80]
    draft = {"status": "answer", "claims": [{"text": "The retrieved report states: " + quote,
             "citations": [{"evidence_id": entry["evidence_id"], "quote": quote}]}]}
    approved = {"answers_question": True, "claim_verdicts": [{"claim_index": 0, "supported": True, "reason": "Controlled fixture approval."}]}
    denied = copy.deepcopy(approved)
    denied["claim_verdicts"][0]["supported"] = False
    bad_claim = copy.deepcopy(draft)
    bad_claim["claims"][0]["text"] = "The report guarantees revenue of $999 trillion next year."
    bad_id = copy.deepcopy(draft)
    bad_id["claims"][0]["citations"][0]["evidence_id"] = "E999999"
    bad_quote = copy.deepcopy(draft)
    bad_quote["claims"][0]["citations"][0]["quote"] = "Invented quote that is absent from the source."
    incomplete = copy.deepcopy(approved)
    incomplete["answers_question"] = False
    responses, requests = deque(), []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            response = responses.popleft() if responses else {"http_error": 500}
            if "http_error" in response:
                self.send_response(response["http_error"])
                self.end_headers()
                return
            finish = response.pop("_finish_reason", "stop")
            envelope = {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(response)}}]}
            data = json.dumps(envelope).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ChatCompletionsClient(ModelConfig(f"http://127.0.0.1:{server.server_port}/v1", "controlled-fixture"))
    cases = [
        ("cited_answer_and_full_source_trace", candidates, -100.0, [draft, approved], "answer", 2),
        ("true_quote_but_false_claim", candidates, -100.0, [bad_claim, denied], "abstain", 2),
        ("invented_evidence_id", candidates, -100.0, [bad_id], "abstain", 1),
        ("altered_quote", candidates, -100.0, [bad_quote], "abstain", 1),
        ("question_not_fully_answered", candidates, -100.0, [draft, incomplete], "abstain", 2),
        ("low_score_no_provider_call", candidates, 100.0, [], "abstain", 0),
        ("unconfigured_gate_no_provider_call", candidates, None, [], "abstain", 0),
        ("empty_retrieval_no_provider_call", [], -100.0, [], "abstain", 0),
        ("provider_error", candidates, -100.0, [{"http_error": 503}], "abstain", 1),
        ("malformed_semantic_verdict", candidates, -100.0, [draft, {"answers_question": True, "claim_verdicts": []}], "abstain", 2),
        ("truncated_model_response", candidates, -100.0, [{**draft, "_finish_reason": "length"}], "abstain", 1),
    ]
    results = []
    try:
        for name, selected, threshold, outputs, expected_status, expected_calls in cases:
            responses.clear()
            responses.extend(copy.deepcopy(outputs))
            count_before = len(requests)
            result = generate_answer(query, selected, threshold=threshold, source_store=store,
                                     generator=client, verifier=client)
            calls = len(requests) - count_before
            assert result["status"] == expected_status, (name, result["reason"])
            assert calls == expected_calls, (name, calls)
            if expected_status == "abstain":
                assert result["answer"] == "I'm not sure" and result["citations"] == []
                assert "$999 trillion" not in json.dumps(result)
            else:
                assert result["citations"][0]["source_text"] == quote
                assert result["citations"][0]["citation"]["pdf_page"] == entry["citation"]["pdf_page"]
            results.append({"case": name, "passed": True, "provider_calls": calls,
                            "result": result})
        assert all("ground_truth" not in json.dumps(request) and "answer_points" not in json.dumps(request) for request in requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    summary = {"schema_version": 1, "mode": "offline_controlled_http_fixtures",
               "live_llm_evaluated": False, "answer_accuracy_measured": False,
               "source_question_id": "test-001", "case_count": len(results),
               "retrieval_query": retrieval_query, "fixture_query": query,
               "passed_count": sum(row["passed"] for row in results),
               "source_candidate_sha256": sha256_file(args.reranker_run / "candidates.jsonl"),
               "limitation": "Prescribed drafts/verdicts test control flow and exact references, not semantic verifier quality or real-world hallucination rate.",
               "cases": results}
    write_json(args.output / "results.json", summary)
    print(json.dumps({key: summary[key] for key in ("mode", "case_count", "passed_count", "live_llm_evaluated")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
