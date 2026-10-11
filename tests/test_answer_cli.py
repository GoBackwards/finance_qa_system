"""Offline replay integrity and public CLI refusal behavior; no model calls."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from report_search.answer_cli import load_cached_context, main
from report_search.confidence import calibrate_threshold, make_gate_policy


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedRun:
    """Two questions and actual twenty-candidate replay schema in a temp folder."""

    def __init__(self, root):
        self.root = root
        self.candidates, self.scores = [], []
        for identifier in ("q1", "q2"):
            pool = [{"chunk_id": f"{identifier}-c{rank}", "rank": rank, "score": 1 - rank / 100,
                     "strategy_id": "table_preserving_s512_o50", "document_id": "demo-report",
                     "text": f"Source paragraph {rank}.", "evidence": [], "citations": []}
                    for rank in range(1, 21)]
            self.candidates.append({"question_id": identifier, "query": f"Question {identifier}?", "candidates": pool})
            self.scores.append({"question_id": identifier, "candidate_ids": [c["chunk_id"] for c in pool],
                                "scores": list(range(1, 21)), "rerank_ms": 1.0})
        self.candidate_manifest = {
            "candidate_depth": 20, "question_count": 2, "pair_count": 40, "question_ids": ["q1", "q2"],
            "corpus": {"chunks_sha256": "a" * 64},
            "vector_query_manifest": {"contract": {"corpus_encoder": {"model_revision": "encoder-revision"},
                                                     "corpus_vectors_file_sha256": "b" * 64,
                                                     "corpus_embedding_manifest_sha256": "c" * 64}}}
        self.score_manifest = {
            "status": "complete", "question_count": 2, "pair_count": 40,
            "contract": {"reranker": {"model_id": "BAAI/bge-reranker-v2-m3", "model_revision": "revision",
                                       "model_weights_sha256": "d" * 64, "implementation_sha256": "e" * 64,
                                       "inference_precision": "float16", "document_rendering": "version1",
                                       "tokenizer_config_sha256": "f" * 64, "score_type": "raw_logit", "normalized": False}}}
        self.save()

    def save(self):
        write_rows(self.root / "candidates.jsonl", self.candidates)
        write_rows(self.root / "scores.jsonl", self.scores)
        candidate_sha = fingerprint(self.root / "candidates.jsonl")
        self.candidate_manifest["candidate_file_sha256"] = candidate_sha
        self.score_manifest["scores_sha256"] = fingerprint(self.root / "scores.jsonl")
        self.score_manifest["contract"]["candidate_file_sha256"] = candidate_sha
        write_json(self.root / "candidate_manifest.json", self.candidate_manifest)
        write_json(self.root / "score_manifest.json", self.score_manifest)

    def policy(self):
        _, _, binding = load_cached_context(self.root, "q1")
        calibrated = calibrate_threshold([{"question_id": "dev", "split": "dev", "max_score": 10,
                                           "complete_evidence": True}], target_precision=1, min_accepted=1)
        return make_gate_policy(calibrated, binding)


class CachedContextTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.run = CachedRun(Path(self.folder.name))

    def test_verified_replay_returns_reranked_source_payload_without_labels(self):
        query, selected, binding = load_cached_context(self.run.root, "q1")
        self.assertEqual(query, "Question q1?")
        self.assertEqual([c["chunk_id"] for c in selected], [f"q1-c{i}" for i in range(20, 15, -1)])
        self.assertEqual([c["reranker_score"] for c in selected], [20, 19, 18, 17, 16])
        self.assertEqual(selected[0]["text"], "Source paragraph 20.")
        self.assertEqual(selected[0]["vector_rank"], 20)
        self.assertEqual(binding["candidate_depth"], 20)
        self.assertNotIn("ground_truth_doc_ids", selected[0])

    def test_any_candidate_or_score_byte_change_requires_updated_provenance(self):
        for name, message in (("candidates.jsonl", "Candidate artifact changed"), ("scores.jsonl", "Score artifact changed")):
            with self.subTest(name=name):
                self.run.save()
                path = self.run.root / name
                with path.open("ab") as stream:
                    stream.write(b"\n")
                with self.assertRaisesRegex(ValueError, message):
                    load_cached_context(self.run.root, "q1")

    def test_incomplete_score_rows_or_partial_status_are_rejected(self):
        self.run.scores.pop()
        self.run.save()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_cached_context(self.run.root, "q1")
        self.run = CachedRun(self.run.root)
        self.run.score_manifest["status"] = "partial"
        self.run.save()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_cached_context(self.run.root, "q1")

    def test_manifest_question_order_and_score_candidate_order_are_enforced(self):
        self.run.candidates.reverse()
        self.run.save()
        with self.assertRaisesRegex(ValueError, "identity/order"):
            load_cached_context(self.run.root, "q1")
        self.run = CachedRun(self.run.root)
        self.run.scores[0]["candidate_ids"].reverse()
        self.run.save()
        with self.assertRaisesRegex(ValueError, "query/candidate order"):
            load_cached_context(self.run.root, "q1")

    def test_unknown_question_and_duplicate_question_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown question"):
            load_cached_context(self.run.root, "missing")
        self.run.candidates[1]["question_id"] = "q1"
        self.run.candidate_manifest["question_ids"] = ["q1", "q1"]
        self.run.save()
        with self.assertRaisesRegex(ValueError, "identity/order"):
            load_cached_context(self.run.root, "q1")

    def test_vector_ranking_and_chunk_strategy_contracts_are_preserved(self):
        self.run.candidates[0]["candidates"][0]["rank"] = 20
        self.run.save()
        with self.assertRaisesRegex(ValueError, "original vector Top20"):
            load_cached_context(self.run.root, "q1")
        self.run = CachedRun(self.run.root)
        self.run.candidates[0]["candidates"][0]["strategy_id"] = "different"
        self.run.save()
        with self.assertRaisesRegex(ValueError, "Mixed chunk strategies"):
            load_cached_context(self.run.root, "q1")


class AnswerCliTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def invoke(self, args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(args)
        return code, json.loads(output.getvalue())

    def test_unconfigured_threshold_refuses_without_source_reads_or_model_construction(self):
        source = self.root / "input.json"
        write_json(source, {"query": "What was revenue?", "candidates": [{"chunk_id": "c1", "reranker_score": 100}]})
        with patch("report_search.generation.ChatCompletionsClient", side_effect=AssertionError("No model should load")) as model, \
                patch("report_search.answer_cli.SourceEvidenceStore.build", side_effect=AssertionError("No sources should load")) as store:
            code, result = self.invoke(["--input", str(source)])
        self.assertEqual(code, 2)
        self.assertEqual(result["answer"], "I'm not sure")
        self.assertEqual(result["reason"], "unconfigured_threshold")
        self.assertFalse(result["audit"]["generation_called"])
        model.assert_not_called()
        store.assert_not_called()

    def test_cached_policy_binding_mismatch_refuses_before_generation(self):
        run = CachedRun(self.root)
        policy = run.policy()
        policy["binding"]["reranker"]["model_revision"] = "different-revision"
        path = self.root / "policy.json"
        write_json(path, policy)
        with patch("report_search.answer_cli.generate_answer", side_effect=AssertionError("No generation should occur")) as generator:
            code, result = self.invoke(["--question-id", "q1", "--reranker-run", str(self.root), "--gate-policy", str(path)])
        self.assertEqual(code, 2)
        self.assertEqual(result["reason"], "input_or_policy_validation_failed")
        self.assertFalse(result["audit"]["generation_called"])
        generator.assert_not_called()

    def test_cached_policy_cannot_be_applied_to_arbitrary_caller_scores(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--input", str(self.root / "input.json"), "--gate-policy", str(self.root / "policy.json")])
        self.assertEqual(raised.exception.code, 2)

    def test_corrupt_cache_and_unknown_query_produce_same_public_abstention(self):
        CachedRun(self.root)
        _, result = self.invoke(["--question-id", "missing", "--reranker-run", str(self.root)])
        self.assertEqual(result["answer"], "I'm not sure")
        path = self.root / "scores.jsonl"
        with path.open("ab") as stream:
            stream.write(b"\n")
        _, changed = self.invoke(["--question-id", "q1", "--reranker-run", str(self.root)])
        self.assertEqual(result, changed)

    def test_cli_refuses_nonfinite_cutoff_and_protects_input_from_output_overwrite(self):
        source = self.root / "input.json"
        write_json(source, {"query": "Question?", "candidates": []})
        before = source.read_bytes()
        for extras in (["--threshold", "nan"], ["--threshold", "inf"], ["--output", str(source)]):
            with self.subTest(extras=extras), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(["--input", str(source), *extras])
            self.assertEqual(raised.exception.code, 2)
        self.assertEqual(source.read_bytes(), before)

    def test_caller_input_extra_answer_or_gold_label_fields_are_rejected(self):
        source = self.root / "input.json"
        write_json(source, {"query": "Question?", "candidates": [], "answer": "A leaked benchmark answer"})
        _, result = self.invoke(["--input", str(source)])
        self.assertEqual(result["reason"], "input_or_policy_validation_failed")
        self.assertNotIn("leaked", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
