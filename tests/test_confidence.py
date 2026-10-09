"""Raw score refusal boundaries, development isolation, and policy compatibility."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from report_search.confidence import (binding_from_manifests, calibrate_threshold, confidence_gate,
                                      evaluate_threshold, load_gate_policy, make_gate_policy)
from scripts.calibrate_generation_gate import run_diagnostic


def row(identifier, score, complete, split="dev", category="factual"):
    return {"question_id": identifier, "max_score": score, "complete_evidence": complete,
            "split": split, "category": category}


def manifests():
    identity = {"model_id": "model", "model_revision": "revision", "model_weights_sha256": "weights",
                "implementation_sha256": "adapter", "inference_precision": "float16",
                "document_rendering": "version1", "tokenizer_config_sha256": "tokens",
                "score_type": "raw_logit", "normalized": False, "device": "cuda:0", "batch_size": 4}
    candidates = {"candidate_depth": 20, "candidate_file_sha256": "candidates", "corpus": {"chunks_sha256": "chunks"},
                  "vector_query_manifest": {"contract": {"corpus_encoder": {"model_revision": "encoder"},
                                                         "corpus_vectors_file_sha256": "vectors",
                                                         "corpus_embedding_manifest_sha256": "embeddings"}}}
    scores = {"status": "complete", "contract": {"candidate_file_sha256": "candidates", "reranker": identity}}
    return candidates, scores


def binding():
    candidates, scores = manifests()
    return binding_from_manifests(candidates, scores, chunk_strategy="table_preserving_s512_o50")


class ConfidenceGateTests(unittest.TestCase):
    def test_maximum_raw_score_and_inclusive_boundary(self):
        candidates = [{"reranker_score": -9, "score": 1}, {"reranker_score": 2.5, "score": 0}]
        original = copy.deepcopy(candidates)
        decision = confidence_gate(candidates, 2.5)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.max_score, 2.5)
        self.assertEqual(decision.to_dict()["score_kind"], "reranker_raw_logit")
        self.assertFalse(confidence_gate(candidates, 2.5001).allowed)
        self.assertEqual(candidates, original)

    def test_unconfigured_invalid_or_empty_inputs_refuse(self):
        for threshold in (None, "0.5", True, 10 ** 1000, float("nan"), float("inf"), float("-inf")):
            with self.subTest(threshold=threshold):
                decision = confidence_gate([{"reranker_score": 9}], threshold)
                self.assertFalse(decision.allowed)
                json.dumps(decision.to_dict(), allow_nan=False)
        self.assertEqual(confidence_gate([], 0).reason, "empty_candidates")
        for candidates in (None, "text", {"reranker_score": 9}):
            self.assertEqual(confidence_gate(candidates, 0).reason, "invalid_candidates")

    def test_never_falls_back_to_vector_hybrid_or_normalized_scores(self):
        for candidate in ({"score": 1}, {"reranker_normalized_score": 1}, {"vector_score": .99},
                          {"score_breakdown": {"vector_raw_score": 1}}, None):
            self.assertFalse(confidence_gate([candidate], -100).allowed)

    def test_one_bad_score_refuses_whole_request_even_if_other_is_high(self):
        for value in (None, True, "99", 10 ** 1000, float("nan"), float("inf"), float("-inf")):
            decision = confidence_gate([{"reranker_score": 10}, {"reranker_score": value}], 1)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "invalid_reranker_score")
        self.assertFalse(confidence_gate([{"reranker_score": 10}, {}], 1).allowed)


class CalibrationTests(unittest.TestCase):
    def test_selects_maximum_dev_coverage_meeting_declared_target(self):
        rows = [row("a", 5, 1), row("b", 4, 1), row("c", 3, 0), row("d", 2, 1), row("e", 1, 0)]
        result = calibrate_threshold(rows, target_precision=.75, min_accepted=2)
        self.assertEqual(result["threshold"], 2)
        self.assertEqual(result["selected_dev"]["accepted_count"], 4)
        self.assertEqual(result["selected_dev"]["accepted_complete_evidence_precision"], .75)
        self.assertEqual(result["selected_dev"]["coverage"], .8)
        self.assertFalse(result["production_certified"])

    def test_score_ties_cannot_be_split_to_improve_precision(self):
        rows = [row("a", 9, 1), row("b", 9, 0), row("c", 1, 1)]
        result = calibrate_threshold(rows, target_precision=1, min_accepted=1)
        self.assertIsNone(result["threshold"])
        self.assertEqual(len(result["dev_tradeoff_grid"]), 2)
        self.assertEqual(result["dev_tradeoff_grid"][0]["accepted_count"], 2)

    def test_no_eligible_threshold_fails_closed_without_infinity(self):
        rows = [row("a", 10, 0), row("b", 1, 1)]
        result = calibrate_threshold(rows, target_precision=.9, min_accepted=2)
        self.assertEqual(result["status"], "refusal_only_no_eligible_threshold")
        self.assertIsNone(result["threshold"])
        self.assertIsNone(result["selected_dev"]["accepted_complete_evidence_precision"])
        self.assertEqual(result["selected_dev"]["accepted_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_minimum_support_prevents_one_lucky_question_threshold(self):
        rows = [row("a", 10, 1), row("b", 9, 0)]
        self.assertIsNone(calibrate_threshold(rows, target_precision=1, min_accepted=2)["threshold"])
        self.assertIsNone(calibrate_threshold([], target_precision=.9, min_accepted=2)["threshold"])

    def test_development_api_rejects_test_rows_and_bad_labels(self):
        for rows in ([row("a", 1, 1, "test")], [row("a", 1, 1), row("a", 2, 0)],
                     [row("a", float("nan"), 1)], [row("a", True, 1)], [row("a", 1, 2)], [row("a", 1, "yes")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                calibrate_threshold(rows, target_precision=.9, min_accepted=1)

    def test_invalid_selection_configuration_rejected(self):
        for precision in (0, -1, 1.1, float("nan"), True, "0.9"):
            with self.assertRaises(ValueError):
                calibrate_threshold([], target_precision=precision, min_accepted=1)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                calibrate_threshold([], target_precision=.9, min_accepted=count)

    def test_selective_metrics_and_wilson_interval_have_hand_computed_counts(self):
        rows = [row("a", 3, 1), row("b", 2, 0), row("c", 1, 1)]
        result = evaluate_threshold(rows, 2)
        self.assertEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["accepted_complete_evidence_precision"], .5)
        self.assertEqual(result["accepted_incomplete_evidence_rate"], .5)
        self.assertEqual(result["retained_complete_evidence_fraction"], .5)
        self.assertEqual(result["refused_complete_evidence_count"], 1)
        low, high = result["accepted_complete_evidence_precision_wilson95"]
        self.assertAlmostEqual(low, 1 - high)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertIsNone(evaluate_threshold([], None)["coverage"])

    def test_test_labels_do_not_change_threshold_or_development_grid(self):
        rows = [row("a", 5, 1), row("b", 4, 1), row("c", 3, 0), row("test", 100, 0, "test")]
        before, policy = run_diagnostic(rows, binding(), target_precision=1, min_accepted=2)
        altered = copy.deepcopy(rows)
        altered[-1].update(max_score=-100, complete_evidence=True)
        after, other_policy = run_diagnostic(altered, binding(), target_precision=1, min_accepted=2)
        self.assertEqual(policy, other_policy)
        self.assertEqual(before["calibration"], after["calibration"])
        self.assertNotEqual(before["test"], after["test"])
        self.assertNotIn("test_tradeoff_grid", before)


class GatePolicyTests(unittest.TestCase):
    def test_valid_policy_roundtrip_and_compatible_hardware_exclusion(self):
        active = binding()
        calibrated = calibrate_threshold([row("a", 3, 1)], target_precision=1, min_accepted=1)
        policy = make_gate_policy(calibrated, active)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            self.assertEqual(load_gate_policy(path, active), policy)
        self.assertNotIn("device", active["reranker"])
        self.assertNotIn("batch_size", active["reranker"])

    def test_policy_model_corpus_selection_and_score_mismatches_rejected(self):
        active = binding()
        policy = make_gate_policy(calibrate_threshold([row("a", 3, 1)], target_precision=1, min_accepted=1), active)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            for field, value in (("chunks_sha256", "changed"), ("chunk_strategy", "changed"), ("candidate_depth", 30),
                                 ("selected_depth", 10)):
                changed = dict(active, **{field: value})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_gate_policy(path, changed)
            changed = copy.deepcopy(active)
            changed["reranker"]["model_revision"] = "different"
            with self.assertRaises(ValueError):
                load_gate_policy(path, changed)
            for field, value in (("threshold", float("inf")), ("score_kind", "hybrid_normalized"),
                                 ("selection_split", "test"), ("production_certified", True), ("status", "unknown"),
                                 ("schema_version", True), ("criterion", None)):
                path.write_text(json.dumps(dict(policy, **{field: value})), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_gate_policy(path, active)

    def test_refusal_only_policy_is_loadable_but_never_allows(self):
        active = binding()
        policy = make_gate_policy(calibrate_threshold([], target_precision=.9, min_accepted=10), active)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            loaded = load_gate_policy(path, active)
        self.assertFalse(confidence_gate([{"reranker_score": 9999}], loaded["threshold"]).allowed)

    def test_manifest_contract_mismatch_or_normalized_model_rejected(self):
        for kind in ("checksum", "normalized", "incomplete"):
            candidates, scores = manifests()
            if kind == "checksum":
                scores["contract"]["candidate_file_sha256"] = "different"
            elif kind == "normalized":
                scores["contract"]["reranker"]["normalized"] = True
            else:
                scores["status"] = "partial"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                binding_from_manifests(candidates, scores, chunk_strategy="strategy")


if __name__ == "__main__":
    unittest.main()
