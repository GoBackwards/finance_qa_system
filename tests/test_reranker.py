"""Reranking score/order/source-preservation checks without model downloads."""
import copy
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from report_search.reranker import (
    BGEReranker, MODEL_REVISION, PairTokenLimitError, _bounded_batch,
    _safe_flag_single_gpu, apply_scores,
)


class CharacterTokenizer:
    """Characters are tokens; paired inputs have four special tokens."""

    def __call__(self, texts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        return {"input_ids": [[ord(char) for char in text] for text in texts]}

    def build_inputs_with_special_tokens(self, query, passage):
        return [0] + query + [2, 2] + passage + [2]


def fake_reranker(max_tokens=8192):
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.max_tokens = reranker.query_max_length = max_tokens
    reranker.batch_size = 4
    reranker.tokenizer = CharacterTokenizer()
    reranker._model = Mock()
    reranker._model.compute_score.return_value = [1.0]
    reranker._load_model = Mock(return_value=reranker._model)
    return reranker


def candidate(chunk_id, rank=1, score=.7):
    return {
        "chunk_id": chunk_id, "document_id": "report-a", "rank": rank,
        "score": score, "retrieval_approach": "vector", "ticker": "TEST",
        "fiscal_year": 2024, "company": "Example Corp", "kind": "table",
        "text": "| Year | Revenue |\n| 2024 | $42 |",
        "rows": [{"cells": ["2024", "$42"]}],
        "evidence": [{"kind": "table_row", "row_index": 1,
                      "citation": {"document_id": "report-a", "pdf_page": 31}}],
        "citations": [{"document_id": "report-a", "pdf_page": 31}],
    }


class ApplyScoresTests(unittest.TestCase):
    def test_raw_logit_order_with_negative_scores_and_stable_rank_ties(self):
        candidates = [candidate("z", 3), candidate("b", 1), candidate("a", 1), candidate("x", 2)]
        result = apply_scores(candidates, [-2.0, -2.0, -2.0, 1.5], top_k=3)
        self.assertEqual([row["chunk_id"] for row in result], ["x", "a", "b"])
        self.assertEqual([row["score"] for row in result], [1.5, -2.0, -2.0])
        self.assertEqual([row["rank"] for row in result], [1, 2, 3])
        self.assertEqual([row["vector_rank"] for row in result], [2, 1, 1])

    def test_table_rows_citations_and_nested_payload_preserved_independently(self):
        original = [candidate("a", 1), candidate("b", 2)]
        before = copy.deepcopy(original)
        result = apply_scores(original, [.2, .8])
        self.assertEqual(original, before)
        selected = result[0]
        self.assertEqual(selected["rows"], original[1]["rows"])
        self.assertEqual(selected["text"], original[1]["text"])
        self.assertEqual(selected["evidence"], original[1]["evidence"])
        self.assertEqual(selected["citations"], original[1]["citations"])
        self.assertEqual(selected["vector_score"], .7)
        self.assertEqual(selected["reranker_score"], .8)
        selected["rows"][0]["cells"][1] = "modified"
        self.assertEqual(original, before)

    def test_missing_vector_ranks_use_input_order(self):
        rows = [candidate("z"), candidate("a")]
        for row in rows:
            row.pop("rank")
        self.assertEqual([row["chunk_id"] for row in apply_scores(rows, [1.0, 1.0])], ["z", "a"])

    def test_duplicate_ids_invalid_scores_and_wrong_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            apply_scores([candidate("a"), candidate("a")], [1., 2.])
        for scores in ([1., 2.], [], [float("nan")], [float("inf")], ["1"], [True], None):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                apply_scores([candidate("a")], scores)

    def test_invalid_cutoff_identifiers_and_original_rank_fail(self):
        for top_k in (0, -1, True, 2.5):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                apply_scores([], [], top_k)
        for row in ({}, candidate(""), candidate("a", rank=0), candidate("a", score=float("nan"))):
            with self.subTest(row=row), self.assertRaises(ValueError):
                apply_scores([row], [1.])
        self.assertEqual(apply_scores([], []), [])


class RerankerAdapterTests(unittest.TestCase):
    def test_bounded_flag_backend_restores_pair_order_and_keeps_complete_text(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Tokens(CharacterTokenizer):
            def __call__(self, queries, passages, **kwargs):
                self.options = kwargs
                return {"input_ids": [self.build_inputs_with_special_tokens(
                    [ord(char) for char in query], [ord(char) for char in passage])
                    for query, passage in zip(queries, passages)]}

            def pad(self, items, **kwargs):
                return Batch(input_ids=[item["input_ids"] for item in items])

        class Tensor:
            def __init__(self, scores):
                self.scores = scores

            def view(self, *args):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.scores

        class Model:
            def half(self):
                pass

            def to(self, device):
                return self

            def eval(self):
                pass

            def __call__(self, input_ids, **kwargs):
                return SimpleNamespace(logits=Tensor([float(row[-2]) for row in input_ids]))

        flag = SimpleNamespace(batch_size=2, max_length=8192, query_max_length=8192,
                               normalize=False, target_devices=["cpu"], use_fp16=False,
                               model=Model(), tokenizer=Tokens())
        fake_torch = SimpleNamespace(inference_mode=nullcontext,
                                     cuda=SimpleNamespace(OutOfMemoryError=MemoryError, empty_cache=Mock()))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            scores = _safe_flag_single_gpu(flag, [("xx", "z"), ("q", "a"), ("long query", "m")])
            self.assertEqual(scores, [122.0, 97.0, 109.0])
            self.assertEqual(flag.tokenizer.options,
                             {"add_special_tokens": True, "truncation": False, "padding": False})
            with self.assertRaisesRegex(ValueError, "refusing to truncate"):
                _safe_flag_single_gpu(flag, [("query", "passage")], max_length=8)

    def test_cuda_oom_retry_stops_at_one_and_other_errors_propagate(self):
        class CUDAOutOfMemory(RuntimeError):
            pass

        attempted = []
        clear_cache = Mock()

        def too_large(size):
            attempted.append(size)
            if size > 1:
                raise CUDAOutOfMemory("too large")
            return [1.5]

        self.assertEqual(_bounded_batch(too_large, 4, CUDAOutOfMemory, clear_cache), ([1.5], 1))
        self.assertEqual(attempted, [4, 2, 1])
        self.assertEqual(clear_cache.call_count, 2)
        failure = Mock(side_effect=CUDAOutOfMemory("cannot fit one pair"))
        with self.assertRaisesRegex(CUDAOutOfMemory, "one pair"):
            _bounded_batch(failure, 1, CUDAOutOfMemory, clear_cache)
        failure.assert_called_once_with(1)
        failure = Mock(side_effect=RuntimeError("model bug"))
        with self.assertRaisesRegex(RuntimeError, "model bug"):
            _bounded_batch(failure, 4, CUDAOutOfMemory, clear_cache)
        failure.assert_called_once_with(4)

    def test_complete_pair_preflight_includes_four_special_tokens(self):
        reranker = fake_reranker(max_tokens=10)
        audit = reranker.preflight_pairs([("ab", "cdef")])
        self.assertEqual(audit["max_pair_tokens"], 10)
        self.assertEqual(audit["max_query_tokens"], 2)
        self.assertEqual(audit["max_passage_tokens"], 4)
        self.assertEqual(audit["status"], "passed")
        with self.assertRaises(PairTokenLimitError) as failure:
            reranker.preflight_pairs([("abc", "cdef")])
        self.assertEqual(failure.exception.audit["overflow_examples"][0]["pair_tokens"], 11)

    def test_late_overflow_stops_all_inference_including_early_valid_pairs(self):
        reranker = fake_reranker(max_tokens=10)
        pairs = [("a", "b")] * 129 + [("abc", "defghi")]
        with self.assertRaises(PairTokenLimitError) as failure:
            reranker.score_pairs(pairs)
        self.assertEqual(failure.exception.audit["pair_count"], 130)
        self.assertEqual(failure.exception.audit["overflow_examples"][0]["pair_index"], 129)
        reranker._load_model.assert_not_called()
        reranker._model.compute_score.assert_not_called()

    def test_long_query_is_not_silently_limited_to_three_quarters_of_pair_limit(self):
        reranker = fake_reranker(max_tokens=100)
        self.assertEqual(reranker.pair_token_lengths([("q" * 90, "doc")]), [97])
        reranker.score_pairs([("q" * 90, "doc")])
        self.assertEqual(reranker._model.compute_score.call_args.kwargs["query_max_length"], 100)

    def test_explicit_flag_limits_and_raw_scores_keep_order(self):
        reranker = fake_reranker()
        pairs = [("Revenue?", "| FY2024 | $42 |"), ("Debt?", "Debt was $6.")]
        reranker._model.compute_score.return_value = [-4.5, 8.25]
        self.assertEqual(reranker.score_pairs(pairs), [-4.5, 8.25])
        reranker._model.compute_score.assert_called_once_with(
            pairs, batch_size=4, max_length=8192, query_max_length=8192, normalize=False)

    def test_nonfinite_or_wrong_model_outputs_fail(self):
        for scores in ([float("nan")], [float("inf")], [], [1., 2.], [True]):
            reranker = fake_reranker()
            reranker._model.compute_score.return_value = scores
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                reranker.score_pairs([("q", "doc")])

    def test_rank_candidates_sends_only_question_source_metadata_and_text(self):
        reranker = fake_reranker()
        row = candidate("a")
        row["answer"] = "SECRET ANSWER THAT MUST NOT BE AN INPUT"
        row["ground_truth_doc_ids"] = ["SECRET GOLD ID"]
        selected = reranker.rank_candidates("How much revenue?", [row])
        pairs = reranker._model.compute_score.call_args.args[0]
        self.assertEqual(pairs[0][0], "How much revenue?")
        self.assertEqual(pairs[0][1], "Company: Example Corp\nTicker: TEST\nAnnual report fiscal year: FY2024\n\n" + row["text"])
        self.assertNotIn("SECRET", pairs[0][1])
        self.assertEqual(selected[0]["citations"], row["citations"])

    def test_empty_pairs_do_not_load_weights_and_malformed_pairs_fail(self):
        reranker = fake_reranker()
        self.assertEqual(reranker.score_pairs([]), [])
        reranker._load_model.assert_not_called()
        for pair in ("a", ("", "a"), ("q", "  "), ("a", "b", "c"), (None, "a")):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                reranker.score_pairs([pair])
        reranker._load_model.assert_not_called()

    def test_bad_options_rejected_before_tokenizer_or_model_access(self):
        for kwargs in ({"revision": "main"}, {"revision": "a" * 40}, {"model_id": "other"},
                       {"device": "cuda"}, {"device": "auto"}, {"batch_size": 0}, {"batch_size": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BGEReranker(**kwargs)

    def test_local_snapshot_must_match_revision_and_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            reranker = BGEReranker.__new__(BGEReranker)
            reranker.revision = MODEL_REVISION
            reranker._local_model_path = Path(directory)
            with self.assertRaisesRegex(ValueError, "match the pinned revision"):
                reranker._snapshot(weights=False)
            snapshot = Path(directory) / MODEL_REVISION
            snapshot.mkdir()
            reranker._local_model_path = snapshot
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                reranker._snapshot(weights=False)

    def test_changed_configuration_or_weight_hash_stops_before_importing_model(self):
        reranker = BGEReranker.__new__(BGEReranker)
        reranker._model = None
        reranker._snapshot = Mock(return_value=Path("snapshot"))
        reranker._tokenizer_hashes = {"different": "hash"}
        with patch("report_search.reranker._sha256", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "Tokenizer/configuration changed"):
                reranker._load_model()
        from report_search.reranker import TOKENIZER_FILES
        reranker._tokenizer_hashes = {name: "hash" for name in TOKENIZER_FILES}
        with patch("report_search.reranker._sha256", return_value="hash"):
            with self.assertRaisesRegex(ValueError, "weights do not match"):
                reranker._load_model()


if __name__ == "__main__":
    unittest.main()
