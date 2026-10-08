"""Embedding invariants using a deterministic encoder, without model downloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from report_index.embedding import (
    DenseEncoder, MODEL_SPECS, TokenLimitError, build_embeddings,
    normalized_float32, preflight_chunks, render_document,
)


class FakeTokenizer:
    """One character per token plus two special tokens, deliberately not BPE."""

    def __call__(self, texts, **kwargs):
        assert kwargs["add_special_tokens"] is True
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        return {"input_ids": [[0] * (len(text) + 2) for text in texts]}


class FakeEncoder:
    dimension = 3

    def __init__(self, max_tokens=512, fail_on_call=None):
        self.max_tokens = max_tokens
        self.identity = {"model_id": "fixture", "model_revision": "a" * 40,
                         "tokenizer_config_sha256": "b" * 64, "max_tokens": max_tokens,
                         "dimension": self.dimension}
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.fail_on_call = fail_on_call

    def encode_documents(self, texts):
        self.calls.append(texts)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("Simulated interruption")
        return np.array([[len(text), 2.0, 3.0] for text in texts], dtype=np.float16)


def record(number, **overrides):
    return {"chunk_id": f"c{number}", "document_id": f"doc{number % 2}",
            "ticker": "TEST", "fiscal_year": 2025, "company": "Example Company",
            "text": f"Revenue was {number + 10} million.", "source_pages": [number + 1],
            "kind": "table" if number % 2 else "prose", **overrides}


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "chunks.jsonl"
        self.output = self.root / "output"

    def write(self, records):
        self.input.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                              encoding="utf-8")

    def test_render_preserves_table_and_adds_only_source_metadata(self):
        table = "| Year | Revenue |\n| 2025 | $42 |"
        text = render_document(record(1, text=table, irrelevant_gold="Do not use this"))
        self.assertEqual(text, "Company: Example Company\nTicker: TEST\nAnnual report fiscal year: FY2025\n\n" + table)
        self.assertNotIn("Do not use", text)

    def test_normalization_produces_float32_unit_vectors(self):
        result = normalized_float32(np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float16),
                                    count=2, dimension=3)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_allclose(result, [[0.6, 0.8, 0], [0, 0, 1]])

    def test_invalid_encoder_output_fails(self):
        for output in ([[0, 0, 0]], [[float("nan"), 2, 3]], [[float("inf"), 2, 3]], [[1, 2]]):
            with self.subTest(output=output), self.assertRaises(ValueError):
                normalized_float32(output, count=1, dimension=3)

    def test_complete_preflight_precedes_any_encoding(self):
        self.write([record(i) for i in range(130)] + [record(130, text="x" * 1000, kind="page_fallback")])
        encoder = FakeEncoder()
        with self.assertRaises(TokenLimitError) as failure:
            build_embeddings(self.input, self.output, encoder, shard_size=2)
        self.assertEqual(encoder.calls, [])
        self.assertEqual(failure.exception.audit["chunk_count"], 131)
        self.assertEqual(failure.exception.audit["overflow_count"], 1)
        self.assertEqual(failure.exception.audit["overflow_by_kind"], {"page_fallback": 1})
        self.assertEqual(failure.exception.audit["overflow_examples"][0]["chunk_id"], "c130")
        self.assertFalse((self.output / "embedding_manifest.json").exists())

    def test_model_token_count_includes_metadata_and_special_tokens(self):
        row = record(1, text="abc")
        self.write([row])
        expected = len(render_document(row)) + 2
        with self.assertRaises(TokenLimitError):
            preflight_chunks(self.input, self.output, FakeEncoder(max_tokens=expected - 1))
        audit = preflight_chunks(self.input, self.output, FakeEncoder(max_tokens=expected))
        self.assertEqual(audit["max_actual_model_tokens"], expected)
        self.assertEqual(audit["status"], "passed")

    def test_shards_count_documents_separately_and_preserve_source_offsets(self):
        rows = [record(i, text=f"Revenue naïve 東京 {i}") for i in range(5)]
        self.write(rows)
        encoder = FakeEncoder()
        progress = []
        result = build_embeddings(self.input, self.output, encoder, shard_size=2, on_progress=progress.append)
        self.assertEqual(result["vector_count"], 5)
        self.assertEqual(result["chunk_count"], 5)
        self.assertEqual(result["document_count"], 2)
        self.assertEqual(result["dimension"], 3)
        self.assertEqual([entry["count"] for entry in result["shards"]], [2, 2, 1])
        self.assertEqual(result["truncated_chunks"], 0)
        metadata = []
        for shard in result["shards"]:
            vectors = np.load(self.output / shard["vectors_path"], allow_pickle=False)
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
            metadata.extend(json.loads(line) for line in
                            (self.output / shard["metadata_path"]).read_text(encoding="utf-8").splitlines())
        with self.input.open("rb") as handle:
            for item, row in zip(metadata, rows):
                handle.seek(item["source_jsonl_offset"])
                self.assertEqual(json.loads(handle.readline()), row)
                self.assertEqual(item["rendered_text_sha256"],
                                 hashlib.sha256(render_document(row).encode("utf-8")).hexdigest())
        self.assertEqual([item["vector_id"] for item in metadata], list(range(5)))
        self.assertEqual(progress[-1]["count"], 5)

    def test_resume_reuses_committed_shards_after_interruption(self):
        self.write([record(i) for i in range(5)])
        first = FakeEncoder(fail_on_call=2)
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            build_embeddings(self.input, self.output, first, shard_size=2)
        preserved = (self.output / "shards/vectors-000000.npy").read_bytes()
        self.assertFalse((self.output / "embedding_manifest.json").exists())
        resumed = FakeEncoder()
        result = build_embeddings(self.input, self.output, resumed, shard_size=2)
        self.assertEqual([len(call) for call in resumed.calls], [2, 1])
        self.assertEqual(result["vector_count"], 5)
        self.assertEqual((self.output / "shards/vectors-000000.npy").read_bytes(), preserved)
        completed = FakeEncoder()
        build_embeddings(self.input, self.output, completed, shard_size=2)
        self.assertEqual(completed.calls, [])

    def test_source_mutation_prevents_resuming_incompatible_vectors(self):
        self.write([record(0)])
        build_embeddings(self.input, self.output, FakeEncoder(), shard_size=2)
        self.write([record(0, text="Different financial evidence")])
        encoder = FakeEncoder()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            build_embeddings(self.input, self.output, encoder, shard_size=2)
        self.assertEqual(encoder.calls, [])

    def test_tokenizer_or_model_change_prevents_resume(self):
        self.write([record(0)])
        build_embeddings(self.input, self.output, FakeEncoder())
        for key in ("model_revision", "tokenizer_config_sha256"):
            encoder = FakeEncoder()
            encoder.identity[key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity differs"):
                build_embeddings(self.input, self.output, encoder)

    def test_corrupt_shard_is_detected_instead_of_reused(self):
        self.write([record(0)])
        build_embeddings(self.input, self.output, FakeEncoder())
        path = self.output / "shards/vectors-000000.npy"
        path.write_bytes(path.read_bytes() + b"damage")
        encoder = FakeEncoder()
        with self.assertRaisesRegex(ValueError, "missing or modified"):
            build_embeddings(self.input, self.output, encoder)
        self.assertEqual(encoder.calls, [])

    def test_source_mutation_during_encoding_has_no_complete_manifest(self):
        self.write([record(0)])
        encoder = FakeEncoder()
        original_encode = encoder.encode_documents

        def mutate(texts):
            self.input.write_text(self.input.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return original_encode(texts)

        encoder.encode_documents = mutate
        with self.assertRaisesRegex(ValueError, "changed during embedding"):
            build_embeddings(self.input, self.output, encoder)
        self.assertFalse((self.output / "embedding_manifest.json").exists())

    def test_duplicate_ids_and_empty_input_rejected(self):
        for rows in ([], [record(0), record(0)]):
            self.write(rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                preflight_chunks(self.input, self.output, FakeEncoder())

    def test_blank_jsonl_lines_rejected_before_encoding(self):
        self.write([record(0)])
        with self.input.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        encoder = FakeEncoder()
        with self.assertRaisesRegex(ValueError, "Blank JSONL line"):
            build_embeddings(self.input, self.output, encoder)
        self.assertEqual(encoder.calls, [])

    def test_queries_receive_instruction_only_for_english_model(self):
        for model, spec in MODEL_SPECS.items():
            encoder = DenseEncoder.__new__(DenseEncoder)
            encoder.query_instruction = spec["query_instruction"]
            encoder._encode = lambda texts: texts
            with self.subTest(model=model):
                self.assertEqual(encoder.encode_queries(["Revenue?"]), [spec["query_instruction"] + "Revenue?"])
                self.assertEqual(encoder.encode_documents(["Revenue."]), ["Revenue."])

    def test_encoder_rejects_mutable_revision_before_model_access(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            DenseEncoder(revision="main")


if __name__ == "__main__":
    unittest.main()
