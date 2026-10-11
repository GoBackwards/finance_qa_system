import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest.mock import Mock, patch

from report_search.demo_service import DemoComponents, DemoError, DemoService, _default_factory
from report_search.llm_client import ModelConfig


TEXT = "Revenue was $10 million."
CANDIDATE = {"chunk_id": "c1", "document_id": "DEMO", "text": TEXT, "score": 0.8, "rank": 1}
BUNDLE = {"source_verified": True, "evidence": {"E1": {
    "evidence_id": "E1", "text": TEXT, "kind": "prose",
    "citation": {"document_id": "DEMO", "source_file": "DEMO.pdf", "sha256": "a" * 64, "pdf_page": 7},
    "source_locator": {"element_id": "p7_b2", "char_start": 0, "char_end": len(TEXT)}, "chunk_ids": ["c1"]}},
    "chunks": [{"chunk_id": "c1", "text": TEXT, "evidence_ids": ["E1"]}]}
DRAFT = {"status": "answer", "claims": [{"text": TEXT, "citations": [{"evidence_id": "E1", "quote": TEXT}]}]}
VERDICT = {"answers_question": True, "claim_verdicts": [{"claim_index": 0, "supported": True, "reason": "Direct statement."}]}


class Resources:
    def __init__(self, *, fail_reranker=False):
        self.calls = []
        self.builds = 0
        self.fail_reranker = fail_reranker
        self.source_inputs = []
        self.connection = None

    def record(self, name):
        self.calls.append((name, threading.get_ident()))

    def factory(self, root, device):
        self.builds += 1
        self.record("factory")
        self.connection = sqlite3.connect(":memory:")
        owner = self

        class Encoder:
            def encode_queries(self, queries):
                owner.record("encode")
                return [[1.0, 0.0]]

        class Engine:
            stats = {"chunk_count": 1, "document_count": 1, "dimension": 2}

            def vector(self, vector, top_k):
                owner.record("vector")
                owner.connection.execute("SELECT 1").fetchone()
                if top_k != 20:
                    raise AssertionError("Wrong vector candidate depth")
                return [copy.deepcopy(CANDIDATE)]

            def close(self):
                owner.record("close")
                owner.connection.close()

        class Reranker:
            def rank_candidates(self, query, candidates, top_k):
                owner.record("rerank")
                if owner.fail_reranker:
                    raise RuntimeError("private-error-content")
                if top_k != 5:
                    raise AssertionError("Wrong reranker candidate depth")
                selected = copy.deepcopy(candidates)
                selected[0].update(score=8.0, reranker_score=8.0, vector_score=0.8,
                                   vector_rank=1, retrieval_approach="vector_reranker")
                return selected

        class Source:
            def build(self, candidates):
                owner.record("source")
                owner.source_inputs.append(copy.deepcopy(candidates))
                return copy.deepcopy(BUNDLE)

        return DemoComponents(Encoder(), Engine(), Reranker(), Source())


class DemoServiceTests(unittest.TestCase):
    def test_single_worker_owns_and_reuses_sqlite_across_calling_threads(self):
        resources = Resources()
        with DemoService(".", worker_factory=resources.factory) as service:
            self.assertEqual(resources.builds, 0)
            with ThreadPoolExecutor(max_workers=3) as callers:
                results = list(callers.map(service.search, ["Question one?", "Question two?", "Question three?"]))
            self.assertEqual([item["query"] for item in results], ["Question one?", "Question two?", "Question three?"])
            self.assertEqual(resources.builds, 1)
            self.assertEqual([name for name, _ in resources.calls], ["factory"] + ["encode", "vector", "rerank"] * 3)
            self.assertTrue(all("queue_wait_ms" in result["timings"] for result in results))
        self.assertEqual(resources.calls[-1][0], "close")
        self.assertEqual(len({thread for _, thread in resources.calls}), 1)
        self.assertNotEqual(resources.calls[0][1], threading.get_ident())

    def test_queries_and_returned_payloads_are_not_cached(self):
        resources = Resources()
        with DemoService(".", worker_factory=resources.factory) as service:
            first = service.search("Revenue?")
            first["selected"][0]["text"] = "mutated"
            second = service.search("Revenue?")
            self.assertEqual(second["selected"][0]["text"], TEXT)
        self.assertEqual(sum(name == "encode" for name, _ in resources.calls), 2)

    def test_reranker_failure_retains_vector_evidence_and_sanitizes_error(self):
        resources = Resources(fail_reranker=True)
        with DemoService(".", worker_factory=resources.factory) as service:
            with self.assertRaises(DemoError) as caught:
                service.search("Revenue?")
            self.assertEqual(caught.exception.code, "reranking_failed")
            self.assertNotIn("private-error-content", str(caught.exception))
            partial = caught.exception.partial_result
            self.assertEqual(partial["candidates"], [CANDIDATE])
            self.assertEqual(partial["selected"], [])
            with self.assertRaisesRegex(DemoError, "invalid_search_result"):
                service.answer(partial, threshold=6)

    def test_missing_threshold_or_provider_refuses_without_implicit_environment(self):
        resources = Resources()
        with DemoService(".", worker_factory=resources.factory) as service:
            result = service.search("Revenue?")
            with patch.object(ModelConfig, "from_env", side_effect=AssertionError("Must not read shared environment")):
                no_threshold = service.answer(result)
                no_provider = service.answer(result, threshold=6)
            self.assertEqual(no_threshold["reason"], "unconfigured_threshold")
            self.assertEqual(no_provider["reason"], "model_not_configured")
            self.assertEqual(no_provider["answer"], "I'm not sure")

    def test_answer_uses_selected_sources_and_per_request_model_configs(self):
        resources = Resources()
        observed = []

        class Client:
            def __init__(self, config):
                observed.append((config.model, config.api_key, threading.get_ident()))
                self.identity = {"backend": "fixture", "model": config.model}

            def complete_json(self, messages):
                return copy.deepcopy(DRAFT if "You answer questions" in messages[0]["content"] else VERDICT)

        with DemoService(".", worker_factory=resources.factory) as service:
            search = service.search("Revenue?")
            with patch("report_search.demo_service.ChatCompletionsClient", Client):
                first = service.answer(search, threshold=6,
                                       generator_config=ModelConfig("https://provider.example/v1", "g1", api_key="key-one"),
                                       verifier_config=ModelConfig("https://provider.example/v1", "v1", api_key="key-two"))
                second = service.answer(search, threshold=6,
                                        generator_config=ModelConfig("https://provider.example/v1", "g2", api_key="key-three"))
            self.assertEqual(resources.source_inputs, [search["selected"], search["selected"]])
            self.assertEqual(first["status"], "answer")
            self.assertEqual(second["status"], "answer")
            self.assertEqual(first["citations"][0]["citation"]["pdf_page"], 7)
            self.assertNotIn("key-one", json.dumps(first))
        self.assertEqual([(model, key) for model, key, _ in observed],
                         [("g1", "key-one"), ("v1", "key-two"), ("g2", "key-three"), ("g2", "key-three")])
        self.assertEqual({thread for _, _, thread in observed}, {resources.calls[0][1]})

    def test_scoped_search_balances_companies_and_recomputes_scope_for_answers(self):
        calls = {"vector": [], "rerank": [], "generate": []}

        class Scope:
            def __init__(self, query):
                self.query = query

            def to_dict(self):
                return {"tickers": ["KO", "PEP"], "years": [2022], "period_mode": "explicit",
                        "report_years": {"KO": [2022], "PEP": [2022]}, "latest_year": {"KO": 2023, "PEP": 2023},
                        "notes": [f"scope for {self.query}"]}

        class Resolver:
            def resolve(self, query):
                return Scope(query)

            def entities_in_text(self, text):
                return frozenset({"unknown:Apple"}) if "Apple" in text else frozenset()

        def chunk(ticker, number):
            # Chunk 12 is a margin table below the pool depth: only the context-extras search finds it.
            kind = {"kind": "table", "rows": [["Operating margin", "12.0"]]} if number == 12 else {"kind": "prose"}
            return {"chunk_id": f"{ticker}{number}", "document_id": f"{ticker}-2022", "ticker": ticker,
                    "fiscal_year": 2022, "text": f"{ticker} passage {number}", "rank": number, "score": 0.5, **kind}

        class Engine:
            stats = {}

            def vector(self, vector, top_k, document_ids=None):
                calls["vector"].append((top_k, tuple(document_ids or ())))
                return [chunk(document_ids[0].split("-")[0], number) for number in range(1, top_k + 1)]

            def neighbour(self, chunk_id, offset):
                return None

            def bm25_report(self, question, top_k, *, ticker, fiscal_year):
                return []

            def close(self):
                pass

        class Reranker:
            def rank_candidates(self, query, candidates, top_k):
                calls["rerank"].append(top_k)
                # PEP dominates the scores; selection must still keep KO evidence.
                scored = [dict(c, reranker_score=(9.0 if c["ticker"] == "PEP" else 1.0) - c["rank"] / 100,
                               vector_rank=c["rank"]) for c in candidates]
                return sorted(scored, key=lambda c: -c["reranker_score"])[:top_k]

        class Encoder:
            def encode_queries(self, queries):
                return [[1.0, 0.0]]

        def factory(root, device):
            return DemoComponents(Encoder(), Engine(), Reranker(), object(), scope_resolver=Resolver(),
                                  documents={"KO": {2022: "KO-2022"}, "PEP": {2022: "PEP-2022"}})

        def fake_generate(query, selected, **options):
            calls["generate"].append(options["scope"])
            # Claim grounding must see out-of-corpus companies too, not only corpus tickers.
            calls["entities"] = options["company_resolver"]("Apple's revenue")
            return {"status": "abstain", "answer": "I'm not sure", "citations": [], "reason": "fixture"}

        with DemoService(".", worker_factory=factory) as service:
            search = service.search("coke vs pepsi 2022 operating margin?")
            self.assertEqual(calls["vector"], [(10, ("KO-2022",)), (10, ("PEP-2022",)),
                                               (20, ("KO-2022",)), (20, ("PEP-2022",))])
            self.assertEqual(calls["rerank"], [20, 2])  # the pool, then the two context extras
            self.assertEqual(len(search["selected"]), 7)
            self.assertEqual(sum(row["ticker"] == "KO" for row in search["selected"][:5]), 2)
            extras = search["selected"][5:]
            self.assertEqual(sorted(row["chunk_id"] for row in extras), ["KO12", "PEP12"])
            self.assertEqual([row["rank"] for row in extras], [6, 7])
            self.assertTrue(all("reranker_score" in row for row in extras))
            self.assertEqual({row["reason"] for row in search["retrieval_trace"]["context_extras"]}, {"row_label_table"})
            self.assertEqual(search["retrieval_trace"]["quota"], {"KO": 2, "PEP": 2})
            self.assertEqual(search["scope"]["tickers"], ["KO", "PEP"])
            tampered = copy.deepcopy(search)
            tampered["scope"]["tickers"] = ["XYZ"]
            with patch("report_search.demo_service.generate_answer", side_effect=fake_generate):
                service.answer(tampered, threshold=6)
            self.assertEqual(calls["generate"][0]["tickers"], ["KO", "PEP"])
            self.assertIn("coke vs pepsi", calls["generate"][0]["notes"][0])
            self.assertEqual(calls["entities"], frozenset({"unknown:Apple"}))

    def test_corpus_versions_are_explicit_and_complete(self):
        from report_search.demo_service import CORPORA
        self.assertEqual(DemoService(".", worker_factory=Resources().factory).corpus, "v1")
        with self.assertRaisesRegex(DemoError, "invalid_corpus"):
            DemoService(".", corpus="v9")
        for version, paths in CORPORA.items():
            self.assertEqual(set(paths), {"processed", "chunks", "vectors", "bm25"}, version)
        self.assertEqual(len({paths["vectors"] for paths in CORPORA.values()}), len(CORPORA))  # never shared

    def test_tampered_or_cross_service_search_is_rejected_before_generation(self):
        resources = Resources()
        with DemoService(".", worker_factory=resources.factory) as first, DemoService(".", worker_factory=Resources().factory) as second:
            result = first.search("Revenue?")
            for field in ("query", "selected"):
                changed = copy.deepcopy(result)
                if field == "query":
                    changed[field] = "Different query?"
                else:
                    changed[field][0]["reranker_score"] = 100.0
                with self.assertRaisesRegex(DemoError, "invalid_search_result"):
                    first.answer(changed, threshold=6)
            with self.assertRaisesRegex(DemoError, "invalid_search_result"):
                second.answer(result, threshold=6)
            self.assertEqual(resources.source_inputs, [])

    def test_invalid_query_and_closed_service_never_start_resources(self):
        factory = Mock()
        service = DemoService(".", worker_factory=factory)
        for query in ("", " \n", "invalid\ud800", None, "x" * 10_001):
            with self.subTest(query=repr(query)[:30]), self.assertRaisesRegex(DemoError, "invalid_query"):
                service.search(query)
        service.close()
        service.close()
        with self.assertRaisesRegex(DemoError, "service_closed"):
            service.search("Valid query?")
        factory.assert_not_called()

    def test_startup_failure_is_sanitized_and_can_retry(self):
        resources = Resources()
        calls = []

        def factory(root, device):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("private-loader-details")
            return resources.factory(root, device)

        with DemoService(".", worker_factory=factory) as service:
            with self.assertRaisesRegex(DemoError, "^startup_failed$"):
                service.search("Revenue?")
            self.assertTrue(service.search("Revenue?")["selected"])

    def test_atexit_cleanup_runs_on_owning_worker_after_search(self):
        script = textwrap.dedent('''
            import atexit, json, sqlite3, sys, threading
            from pathlib import Path
            from report_search.demo_service import DemoComponents, DemoService
            marker = Path(sys.argv[1])
            def factory(root, device):
                owner = threading.get_ident()
                connection = sqlite3.connect(':memory:')
                class Encoder:
                    def encode_queries(self, queries): return [[1.0]]
                class Engine:
                    stats = {}
                    def vector(self, query, top_k):
                        connection.execute('SELECT 1').fetchone()
                        return []
                    def close(self):
                        connection.close()
                        marker.write_text(json.dumps({'owner': owner, 'closed_on': threading.get_ident()}))
                class Reranker:
                    def rank_candidates(self, query, candidates, top_k): return []
                return DemoComponents(Encoder(), Engine(), Reranker(), object())
            service = DemoService('.', worker_factory=factory)
            atexit.register(service.close)
            service.search('Revenue?')
            print('search_completed', flush=True)
        ''')
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "closed.json"
            process = subprocess.run([sys.executable, "-c", script, str(marker)],
                                     cwd=Path(__file__).resolve().parents[1],
                                     text=True, capture_output=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("search_completed", process.stdout)
            self.assertNotIn("Exception ignored in atexit callback", process.stderr)
            self.assertTrue(marker.is_file(), process.stderr)
            recorded = json.loads(marker.read_text())
            self.assertEqual(recorded["owner"], recorded["closed_on"])

    def test_default_factory_rejects_mismatched_artifacts_before_loading_weights(self):
        identity = {"model_id": "BAAI/bge-m3", "model_revision": "a" * 40, "dimension": 2,
                    "pooling": "cls", "normalized": True, "query_instruction": "",
                    "tokenizer_config_sha256": "tokenizer", "document_rendering": "rendering"}
        metadata = {"encoder": identity, "status": "complete", "normalized": True,
                    "dimension": 2, "source_sha256": "chunks", "vector_count": 1}
        inventory = [{"relative_path": "data/raw/DEMO/DEMO.pdf", "sha256": "b" * 64, "status": "validated"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data/indexes/bge-m3/table_preserving_s512_o50/embeddings/embedding_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(metadata), encoding="utf-8")
            (root / "data/manifest.json").write_text(json.dumps(inventory), encoding="utf-8")
            for mismatch in ("encoder", "vectors", "inventory", "source"):
                with self.subTest(mismatch=mismatch):
                    encoder = Mock(identity={**identity, "pooling": "mean"} if mismatch == "encoder" else identity)
                    engine = Mock(stats={"dimension": 2, "chunks_sha256": "wrong" if mismatch == "source" else "chunks",
                                         "chunk_count": 1, "document_ids": ["WRONG"] if mismatch == "inventory" else ["DEMO-" + "b" * 12]},
                                  vector_manifest={"vectors_file_sha256": "vectors"})
                    with patch("report_index.embedding.DenseEncoder", return_value=encoder), \
                         patch("report_index.embedding.sha256_file", return_value="wrong" if mismatch == "vectors" else "vectors"), \
                         patch("report_search.retrieval.RetrievalEngine", return_value=engine):
                        with self.assertRaises(ValueError):
                            _default_factory(root, "cpu")
                    engine.close.assert_called_once()
                    encoder._load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
