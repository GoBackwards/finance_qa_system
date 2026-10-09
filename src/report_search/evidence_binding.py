"""Bind retrieved chunks to hash-checked parsed source paragraphs and table rows.

Only the existing table-preserving chunk schema is accepted. Canonical chunks
are reconstructed from selected reports and compared with retrieval payloads;
the model never gets to invent document identities, pages, or source offsets.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from report_chunker.chunking import ChunkConfig, chunk_document
from report_chunker.validation import validate_document_chunks


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"Missing {name}")
    return value


def _json(value) -> str:
    # JSON equality also rejects booleans substituted for numeric coordinates.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _read_jsonl(path: Path, expected_sha: str) -> list[dict]:
    _require(isinstance(expected_sha, str) and bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha)),
             f"Missing parsed artifact checksum: {path.name}")
    digest, records = hashlib.sha256(), []
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                value = json.loads(line)
                _require(isinstance(value, dict), f"Invalid source record: {path.name}")
                records.append(value)
    _require(digest.hexdigest() == expected_sha, f"Parsed artifact checksum mismatch: {path.name}")
    return records


class SourceEvidenceStore:
    """A bounded cache of reconstructed chunks from selected source documents.

    Cache entries are invalidated when any source artifact's stat changes.
    Canonical payloads stay internal; returned dictionaries are detached copies.
    No benchmark answer, label, or retrieval score is forwarded to the model.
    """

    def __init__(self, processed_root: str | Path = "data/processed", *, cache_documents: int = 4):
        _require(isinstance(cache_documents, int) and not isinstance(cache_documents, bool)
                 and cache_documents > 0, "cache_documents must be a positive integer")
        self.processed_root = Path(processed_root).resolve()
        self.cache_documents = cache_documents
        self._cache = OrderedDict()

    def _canonical(self, candidate: dict) -> dict:
        ticker = _text(candidate.get("ticker"), "ticker")
        source_file = _text(candidate.get("source_file"), "source_file")
        _require(bool(re.fullmatch(r"[A-Za-z0-9.-]+", ticker)) and ticker not in {".", ".."},
                 "Unsafe ticker path")
        _require("/" not in source_file and "\\" not in source_file and ":" not in source_file
                 and source_file.lower().endswith(".pdf"), "Unsafe source_file path")
        folder = (self.processed_root / ticker / Path(source_file).stem).resolve()
        _require(folder.is_relative_to(self.processed_root), "Source path leaves processed_root")
        paths = [folder / name for name in ("document.json", "pages.jsonl", "tables.jsonl")]
        _require(all(path.is_file() and path.resolve().is_relative_to(self.processed_root) for path in paths),
                 "Missing or unsafe parsed source artifacts")
        _require(candidate.get("schema_version") == "1.0", "Unsupported chunk schema")
        _require(candidate.get("tokenizer") == "cl100k_base", "Unsupported chunk tokenizer")
        config = ChunkConfig(candidate.get("chunk_size"), candidate.get("requested_overlap"),
                             candidate.get("tokenizer"))
        _require(candidate.get("strategy_id") == config.strategy_id, "Chunk strategy/config mismatch")
        signature = tuple((path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                          for path in paths)
        key = (str(folder), config)
        cached = self._cache.get(key)
        if cached is not None and cached[0] == signature:
            self._cache.move_to_end(key)
            return cached[1]
        metadata = json.loads(paths[0].read_text(encoding="utf-8"))
        _require(isinstance(metadata, dict) and metadata.get("complete") is True
                 and not metadata.get("partial"), "Source parse is incomplete")
        sha = metadata.get("sha256")
        _require(isinstance(sha, str) and bool(re.fullmatch(r"[0-9a-f]{64}", sha)), "Invalid source SHA256")
        _require(metadata.get("document_id") == f"{Path(source_file).stem}-{sha[:12]}",
                 "Source document ID does not match its file/hash identity")
        _require(metadata.get("source_file") == source_file and metadata.get("ticker") == ticker,
                 "Source path identity mismatch")
        pages, tables = [_read_jsonl(path, metadata.get("artifacts", {}).get(path.name, {}).get("sha256"))
                         for path in paths[1:]]
        canonical = list(chunk_document(metadata, pages, tables, config))
        validate_document_chunks(metadata, pages, tables, canonical)
        by_id = {chunk["chunk_id"]: chunk for chunk in canonical}
        _require(len(by_id) == len(canonical), "Duplicate canonical chunk ID")
        self._cache[key] = (signature, by_id)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_documents:
            self._cache.popitem(last=False)
        return by_id

    @staticmethod
    def _validate_candidate(candidate: dict, canonical: dict) -> None:
        for field, original in canonical.items():
            _require(field in candidate and _json(candidate[field]) == _json(original),
                     f"Candidate changed source field {field}: {canonical['chunk_id']}")
        _require(not any(field in candidate for field in ("candidate_rows", "candidate_cells", "raw_rows")),
                 "Untrusted table candidate data is not allowed")
        if "citations" in candidate:
            expected = list({_json(atom["citation"]): atom["citation"]
                             for atom in canonical["evidence"]}.values())
            _require(_json(candidate["citations"]) == _json(expected), "Candidate citations changed")

    @staticmethod
    def _unit(atom: dict) -> dict:
        kind = atom["kind"]
        if kind == "prose":
            locator = {key: atom[key] for key in ("element_id", "char_start", "char_end")}
            text = atom["text"]
        elif kind == "row":
            locator = {key: atom[key] for key in ("table_id", "row_index")}
            locator["role"] = atom["role"]
            text = atom["row_text"]
        elif kind == "table_source":
            locator = {key: atom[key] for key in ("table_id", "char_start", "char_end")}
            text = atom["text"]
        elif kind == "table_context":
            locator = {key: atom[key] for key in ("table_id", "role")}
            if atom["role"] == "unit_context":
                locator["context_index"] = int(atom["evidence_id"].rsplit(":", 1)[1])
            text = atom["text"]
        else:
            raise ValueError(f"Unsupported source evidence kind: {kind!r}")
        _text(text, "source evidence text")
        return {"text": text, "kind": kind, "citation": deepcopy(atom["citation"]),
                "source_locator": locator}

    def build(self, candidates: list[dict]) -> dict:
        """Return source-verified model context and a local citation registry.

        E1/E2 IDs are stable for the same candidate/evidence order. Identical
        canonical source units shared by overlapping chunks use the same ID.
        Split paragraphs keep their original element-relative offsets, and
        each structured row stays complete in both text and source locator.
        """
        _require(isinstance(candidates, list), "Candidates must be a list")
        evidence, chunks, source_keys, seen = {}, [], {}, set()
        for candidate in candidates:
            _require(isinstance(candidate, dict), "Candidate must be an object")
            chunk_id = _text(candidate.get("chunk_id"), "chunk_id")
            _require(chunk_id not in seen, f"Duplicate candidate chunk_id: {chunk_id}")
            seen.add(chunk_id)
            canonical = self._canonical(candidate).get(chunk_id)
            _require(canonical is not None, f"Unknown source chunk_id: {chunk_id}")
            self._validate_candidate(candidate, canonical)
            identifiers, references, unit_keys = [], [], set()
            for atom in canonical["evidence"]:
                unit = self._unit(atom)
                source_key = _json({key: unit[key] for key in ("kind", "citation", "source_locator")})
                _require(source_key not in unit_keys, "Duplicate source unit within chunk")
                unit_keys.add(source_key)
                identifier = source_keys.get(source_key)
                if identifier is None:
                    identifier = f"E{len(evidence) + 1}"
                    source_keys[source_key] = identifier
                    evidence[identifier] = {"evidence_id": identifier, **unit, "chunk_ids": [chunk_id]}
                else:
                    previous = evidence[identifier]
                    _require(all(previous[key] == value for key, value in unit.items()),
                             "Conflicting payload for shared source unit")
                    previous["chunk_ids"].append(chunk_id)
                identifiers.append(identifier)
                # Exact row text is intentionally separate from escaped Markdown
                # and neutral table scaffolding in the original chunk.
                references.append({"evidence_id": identifier, "text": unit["text"], "kind": unit["kind"]})
            chunks.append({"chunk_id": chunk_id, "text": canonical["text"],
                           "evidence_ids": identifiers, "references": references,
                           **{key: canonical[key] for key in
                              ("document_id", "source_file", "ticker", "fiscal_year", "company", "kind")},
                           **{key: deepcopy(canonical[key]) for key in
                              ("source_quality_flags", "warnings", "structured_reliable") if key in canonical}})
        return {"source_verified": True, "evidence": evidence, "chunks": chunks}


def build_evidence_bundle(candidates: list[dict], processed_root: str | Path = "data/processed") -> dict:
    """Build a verified bundle; reuse SourceEvidenceStore for repeated requests."""
    return SourceEvidenceStore(processed_root).build(candidates)
