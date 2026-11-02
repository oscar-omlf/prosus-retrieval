"""Pinned multilingual E5 inference with ONNX Runtime; no network at query time.

The official model card specifies query/passage prefixes, attention-mask mean
pooling, and L2 normalization: https://huggingface.co/intfloat/multilingual-e5-small
Setup downloads only its full-precision ONNX export and tokenizer assets. No
PyTorch pickle, remote Python code, paid inference, or generated vectors are used.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence
import uuid

import numpy as np

from .retrieval import normalize_vectors
from .store import Store, canonical_json, content_hash, utc_now


MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSIONS = 384
MAX_TOKENS = 512
MODEL_FILE = "onnx/model.onnx"
MODEL_FILES = (
    MODEL_FILE, "tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json", "README.md",
)
PREFIXES = {"query": "query: ", "passage": "passage: "}
MANIFEST_NAME = "local-model-manifest.json"
# Published SHA-256 values from the official repository's file pages.
PINNED_HASHES = {
    MODEL_FILE: "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
}


class LocalEmbeddingError(RuntimeError):
    """Local model assets or inputs cannot support a valid embedding."""


def _validate_revision(revision: str) -> None:
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise LocalEmbeddingError("local model revision must be a complete 40-character commit SHA, never 'main'")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest[key] for key in (
        "schema_version", "repository", "revision", "backend", "dimensions", "max_tokens", "prefixes", "pooling", "files",
    )}


def _validate_model_settings(target: Path) -> None:
    try:
        config = json.loads((target / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads((target / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalEmbeddingError("local model configuration cannot be read") from exc
    if not isinstance(config, dict) or not isinstance(tokenizer_config, dict):
        raise LocalEmbeddingError("local model configuration must be a JSON object")
    if config.get("hidden_size") != DIMENSIONS or config.get("max_position_embeddings") != MAX_TOKENS:
        raise LocalEmbeddingError("local model architecture does not match the registered E5 dimensions/context")
    if tokenizer_config.get("model_max_length") != MAX_TOKENS:
        raise LocalEmbeddingError("tokenizer context differs from the registered E5 context")


def validate_model_directory(target: str | Path, revision: str = DEFAULT_REVISION) -> dict[str, Any]:
    """Check the pinned manifest and file bytes, without opening any network."""
    _validate_revision(revision)
    target = Path(target)
    manifest_path = target / MANIFEST_NAME
    if not manifest_path.is_file():
        raise LocalEmbeddingError("local model is not set up; run the explicit model-download command first")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1, "repository": MODEL_ID, "revision": revision,
            "backend": "onnxruntime-cpu-fp32", "dimensions": DIMENSIONS, "max_tokens": MAX_TOKENS,
            "prefixes": PREFIXES, "pooling": "attention_mask_mean_then_l2_v1",
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise LocalEmbeddingError("local model manifest disagrees with the requested model contract")
        if set(manifest["files"]) != set(MODEL_FILES):
            raise LocalEmbeddingError("local model manifest has an unexpected file set")
        if content_hash(_identity(manifest)) != manifest["model_fingerprint"]:
            raise LocalEmbeddingError("local model manifest fingerprint is invalid")
        for filename in MODEL_FILES:
            path = target / filename
            if not path.is_file() or _hash_file(path) != manifest["files"][filename]:
                raise LocalEmbeddingError(f"local model file is missing or corrupt: {filename}")
            if revision == DEFAULT_REVISION and filename in PINNED_HASHES and manifest["files"][filename] != PINNED_HASHES[filename]:
                raise LocalEmbeddingError(f"local model file disagrees with the published pin: {filename}")
        _validate_model_settings(target)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise LocalEmbeddingError("local model metadata cannot be validated") from exc
    return manifest


def download_model(target: str | Path, revision: str = DEFAULT_REVISION) -> dict[str, Any]:
    """Explicit setup action: download a public, commit-pinned model bundle.

    This is the only function in this module that may contact the network. The
    completed manifest is published last; interrupted downloads remain unusable
    until setup resumes successfully. Existing completed bundles are immutable.
    """
    _validate_revision(revision)
    target = Path(target)
    if (target / MANIFEST_NAME).exists():
        return validate_model_directory(target, revision)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise LocalEmbeddingError("install the project's local-model extra before downloading the model") from None
    target.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for filename in MODEL_FILES:
        hf_hub_download(
            repo_id=MODEL_ID, filename=filename, revision=revision,
            local_dir=str(target), token=False,
        )
        path = target / filename
        digest = _hash_file(path)
        if revision == DEFAULT_REVISION and filename in PINNED_HASHES and digest != PINNED_HASHES[filename]:
            raise LocalEmbeddingError(f"downloaded file does not match its published checksum: {filename}")
        file_hashes[filename] = digest
    _validate_model_settings(target)
    manifest = {
        "schema_version": 1, "repository": MODEL_ID, "revision": revision,
        "backend": "onnxruntime-cpu-fp32", "dimensions": DIMENSIONS, "max_tokens": MAX_TOKENS,
        "prefixes": PREFIXES, "pooling": "attention_mask_mean_then_l2_v1", "files": file_hashes,
        "license": "MIT", "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{revision}",
        "created_at": utc_now(),
    }
    manifest["model_fingerprint"] = content_hash(_identity(manifest))
    temporary = target / f".{MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    os.replace(temporary, target / MANIFEST_NAME)
    return validate_model_directory(target, revision)


def local_embedding_cache_key(
    text: str, manifest: dict[str, Any], representation_version: str, kind: str,
) -> str:
    if kind not in PREFIXES:
        raise LocalEmbeddingError("E5 input kind must be 'query' or 'passage'")
    return "local-embedding:" + content_hash({
        "schema": 1, "provider": "local", "model": MODEL_ID, "revision": manifest["revision"],
        "dimensions": DIMENSIONS, "model_fingerprint": manifest["model_fingerprint"],
        "tokenizer_hash": manifest["files"]["tokenizer.json"], "pooling": manifest["pooling"],
        "representation_version": representation_version, "kind": kind, "prefix": PREFIXES[kind],
        "submitted_text_hash": hashlib.sha256((PREFIXES[kind] + text).encode("utf-8")).hexdigest(),
    })


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average nonpadding token vectors, including attended special tokens."""
    hidden = np.asarray(last_hidden_state, dtype=np.float32)
    mask = np.asarray(attention_mask)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or hidden.shape[2] == 0:
        raise LocalEmbeddingError("ONNX output and attention mask are not aligned")
    if not np.all(np.isfinite(hidden)) or not np.all((mask == 0) | (mask == 1)):
        raise LocalEmbeddingError("ONNX output/mask contains invalid values")
    lengths = mask.sum(axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise LocalEmbeddingError("cannot pool an input without attended tokens")
    masked = np.where(mask[:, :, None].astype(bool), hidden, np.float32(0))
    return np.asarray(masked.sum(axis=1, dtype=np.float32) / lengths, dtype=np.float32)


class LocalEmbeddingClient:
    """Local E5 with persisted vectors and an ONNX session loaded only on misses."""

    def __init__(
        self, store: Store, model_dir: str | Path, revision: str = DEFAULT_REVISION,
        *, batch_size: int = 16, intra_op_threads: int = 4,
    ):
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise LocalEmbeddingError("local batch_size must be an integer in 1..64")
        if type(intra_op_threads) is not int or intra_op_threads < 1:
            raise LocalEmbeddingError("intra_op_threads must be a positive integer")
        self.store, self.model_dir, self.revision = store, Path(model_dir), revision
        self.manifest = validate_model_directory(self.model_dir, revision)
        self.model_id, self.dimensions = MODEL_ID, DIMENSIONS
        self.batch_size, self.intra_op_threads = batch_size, intra_op_threads
        self._tokenizer = None
        self._session = None
        self._pad_id: int | None = None
        self.last_run: dict[str, Any] = {}

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ImportError:
                raise LocalEmbeddingError("install the project's local-model extra for local tokenization") from None
            tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            special_tokens = json.loads((self.model_dir / "special_tokens_map.json").read_text(encoding="utf-8"))
            pad_token = special_tokens.get("pad_token")
            if isinstance(pad_token, dict):
                pad_token = pad_token.get("content")
            pad_id = tokenizer.token_to_id(pad_token) if isinstance(pad_token, str) else None
            if pad_id is None:
                raise LocalEmbeddingError("tokenizer has no declared padding token")
            # E5's model config and tokenizer use different pad IDs. Follow the
            # tokenizer's actual token ID, matching the model-card AutoTokenizer.
            self._tokenizer, self._pad_id = tokenizer, pad_id
        return self._tokenizer

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise LocalEmbeddingError("install the project's local-model extra for local ONNX inference") from None
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.intra_op_threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.model_dir / MODEL_FILE), sess_options=options, providers=["CPUExecutionProvider"],
            )
        return self._session

    def count_tokens(self, text: str, kind: str = "query") -> int:
        if kind not in PREFIXES or not isinstance(text, str) or not text.strip():
            raise LocalEmbeddingError("E5 requires nonblank text and query/passage kind")
        return len(self._get_tokenizer().encode(PREFIXES[kind] + text, add_special_tokens=True).ids)

    def _infer(self, encodings: Sequence[Any]) -> np.ndarray:
        batch_size, length = len(encodings), max(len(encoding.ids) for encoding in encodings)
        input_ids = np.full((batch_size, length), self._pad_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, length), dtype=np.int64)
        token_type_ids = np.zeros((batch_size, length), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            count = len(encoding.ids)
            input_ids[row, :count] = encoding.ids
            attention_mask[row, :count] = encoding.attention_mask
            token_type_ids[row, :count] = encoding.type_ids
        available = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        session = self._get_session()
        inputs = session.get_inputs()
        if any(item.name not in available or item.type != "tensor(int64)" for item in inputs):
            raise LocalEmbeddingError("ONNX model exposes unexpected input names/types")
        if not {"input_ids", "attention_mask"}.issubset({item.name for item in inputs}):
            raise LocalEmbeddingError("ONNX model lacks required token IDs/attention mask")
        outputs = session.get_outputs()
        output_names = [item.name for item in outputs]
        if "last_hidden_state" not in output_names:
            raise LocalEmbeddingError("ONNX model does not expose token-level last_hidden_state")
        hidden = session.run(["last_hidden_state"], {item.name: available[item.name] for item in inputs})[0]
        if np.asarray(hidden).shape != (batch_size, length, DIMENSIONS):
            raise LocalEmbeddingError("ONNX output dimensions differ from the pinned E5 model")
        return normalize_vectors(mean_pool(hidden, attention_mask), expected_dimensions=DIMENSIONS)

    def embed(
        self, texts: Sequence[str], representation_version: str = "query-v1",
        kind: str = "query", *, bypass_cache: bool = False,
    ) -> np.ndarray:
        if kind not in PREFIXES:
            raise LocalEmbeddingError("E5 input kind must be 'query' or 'passage'")
        if isinstance(texts, str) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise LocalEmbeddingError("embedding inputs must be a sequence of nonblank strings")
        if not isinstance(representation_version, str) or not representation_version.strip():
            raise LocalEmbeddingError("representation_version must not be blank")
        started = time.perf_counter()
        self.last_run = {
            "model": MODEL_ID, "revision": self.revision, "backend": "onnxruntime-cpu-fp32",
            "kind": kind, "inputs": len(texts), "unique_inputs": 0, "cache_hits": 0,
            "computed_vectors": 0, "batches": 0, "input_tokens": 0, "api_requests": 0,
            "api_cost_usd": 0, "elapsed_seconds": 0.0,
        }
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        keys = [local_embedding_cache_key(text, self.manifest, representation_version, kind) for text in texts]
        unique = dict(zip(keys, texts))
        self.last_run["unique_inputs"] = len(unique)
        found: dict[str, np.ndarray] = {}
        missing: list[tuple[str, Any]] = []
        for key, text in unique.items():
            vector = None if bypass_cache else self.store.get_embedding(key, DIMENSIONS)
            if vector is not None:
                if not np.isclose(np.linalg.norm(vector.astype(np.float64)), 1.0, rtol=1e-5):
                    raise LocalEmbeddingError("cached local embedding is not unit normalized")
                found[key] = vector
                self.last_run["cache_hits"] += 1
                continue
            encoding = self._get_tokenizer().encode(PREFIXES[kind] + text, add_special_tokens=True)
            count = len(encoding.ids)
            if count > MAX_TOKENS:
                raise LocalEmbeddingError(f"E5 input has {count} tokens including prefix/special tokens; cap is {MAX_TOKENS}; no truncation")
            if count < 1:
                raise LocalEmbeddingError("tokenizer produced no tokens")
            missing.append((key, encoding))
            self.last_run["input_tokens"] += count
        # Every miss is preflighted before inference. Stable length ordering
        # reduces padding work without changing the caller's output order.
        missing.sort(key=lambda item: len(item[1].ids))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors = self._infer([encoding for _, encoding in batch])
            for (key, _), vector in zip(batch, vectors):
                self.store.put_embedding(key, vector, {
                    "provider": "local", "model": MODEL_ID, "revision": self.revision,
                    "model_fingerprint": self.manifest["model_fingerprint"], "kind": kind,
                    "representation_version": representation_version, "created_at": utc_now(),
                })
                found[key] = vector
            self.last_run["computed_vectors"] += len(batch)
            self.last_run["batches"] += 1
        self.last_run["elapsed_seconds"] = time.perf_counter() - started
        return np.ascontiguousarray(np.stack([found[key] for key in keys]), dtype=np.float32)
