"""Immutable, checksummed search releases and one atomically updated pointer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from .store import canonical_json, content_hash, utc_now


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class Release:
    release_id: str
    path: Path
    manifest: dict[str, Any]
    items: tuple[dict[str, Any], ...]
    ids: tuple[str, ...]
    vectors: np.ndarray


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ArtifactError(f"Cannot read valid JSON artifact: {path.name}") from None


def _packages() -> dict[str, str]:
    result = {}
    for package in ("numpy", "openai", "tiktoken", "bm25s", "onnxruntime", "tokenizers", "huggingface-hub", "sentence-transformers", "torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def _code_hash() -> str:
    directory = Path(__file__).parent
    return content_hash({path.name: file_hash(path) for path in sorted(directory.glob("*.py"))})


def _item_rows(prepared: Any) -> list[dict[str, Any]]:
    rows = []
    for item in prepared.items:
        record = item.record
        rows.append({
            "item_id": record.item_id, "merchant_id": record.merchant_id,
            "metadata": record.metadata, "source_ids": list(record.source_ids),
            "source_lines": list(record.source_lines), "text": item.text,
            "text_hash": item.text_hash, "token_count": item.token_count,
            "lexical_tokens": list(item.lexical_tokens),
        })
    ids = [row["item_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ArtifactError("Prepared item IDs must be nonempty, unique, and sorted")
    for row in rows:
        if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["text_hash"]:
            raise ArtifactError("Prepared text hash does not match the exact retrieval text")
    return rows


def _validate_matrix(vectors: Any, count: int, dimensions: int, normalize: bool) -> np.ndarray:
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise ArtifactError("Release vectors are not numeric") from None
    if matrix.shape != (count, dimensions) or dimensions <= 0 or not np.isfinite(matrix).all():
        raise ArtifactError("Release vector shape/finite-value check failed")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ArtifactError("Release vectors must be nonzero and finite")
    if normalize:
        return np.ascontiguousarray(matrix / norms[:, None], dtype=np.float32)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ArtifactError("Release vectors are not L2-normalized")
    return matrix


def publish_release(root: str | Path, prepared: Any, vectors: Any, model: str, dimensions: int,
                    epoch: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Write a validated snapshot without changing the current-release pointer.

    The caller publishes all requested models before publish_current(). A failed
    build can leave a .staging-* directory, but can never become current.
    """
    if not model or not epoch:
        raise ArtifactError("Model and explicit cache epoch are required")
    rows = _item_rows(prepared)
    ids = [row["item_id"] for row in rows]
    matrix = _validate_matrix(vectors, len(ids), dimensions, normalize=True)
    releases = Path(root) / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    staging = releases / (".staging-" + uuid.uuid4().hex)
    staging.mkdir()
    with (staging / "items.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _write_json(staging / "ids.json", ids)
    with (staging / "vectors.npy").open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    files = {name: file_hash(staging / name) for name in ("items.jsonl", "ids.json", "vectors.npy")}
    identity = {
        "schema_version": 1, "source_hash": prepared.source_hash,
        "prepared_hash": prepared.prepared_hash, "representation_version": prepared.representation_version,
        "model": model, "dimensions": dimensions, "epoch": epoch,
        "count": len(ids), "dtype": "float32", "normalized": True,
        "ordered_ids_hash": content_hash(ids), "files": files,
        "package_versions": _packages(), "code_hash": _code_hash(), "metadata": dict(metadata or {}),
    }
    release_id = content_hash(identity)
    manifest = identity | {"release_id": release_id, "created_at": utc_now()}
    _write_json(staging / "manifest.json", manifest)
    # Validate the actual on-disk bytes before making a completed release visible.
    _load_directory(staging, expected_id=release_id)
    destination = releases / release_id
    if destination.exists():
        load_release(root, release_id)
        # Own small staging files only. Never remove a loaded release or recurse.
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
    else:
        staging.rename(destination)
    return release_id


def _load_directory(path: Path, expected_id: str) -> Release:
    manifest = _read_json(path / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ArtifactError("Unsupported release manifest")
    if (type(manifest.get("count")) is not int or manifest["count"] <= 0
            or type(manifest.get("dimensions")) is not int or manifest["dimensions"] <= 0
            or manifest.get("dtype") != "float32" or manifest.get("normalized") is not True
            or not all(isinstance(manifest.get(key), str) and manifest[key]
                       for key in ("model", "epoch", "source_hash", "prepared_hash", "representation_version"))):
        raise ArtifactError("Release manifest has invalid types or model-space metadata")
    identity = {key: value for key, value in manifest.items() if key not in ("release_id", "created_at")}
    if manifest.get("release_id") != expected_id or content_hash(identity) != expected_id:
        raise ArtifactError("Release identity/manifest checksum mismatch")
    expected_files = {"items.jsonl", "ids.json", "vectors.npy"}
    if set(manifest.get("files", {})) != expected_files:
        raise ArtifactError("Release file manifest is incomplete")
    for name in expected_files:
        try:
            digest = file_hash(path / name)
        except OSError:
            raise ArtifactError(f"Missing release artifact: {name}") from None
        if digest != manifest["files"][name]:
            raise ArtifactError(f"Release checksum mismatch: {name}")
    ids = _read_json(path / "ids.json")
    if not isinstance(ids, list) or not all(isinstance(item, str) and item for item in ids):
        raise ArtifactError("Release IDs must be nonblank strings")
    if not ids or ids != sorted(set(ids)) or content_hash(ids) != manifest.get("ordered_ids_hash"):
        raise ArtifactError("Release IDs are not unique, ordered, or checksum-aligned")
    try:
        items = tuple(json.loads(line) for line in (path / "items.jsonl").read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ArtifactError("Release item metadata is invalid") from None
    if (len(items) != len(ids) or not all(isinstance(item, dict) for item in items)
            or [item.get("item_id") for item in items] != ids or manifest.get("count") != len(ids)):
        raise ArtifactError("Release metadata/ID/vector row alignment failed")
    for item in items:
        if not isinstance(item.get("text"), str) or hashlib.sha256(item["text"].encode("utf-8")).hexdigest() != item.get("text_hash"):
            raise ArtifactError("Release retrieval-text checksum mismatch")
    try:
        matrix = np.load(path / "vectors.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        raise ArtifactError("Release vector array cannot be read safely") from None
    if matrix.dtype != np.dtype("float32"):
        raise ArtifactError("Release vectors must be float32")
    _validate_matrix(matrix, len(ids), manifest.get("dimensions", 0), normalize=False)
    return Release(expected_id, path, manifest, items, tuple(ids), matrix)


def load_release(root: str | Path, release_id: str, expected_model: str | None = None,
                 expected_dimensions: int | None = None, expected_epoch: str | None = None) -> Release:
    if not isinstance(release_id, str) or len(release_id) != 64 or any(ch not in "0123456789abcdef" for ch in release_id):
        raise ArtifactError("Invalid release ID; explicit content hashes are required")
    release = _load_directory(Path(root) / "releases" / release_id, release_id)
    for field, expected in (("model", expected_model), ("dimensions", expected_dimensions), ("epoch", expected_epoch)):
        if expected is not None and release.manifest[field] != expected:
            raise ArtifactError(f"Release belongs to a different {field}")
    return release


def publish_current(root: str | Path, model_release_ids: Mapping[str, str], default_mode: str = "semantic-small") -> dict[str, Any]:
    """Atomically advance a complete compatible bundle; never overwrite matrices."""
    if not model_release_ids or default_mode not in model_release_ids:
        raise ArtifactError("Current bundle must contain its default search mode")
    releases = {key: load_release(root, release_id) for key, release_id in model_release_ids.items()}
    signatures = {(release.manifest["source_hash"], release.manifest["prepared_hash"], release.manifest["representation_version"],
                   release.manifest["ordered_ids_hash"]) for release in releases.values()}
    if len(signatures) != 1:
        raise ArtifactError("Cannot publish models built from different prepared catalogs")
    first = next(iter(releases.values()))
    pointer = {"schema_version": 1, "models": dict(model_release_ids), "default_mode": default_mode,
               "prepared_hash": first.manifest["prepared_hash"], "source_hash": first.manifest["source_hash"],
               "published_at": utc_now()}
    root = Path(root)
    temporary = root / (".current-" + uuid.uuid4().hex + ".json")
    _write_json(temporary, pointer)
    os.replace(temporary, root / "current.json")
    return pointer


def load_current(root: str | Path) -> dict[str, Any]:
    pointer = _read_json(Path(root) / "current.json")
    if not isinstance(pointer, dict) or pointer.get("schema_version") != 1 or not isinstance(pointer.get("models"), dict):
        raise ArtifactError("Current release pointer is invalid")
    if pointer.get("default_mode") not in pointer["models"]:
        raise ArtifactError("Current release pointer lacks its default mode")
    return pointer
