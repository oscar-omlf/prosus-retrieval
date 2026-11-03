"""Shared persistence, fingerprints and local inference construction."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any
from .config import Config
from .store import Store
from .representation import prepared_catalog_from_rows

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

def expected_space(config, family):
    if family != "local":
        raise ValueError("Only the local E5 embedding backend is part of this application")
    model=config["local"]
    return model["model"], model["dimensions"], model["revision"]

def make_embedder(config, store, family, count=None, *, allow_api=False):
    expected_space(config, family)
    return local_client(config, store)

def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def open_store(config: Config) -> Store:
    budget = config["budget"]
    return Store(config.resolve("state"),
                 bucket_limits={key: value for key, value in budget.items() if key != "global_micro_usd"},
                 global_limit=budget["global_micro_usd"])


def local_client(config: Config, store: Store):
    from .local_embeddings import LocalEmbeddingClient
    settings = config["local"]
    return LocalEmbeddingClient(store, config.resolve("model_dir"), revision=settings["revision"],
                                batch_size=settings["batch_size"], intra_op_threads=settings["threads"])


def restored_prepared(release):
    return prepared_catalog_from_rows(release.items, source_hash=release.manifest["source_hash"],
                                     prepared_hash=release.manifest["prepared_hash"],
                                     representation_version=release.manifest["representation_version"])
