"""Preparation, release builds and auditable benchmark exports (no judging)."""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import platform
import time
import uuid
from typing import Any, Mapping

import numpy as np

from .artifacts import file_hash, load_release, publish_current, publish_release
from .catalog import load_catalog, load_queries
from .config import Config, ConfigError, token_counter
from .embeddings import EmbeddingClient, EmbeddingConfig
from .evaluation import load_split, validate_rankings
from .representation import prepare_catalog, prepare_query, prepared_catalog_from_rows
from .retrieval import SearchEngine
from .store import Store, content_hash, utc_now


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


def load_inputs(config: Config, *, verify_snapshot: bool = True):
    catalog_path, queries_path = config.resolve("catalog"), config.resolve("queries")
    if verify_snapshot:
        for path, expected in ((catalog_path, config["data"]["catalog_sha256"]),
                               (queries_path, config["data"]["queries_sha256"])):
            if file_hash(path) != expected.lower():
                raise ConfigError(f"Dataset hash differs: {path.name}; record a new dataset version before comparing")
    count = token_counter(config)
    catalog = load_catalog(catalog_path)
    rep = config["representation"]
    prepared = prepare_catalog(catalog, count, rep["item_token_cap"], rep["version"])
    queries = load_queries(queries_path)
    for query in queries:
        prepare_query(query.text, count, rep["query_token_cap"])
    return catalog, prepared, queries, count


def preflight(config: Config) -> dict:
    catalog, prepared, queries, count = load_inputs(config)
    split = load_split(config.resolve("split"), query_source_hash=file_hash(config.resolve("queries")))
    tokens = [item.token_count for item in prepared.items]
    unique = {item.text_hash: item.token_count for item in prepared.items}
    query_tokens = sum(count(prepare_query(query.text, count)) for query in queries)
    return {
        "raw_rows": catalog.raw_count, "canonical_items": len(prepared.items),
        "distinct_retrieval_texts": len(unique), "queries": len(queries),
        "catalog_sha256": catalog.source_hash, "prepared_hash": prepared.prepared_hash,
        "query_sha256": file_hash(config.resolve("queries")), "split_hash": split["manifest_hash"],
        "representation_version": prepared.representation_version,
        "item_tokens": {"total": sum(tokens), "unique_text_total": sum(unique.values()),
                        "max": max(tokens), "median": float(np.median(tokens)),
                        "p95": float(np.percentile(tokens, 95))},
        "query_tokens": query_tokens,
        "hosted_input_cost_estimate_usd_before_rounding_retries": {
            name: (sum(unique.values()) + query_tokens) * float(model["input_rate_per_million"]) / 1_000_000
            for name, model in config["models"].items()},
        "api_calls_made": 0,
    }


def local_client(config: Config, store: Store):
    from .local_embeddings import LocalEmbeddingClient
    settings = config["local"]
    return LocalEmbeddingClient(store, config.resolve("model_dir"), revision=settings["revision"],
                                batch_size=settings["batch_size"], intra_op_threads=settings["threads"])


def expected_space(config: Config, family: str) -> tuple[str, int, str]:
    if family == "local":
        model = config["local"]
        return model["model"], model["dimensions"], model["revision"]
    if family not in config["models"]:
        raise ConfigError(f"Unknown model family: {family}")
    model = config["models"][family]
    return model["model"], model["dimensions"], model["epoch"]


def make_embedder(config: Config, store: Store, family: str, count, *, allow_api: bool = False):
    if family == "local":
        return local_client(config, store)
    return EmbeddingClient(store, EmbeddingConfig(**config["models"][family]), count, allow_api=allow_api)


def build(config: Config, *, backend: str = "local", allow_api: bool = False) -> dict:
    if backend not in {"local", "hosted"}:
        raise ConfigError("Build backend must be local or hosted")
    started = time.perf_counter()
    audit = preflight(config)
    _, prepared, queries, count = load_inputs(config)
    store = open_store(config)
    families = ("local",) if backend == "local" else ("small", "large")
    release_ids, models = {}, {}
    query_texts = [prepare_query(query.text, count, config["representation"]["query_token_cap"]) for query in queries]
    for family in families:
        model_start = time.perf_counter()
        embedder = make_embedder(config, store, family, count, allow_api=allow_api)
        texts = [item.text for item in prepared.items]
        if family == "local":
            # E5 has a distinct tokenizer and required prefixes; preflight both
            # roles before producing any vectors. Catalog evidence is identical.
            local_counts = [embedder.count_tokens(text, kind="passage") for text in texts]
            local_query_counts = [embedder.count_tokens(text, kind="query") for text in query_texts]
            if max(local_counts + local_query_counts) > 512:
                raise ConfigError("Local model input exceeds 512 tokens; no truncation or partial build")
            vectors = embedder.embed(texts, representation_version=prepared.representation_version, kind="passage")
            catalog_stats = dict(embedder.last_run)
            embedder.embed(query_texts, representation_version="query-v1", kind="query")
        else:
            vectors = embedder.embed(texts, representation_version=prepared.representation_version)
            catalog_stats = dict(embedder.last_run)
            embedder.embed(query_texts, representation_version="query-v1")
        model, dimensions, epoch = expected_space(config, family)
        rid = publish_release(config.resolve("artifacts"), prepared, vectors, model, dimensions, epoch,
                              metadata={"config_hash": config.digest, "backend": backend,
                                        "retrieval": config["retrieval"], "query_sha256": audit["query_sha256"]})
        release_ids[f"semantic-{family}"] = rid
        models[family] = {"release_id": rid, "model": model, "dimensions": dimensions,
                          "catalog_embedding": catalog_stats, "query_embedding": dict(embedder.last_run),
                          "elapsed_seconds": time.perf_counter() - model_start,
                          "matrix_bytes": int(vectors.nbytes)}
        del vectors, embedder
    default = "semantic-local" if backend == "local" else "semantic-small"
    # Publish only after every requested model has completed and validated.
    pointer = publish_current(config.resolve("artifacts"), release_ids, default_mode=default)
    manifest = {"schema_version": 1, "created_at": utc_now(), "backend": backend,
                "config_hash": config.digest, "config": config.values, "releases": release_ids,
                "default_mode": default, "audit": audit, "models": models,
                "elapsed_seconds": time.perf_counter() - started, "budget": store.summary()}
    manifest["build_id"] = content_hash(manifest)
    destination = config.resolve("artifacts") / "builds" / (manifest["build_id"] + ".json")
    write_json(destination, manifest)
    return {"build_manifest": str(destination), "manifest": manifest, "current": pointer}


def restored_prepared(release):
    return prepared_catalog_from_rows(release.items, source_hash=release.manifest["source_hash"],
                                     prepared_hash=release.manifest["prepared_hash"],
                                     representation_version=release.manifest["representation_version"])


def load_build(path: str | Path) -> dict:
    manifest = read_json(path)
    if manifest.get("build_id") != content_hash({k: v for k, v in manifest.items() if k != "build_id"}):
        raise ConfigError("Build manifest checksum mismatch")
    return manifest


def write_rankings(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict], *, public: bool = False) -> None:
    fields = ["run_id", "query_id", "item_id", "rank", "score", "score_kind"]
    if not public:
        fields += ["original_query", "merchant_id", "name", "description", "dense_score", "lexical_score",
                   "dense_rank", "lexical_rank", "lexical_evidence", "release_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def timing_summary(samples: list[float]) -> dict:
    return {"count": len(samples), "median_ms": float(np.median(samples) * 1000),
            "p95_ms": float(np.percentile(samples, 95) * 1000), "max_ms": max(samples) * 1000}


def run_benchmark(config: Config, build_path: str | Path, *, allow_api: bool = False) -> dict:
    manifest = load_build(build_path)
    # Explicit immutable release IDs and exact config prevent unnoticed pointer drift.
    if manifest["config_hash"] != config.digest:
        raise ConfigError("Benchmark configuration differs from build; build a versioned release first")
    queries = load_queries(config.resolve("queries"))
    if file_hash(config.resolve("queries")) != manifest["audit"]["query_sha256"]:
        raise ConfigError("Queries differ from the built benchmark snapshot")
    split = load_split(config.resolve("split"), query_source_hash=manifest["audit"]["query_sha256"])
    if split["manifest_hash"] != manifest["audit"]["split_hash"]:
        raise ConfigError("Split changed since build")
    count, store = token_counter(config), open_store(config)
    # Validate the entire requested comparison before any query cache miss can
    # spend money. A valid first model must not conceal an invalid second one.
    for mode, release_id in manifest["releases"].items():
        model, dimensions, epoch = expected_space(config, mode.removeprefix("semantic-"))
        checked = load_release(config.resolve("artifacts"), release_id, model, dimensions, epoch)
        if (checked.manifest["source_hash"] != manifest["audit"]["catalog_sha256"]
                or checked.manifest["prepared_hash"] != manifest["audit"]["prepared_hash"]):
            raise ConfigError("Release catalog differs from the declared build")
        del checked
    ranking_rows, diagnostic_rows, measurements = [], [], {}
    query_texts = [prepare_query(query.text, count, config["representation"]["query_token_cap"]) for query in queries]
    lexical_done = False
    for mode, release_id in manifest["releases"].items():
        family = mode.removeprefix("semantic-")
        load_started = time.perf_counter()
        model, dimensions, epoch = expected_space(config, family)
        release = load_release(config.resolve("artifacts"), release_id, model, dimensions, epoch)
        prepared = restored_prepared(release)
        engine = SearchEngine(prepared, release.vectors, **config["retrieval"])
        if len(prepared.items) < 5:
            raise ConfigError("Benchmark requires at least five distinct catalog items")
        load_seconds = time.perf_counter() - load_started
        embedder = make_embedder(config, store, family, count, allow_api=allow_api)
        embedding_started = time.perf_counter()
        vectors = embedder.embed(query_texts, representation_version="query-v1",
                                 **({"kind": "query"} if family == "local" else {}))
        embed_seconds = time.perf_counter() - embedding_started
        runs = [(f"semantic-{family}", "semantic"), (f"hybrid-{family}", "hybrid")]
        if not lexical_done:
            runs.insert(0, ("lexical", "lexical"))
            lexical_done = True
        for run_id, retrieval_mode in runs:
            # Exclude lazy BM25 construction from warm scoring; report it separately.
            warm_start = time.perf_counter()
            engine.search(query_texts[0], vectors[0], mode=retrieval_mode, k=5)
            warmup_seconds = time.perf_counter() - warm_start
            elapsed = []
            for index, query in enumerate(queries):
                start = time.perf_counter()
                ranked = engine.search(query_texts[index], vectors[index], mode=retrieval_mode, k=min(100, len(prepared.items)))
                elapsed.append(time.perf_counter() - start)
                for item in ranked:
                    row = asdict(item) | {"run_id": run_id, "query_id": query.query_id,
                                          "original_query": query.text, "release_id": release_id}
                    diagnostic_rows.append(row)
                    if item.rank <= 5:
                        ranking_rows.append(row)
            measurements[run_id] = {"warm_scoring_top100": timing_summary(elapsed),
                                     "query_vector_required": retrieval_mode != "lexical",
                                     "initial_load_seconds": load_seconds,
                                     "warmup_including_lazy_index_seconds": warmup_seconds,
                                     "batch_query_vector_seconds": embed_seconds,
                                     "query_vector_cache": dict(embedder.last_run)}
        del engine, vectors, embedder, release
    runs = sorted(measurements)
    validate_rankings(ranking_rows, [query.query_id for query in queries], runs)
    created_at = utc_now()
    output_id = content_hash({"build_id": manifest["build_id"], "rankings": ranking_rows,
                              "split_hash": split["manifest_hash"], "created_at": created_at})
    out = config.resolve("outputs") / output_id
    out.mkdir(parents=True, exist_ok=True)
    write_rankings(out / "rankings.jsonl", ranking_rows)
    write_rankings(out / "diagnostics-top100.jsonl", diagnostic_rows)
    write_csv(out / "rankings.csv", ranking_rows)
    selected = [row for row in ranking_rows if row["run_id"] == manifest["default_mode"]]
    write_csv(out / "selected-top5.csv", selected)
    write_csv(out / "public-rankings.csv", ranking_rows, public=True)
    report = {"schema_version": 1, "benchmark_id": output_id, "build_id": manifest["build_id"],
              "created_at": created_at, "runs": runs, "query_count": len(queries),
              "query_ids": [query.query_id for query in queries],
              "ranking_rows": len(ranking_rows), "unique_items_per_query": 5,
              "releases": manifest["releases"], "split_hash": split["manifest_hash"],
              "default_mode": manifest["default_mode"], "selection_status": "provisional_unjudged",
              "relevance_metrics": None, "human_validation": False,
              "measurements": measurements, "budget": store.summary(),
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "code_hash": content_hash({path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}),
              "files": {name: file_hash(out / name) for name in
                        ("rankings.jsonl", "diagnostics-top100.jsonl", "rankings.csv", "selected-top5.csv", "public-rankings.csv")},
              "limitations": ["Operational measurements are not relevance scores.",
                              "Cached batch query-vector time is not uncached end-to-end query latency.",
                              "No final relevance judgments generated."]}
    report["report_hash"] = content_hash(report)
    write_json(out / "benchmark.json", report)
    return {"output_directory": str(out), "report": report}


def load_benchmark(path: str | Path) -> tuple[dict, list[dict]]:
    path = Path(path)
    report = read_json(path)
    if report.get("report_hash") != content_hash({key: value for key, value in report.items() if key != "report_hash"}):
        raise ConfigError("Benchmark report checksum mismatch")
    expected_files = {"rankings.jsonl", "diagnostics-top100.jsonl", "rankings.csv", "selected-top5.csv", "public-rankings.csv"}
    if set(report.get("files", {})) != expected_files:
        raise ConfigError("Benchmark file manifest is incomplete or contains unexpected paths")
    for name, checksum in report["files"].items():
        if file_hash(path.parent / name) != checksum:
            raise ConfigError(f"Benchmark artifact checksum failed: {name}")
    rows = [json.loads(line) for line in (path.parent / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]
    query_ids = report.get("query_ids", [])
    if report.get("query_count") != 100 or len(query_ids) != 100 or len(set(query_ids)) != 100:
        raise ConfigError("Benchmark must contain all 100 distinct query IDs")
    if set(query_ids) != {row["query_id"] for row in rows} or len(rows) != 500 * len(report["runs"]):
        raise ConfigError("Benchmark query/run coverage or row count differs from manifest")
    expected_id = content_hash({"build_id": report["build_id"], "rankings": rows,
                                "split_hash": report["split_hash"], "created_at": report["created_at"]})
    if report["benchmark_id"] != expected_id:
        raise ConfigError("Benchmark identity differs from ranked outputs")
    validate_rankings(rows, query_ids, report["runs"])
    return report, rows
