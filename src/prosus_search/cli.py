"""Thin CLI; model calls always require explicit --allow-api."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .benchmark import (build, load_benchmark, open_store, preflight, read_json,
                        run_benchmark, write_csv, write_json, expected_space)
from .artifacts import file_hash, load_release
from .catalog import load_queries
from .config import load_config, token_counter
from .evaluation import build_pool, evaluate_runs, group_mapping, load_split
from .service import SearchService


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="food-search", description="Portuguese semantic search and reproducible benchmarks")
    result.add_argument("--config", default="config.toml", help="configuration path (paths resolve beside it)")
    commands = result.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="explicitly download tokenizer assets and optionally the local model")
    setup.add_argument("--local-model", action="store_true", help="download pinned E5 ONNX weights (~490 MB)")
    commands.add_parser("preflight", help="validate inputs, token limits, frozen split and estimated costs")
    make = commands.add_parser("build", help="prepare and publish immutable catalog releases")
    make.add_argument("--backend", choices=("local", "hosted", "comparison"), default="local")
    make.add_argument("--allow-api", action="store_true")
    bench = commands.add_parser("benchmark", help="rank all 100 queries from an explicit build manifest")
    bench.add_argument("--build", required=True, help="immutable build JSON returned by build")
    bench.add_argument("--allow-api", action="store_true")
    query = commands.add_parser("search", help="return the top k items for a new natural-language query")
    query.add_argument("query")
    query.add_argument("--k", type=int, default=5)
    query.add_argument("--mode", choices=("lexical", "semantic-local", "hybrid-local", "grounded-local", "grounded-rerank", "grounded-facets", "semantic-small", "hybrid-small", "semantic-large", "hybrid-large"))
    query.add_argument("--release", help="explicit release ID instead of current pointer")
    query.add_argument("--allow-api", action="store_true")
    query.add_argument("--json", action="store_true", help="include IDs, scores, source excerpts and timings")
    judge = commands.add_parser("evaluate", help="blinded development judging; final work needs its separate release gate")
    judge.add_argument("--benchmark", required=True, help="benchmark.json path")
    judge.add_argument("--allow-api", action="store_true")
    judge.add_argument("--pilot-only", action="store_true", help="stop after the five frozen development pilot queries")
    judge.add_argument("--judge-workers", type=int, default=1, help="1..4 independent judge requests; pilots remain serial")
    judge.add_argument("--dry-run", action="store_true", help="pool and report missing labels without model calls")
    judge.add_argument("--phase", choices=("development", "final"), default="development")
    judge.add_argument("--selection", help="frozen selection JSON required for final evaluation")
    judge.add_argument("--release-final", action="store_true")
    judge.add_argument("--round4-complete", action="store_true")
    judge.add_argument("--iteration-declined", action="store_true")
    judge.add_argument("--output", help="output JSON path; defaults beside benchmark")
    export = commands.add_parser("export", help="write ranked IDs and scores without redistributing source descriptions")
    export.add_argument("--benchmark", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--run", help="one run only; omit to export all")
    commands.add_parser("budget", help="show persistent confirmed/pending/uncertain spending")
    return result


def _emit(value) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def _evaluate(config, args):
    benchmark, rankings = load_benchmark(args.benchmark)
    split = load_split(config.resolve("split"), query_source_hash=file_hash(config.resolve("queries")))
    if benchmark["split_hash"] != split["manifest_hash"]:
        raise ValueError("Benchmark split differs from the frozen evaluation split")
    queries = {query.query_id: query.text for query in load_queries(config.resolve("queries"))}
    metadata = None
    prepared_hash = None
    source_hash = None
    for mode, release_id in benchmark["releases"].items():
        model, dimensions, epoch = expected_space(config, mode.removeprefix("semantic-"))
        release = load_release(config.resolve("artifacts"), release_id, model, dimensions, epoch)
        if prepared_hash and prepared_hash != release.manifest["prepared_hash"]:
            raise ValueError("Evaluation releases do not share the same catalog evidence")
        if source_hash and source_hash != release.manifest["source_hash"]:
            raise ValueError("Evaluation releases do not share the same source snapshot")
        prepared_hash = release.manifest["prepared_hash"]
        source_hash = release.manifest["source_hash"]
        metadata = {row["item_id"]: row["metadata"] for row in release.items}
    if metadata is None or any(row["item_id"] not in metadata for row in rankings):
        raise ValueError("Evaluation rankings contain IDs absent from immutable catalog evidence")
    output = Path(args.output) if args.output else Path(args.benchmark).parent / f"evaluation-{args.phase}.json"
    if args.phase == "final":
        from .evaluation import assert_final_release
        if not args.selection:
            raise ValueError("Final evaluation requires --selection and an explicit end-of-iteration decision")
        selection = read_json(args.selection)
        assert_final_release(selection, release_final=args.release_final,
                             round4_complete=args.round4_complete, iteration_declined=args.iteration_declined)
    else:
        selection = None
    if args.dry_run:
        run_ids = benchmark["runs"] if args.phase == "development" else [
            "lexical", selection["semantic_finalist"], selection["hybrid_finalist"]]
        pool = build_pool(rankings, split[args.phase], runs=run_ids)
        report = evaluate_runs(rankings, pool, {}, {}, groups=group_mapping(split))
        result = {"status": "unjudged", "phase": args.phase, "api_calls_made": 0,
                  "pool": pool, "metrics": report, "selection": None}
    else:
        from .evaluation_workflow import run_evaluation
        result = run_evaluation(rankings=rankings, queries=queries, metadata=metadata, split=split,
                                store=open_store(config), token_counter=token_counter(config),
                                allow_api=args.allow_api, phase=args.phase, selection=selection,
                                release_final=args.release_final, round4_complete=args.round4_complete,
                                iteration_declined=args.iteration_declined,
                                evaluation_config=config["evaluation"], base_url=config["api"]["base_url"],
                                pilot_only=args.pilot_only, judge_workers=args.judge_workers)
    write_json(output, result)
    return {"evaluation_path": str(output), "status": result.get("status"),
            "phase": args.phase, "budget": open_store(config).summary()}


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "setup":
            token_counter(config, allow_download=True)
            result = {"tokenizer": "cl100k_base cached", "api_calls_made": 0}
            if args.local_model:
                from .local_embeddings import download_model
                result["local_model"] = download_model(config.resolve("model_dir"), config["local"]["revision"])
            _emit(result)
        elif args.command == "preflight":
            _emit(preflight(config))
        elif args.command == "build":
            result = build(config, backend=args.backend, allow_api=args.allow_api)
            _emit({"build_manifest": result["build_manifest"], "releases": result["manifest"]["releases"],
                   "elapsed_seconds": result["manifest"]["elapsed_seconds"], "budget": result["manifest"]["budget"]})
        elif args.command == "benchmark":
            result = run_benchmark(config, args.build, allow_api=args.allow_api)
            _emit({"output_directory": result["output_directory"], "runs": result["report"]["runs"],
                   "queries": result["report"]["query_count"], "rows": result["report"]["ranking_rows"]})
        elif args.command == "search":
            service = SearchService(config, mode=args.mode, release_id=args.release, allow_api=args.allow_api)
            result = service.search(args.query, args.k)
            if args.json:
                _emit(result)
            else:
                print(f"{result['mode']} | top {len(result['results'])} | scores are ranking signals, not probabilities")
                for row in result["results"]:
                    print(f"{row['rank']:>2}. {row['name']}  [{row['score_kind']}: {row['score']:.6f}]")
                    print(f"    {row['item_id']}")
        elif args.command == "evaluate":
            _emit(_evaluate(config, args))
        elif args.command == "export":
            report, rankings = load_benchmark(args.benchmark)
            if args.run and args.run not in report["runs"]:
                raise ValueError("Requested run is absent from this benchmark")
            rows = [row for row in rankings if not args.run or row["run_id"] == args.run]
            write_csv(Path(args.output), rows, public=True)
            _emit({"output": args.output, "rows": len(rows), "source_descriptions_included": False})
        elif args.command == "budget":
            _emit(open_store(config).summary())
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        # Providers sanitize exceptions at their boundary; never print SDK objects.
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
