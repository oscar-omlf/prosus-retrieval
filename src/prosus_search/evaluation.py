"""Blinded pooling and explicitly automated relevance estimates.

This module never calls a model. Missing judgments stay missing, all systems use
the same pool, and final-set release is a separate, recorded decision.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SEED = 20261102
INITIAL_RUNS = ("lexical", "semantic-small", "semantic-large", "hybrid-small", "hybrid-large")
GAINS = (0, 1, 3, 7)


class EvaluationError(ValueError):
    """An evaluation cannot be compared under its declared protocol."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_split(path: str | Path, *, query_source_hash: str | None = None) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = manifest.get("manifest_hash")
    content = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if digest != canonical_hash(content):
        raise EvaluationError("Split manifest checksum mismatch")
    if query_source_hash and manifest["query_source_sha256"].lower() != query_source_hash.lower():
        raise EvaluationError("Query snapshot differs from the frozen split")
    dev, final = manifest["development"], manifest["final"]
    if len(dev) != 30 or len(final) != 70 or len(set(dev + final)) != 100:
        raise EvaluationError("Frozen split must contain 30 unique development and 70 final IDs")
    groups = manifest["groups"]
    member_ids = [qid for group in groups for qid in group["query_ids"]]
    if sorted(member_ids) != sorted(dev + final):
        raise EvaluationError("Every query must belong to exactly one intent group")
    for group in groups:
        if set(group["query_ids"]) & set(dev) and set(group["query_ids"]) & set(final):
            raise EvaluationError("An intent group crosses development/final membership")
    panels = manifest["panels"]
    if len(panels["development"]) != 4 or not set(panels["development"]) <= set(dev):
        raise EvaluationError("Expected four development diagnostic queries")
    if len(panels["secondary"]) != 2 or not set(panels["secondary"]) <= set(panels["development"]):
        raise EvaluationError("Secondary panel must contain two development panel queries")
    if len(panels["final"]) != 2 or not set(panels["final"]) <= set(final):
        raise EvaluationError("Expected two final stability queries")
    return manifest


def group_mapping(split: Mapping[str, Any]) -> dict[str, str]:
    return {qid: group["group_id"] for group in split["groups"] for qid in group["query_ids"]}


def validate_rankings(rankings: Sequence[Mapping[str, Any]], query_ids: Sequence[str],
                      runs: Sequence[str], *, k: int = 5) -> dict[str, dict[str, list[dict]]]:
    if k != 5:
        raise EvaluationError("The frozen relevance comparison evaluates exactly top five")
    expected_q, expected_runs = set(query_ids), set(runs)
    grouped: dict[str, dict[str, list[dict]]] = {run: defaultdict(list) for run in runs}
    for row in rankings:
        if row["query_id"] in expected_q and row["run_id"] in expected_runs and row["rank"] <= k:
            grouped[row["run_id"]][row["query_id"]].append(dict(row))
    for run in runs:
        for qid in query_ids:
            rows = sorted(grouped[run][qid], key=lambda row: row["rank"])
            if [row["rank"] for row in rows] != list(range(1, k + 1)):
                raise EvaluationError(f"Expected ranks 1..{k}: {run}/{qid}")
            if len({row["item_id"] for row in rows}) != k:
                raise EvaluationError(f"Duplicate top-five item: {run}/{qid}")
            grouped[run][qid] = rows
    return grouped


def build_pool(rankings: Sequence[Mapping[str, Any]], query_ids: Sequence[str], *,
               runs: Sequence[str] | None = None, seed: int = SEED,
               namespace: str = "prosus-pool-v1") -> dict:
    """Union top-five pairs; opaque IDs do not disclose system, score, or rank.

    The returned manifest is private evaluator bookkeeping, NOT a model prompt.
    Only judge.make_judgment_payload constructs the blinded model view.
    """
    run_ids = sorted(runs if runs is not None else {row["run_id"] for row in rankings})
    if not query_ids or not run_ids or len(query_ids) != len(set(query_ids)):
        raise EvaluationError("Pool needs distinct query IDs and at least one run")
    grouped = validate_rankings(rankings, query_ids, run_ids)
    pairs: dict[tuple[str, str], dict] = {}
    for run in run_ids:
        for qid in query_ids:
            for row in grouped[run][qid]:
                key = (qid, row["item_id"])
                if key not in pairs:
                    opaque = "a_" + canonical_hash([namespace, *key])[:24]
                    pairs[key] = {"assessment_id": opaque, "query_id": qid,
                                  "item_id": row["item_id"], "contributing_runs": []}
                pairs[key]["contributing_runs"].append(run)
    ordered = sorted(pairs.values(), key=lambda pair: canonical_hash([seed, pair["assessment_id"]]))
    manifest = {"version": "pool-v1", "query_ids": list(query_ids), "runs": run_ids,
                "seed": seed, "pairs": ordered,
                "rankings_hash": canonical_hash({run: dict(grouped[run]) for run in run_ids})}
    manifest["pool_hash"] = canonical_hash(manifest)
    return manifest


def _valid_judgment(value: Mapping[str, Any] | None, assessment_id: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    # JudgeClient results wrap a validated payload; tests and imported labels may
    # supply the already validated judgment itself.
    if "status" in value:
        if value["status"] != "valid":
            return None
        value = value.get("value")
    if not isinstance(value, Mapping) or value.get("assessment_id") != assessment_id:
        return None
    grade = value.get("grade")
    if isinstance(grade, bool) or not isinstance(grade, int) or grade not in range(4):
        return None
    return value


def _card_value(card: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not card:
        return None
    if "status" in card:
        return card.get("value") if card["status"] == "valid" else None
    return card


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _aggregate(per_query: Mapping[str, Mapping[str, Any]], query_ids: Sequence[str]) -> dict | None:
    if not query_ids:
        return None
    rows = [per_query[qid] for qid in query_ids]
    denom = sum(row["constraint_denominator"] for row in rows)
    contradicted = sum(row["contradiction_count"] for row in rows)
    unknown = sum(row["unknown_count"] for row in rows)
    return {"query_count": len(rows), "ndcg_at_5": _mean([row["ndcg_at_5"] for row in rows]),
            "success_at_5": _mean([row["success_at_5"] for row in rows]),
            "mrr_at_5": _mean([row["mrr_at_5"] for row in rows]),
            "contradiction_count": contradicted, "unknown_count": unknown,
            "constraint_denominator": denom,
            "contradiction_rate": contradicted / denom if denom else None,
            "unknown_rate": unknown / denom if denom else None,
            "zero_idcg_query_count": sum(row["zero_idcg"] for row in rows)}


def evaluate_runs(rankings: Sequence[Mapping[str, Any]], pool: Mapping[str, Any],
                  judgments: Mapping[str, Mapping[str, Any]],
                  cards: Mapping[str, Mapping[str, Any]], *,
                  groups: Mapping[str, str] | None = None, bootstrap_replicates: int = 2000,
                  seed: int = SEED) -> dict:
    """Evaluate the common complete-query subset; never turn missing into zero."""
    query_ids, runs = pool["query_ids"], pool["runs"]
    grouped = validate_rankings(rankings, query_ids, runs)
    expected_rank_hash = canonical_hash({run: dict(grouped[run]) for run in runs})
    if expected_rank_hash != pool["rankings_hash"]:
        raise EvaluationError("Rankings changed after pooling")
    expected_pool_hash = canonical_hash({key: val for key, val in pool.items() if key != "pool_hash"})
    if pool.get("pool_hash") != expected_pool_hash:
        raise EvaluationError("Pool checksum mismatch")
    by_query: dict[str, list[dict]] = defaultdict(list)
    valid: dict[tuple[str, str], Mapping[str, Any]] = {}
    missing = []
    for pair in pool["pairs"]:
        by_query[pair["query_id"]].append(pair)
        judgment = _valid_judgment(judgments.get(pair["assessment_id"]), pair["assessment_id"])
        if judgment is None:
            missing.append(pair["assessment_id"])
        else:
            valid[(pair["query_id"], pair["item_id"])] = judgment
    complete = [qid for qid in query_ids if _card_value(cards.get(qid)) is not None
                and all((qid, pair["item_id"]) in valid for pair in by_query[qid])]
    missing_cards = [qid for qid in query_ids if _card_value(cards.get(qid)) is None]
    all_complete = len(complete) == len(query_ids)
    result = {"origin": "llm", "calibration_status": "not_performed", "human_metrics": None,
              "label": "LLM-judged, pool-relative relevance estimates", "pool_hash": pool["pool_hash"],
              "complete": all_complete, "query_count": len(query_ids), "complete_query_ids": complete,
              "incomplete_query_ids": [qid for qid in query_ids if qid not in complete],
              "pair_count": len(pool["pairs"]), "valid_pair_count": len(valid),
              "missing_assessment_ids": missing, "missing_card_query_ids": missing_cards,
              "coverage": len(complete) / len(query_ids),
              "limitations": ["Automated source-based estimates; no human calibration or objective ground truth.",
                              "Intervals condition on automated labels, not judge accuracy.",
                              "Top-five pools do not measure full-catalog recall.",
                              "Incomplete-query subsets can have selection bias; no promotion unless complete."],
              "systems": {}, "comparisons": {}}
    for run in runs:
        per_query = {}
        for qid in complete:
            card = _card_value(cards[qid])
            mandatory = {req["id"] for req in card["requirements"] if req["kind"] == "mandatory"}
            labels = [valid[(qid, row["item_id"])] for row in grouped[run][qid]]
            grades = [label["grade"] for label in labels]
            ideal = sorted((valid[(qid, pair["item_id"])]["grade"] for pair in by_query[qid]), reverse=True)[:5]
            dcg = sum(GAINS[grade] / math.log2(i + 2) for i, grade in enumerate(grades))
            idcg = sum(GAINS[grade] / math.log2(i + 2) for i, grade in enumerate(ideal))
            useful_ranks = [i + 1 for i, grade in enumerate(grades) if grade >= 2]
            statuses = [{req["id"]: req["status"] for req in label["requirements"]} for label in labels]
            if any(set(status) != {req["id"] for req in card["requirements"]} for status in statuses):
                raise EvaluationError("Judgment requirement IDs do not match the supplied intent card")
            per_query[qid] = {
                "ndcg_at_5": dcg / idcg if idcg else 0.0,
                "success_at_5": int(bool(useful_ranks)),
                "mrr_at_5": 1 / useful_ranks[0] if useful_ranks else 0.0,
                "zero_idcg": idcg == 0,
                "contradiction_count": sum(any(status[rid] == "contradito" for rid in mandatory) for status in statuses),
                "unknown_count": sum(any(status[rid] in ("não informado", "não aplicável") for rid in mandatory) for status in statuses),
                "constraint_denominator": 5 if mandatory else 0,
            }
        subset = _aggregate(per_query, complete)
        result["systems"][run] = {"full_split": subset if all_complete else None,
                                   "common_complete_subset": subset, "per_query": per_query}
    if complete:
        groups = groups or {qid: qid for qid in complete}
        for i, incumbent in enumerate(runs):
            for candidate in runs[i + 1:]:
                values = {qid: result["systems"][candidate]["per_query"][qid]["ndcg_at_5"]
                          - result["systems"][incumbent]["per_query"][qid]["ndcg_at_5"] for qid in complete}
                result["comparisons"][f"{candidate} - {incumbent}"] = paired_group_bootstrap(
                    values, groups, replicates=bootstrap_replicates, seed=seed)
    return result


def paired_group_bootstrap(deltas: Mapping[str, float], groups: Mapping[str, str], *,
                           replicates: int = 2000, seed: int = SEED) -> dict:
    if not deltas or replicates < 1 or any(qid not in groups for qid in deltas):
        raise EvaluationError("Bootstrap needs nonempty paired deltas and complete group membership")
    members: dict[str, list[float]] = defaultdict(list)
    for qid in sorted(deltas):
        if not math.isfinite(deltas[qid]):
            raise EvaluationError("Bootstrap deltas must be finite")
        members[groups[qid]].append(deltas[qid])
    values = [members[group] for group in sorted(members)]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(replicates):
        sampled = [number for index in rng.integers(0, len(values), len(values)) for number in values[index]]
        means.append(float(np.mean(sampled)))
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_delta": float(np.mean(list(deltas.values()))), "ci95": [float(low), float(high)],
            "group_count": len(values), "query_count": len(deltas), "replicates": replicates,
            "seed": seed, "wins": sum(value > 0 for value in deltas.values()),
            "losses": sum(value < 0 for value in deltas.values()), "ties": sum(value == 0 for value in deltas.values()),
            "interpretation": "Query-group variation conditional on LLM judgments; not human agreement."}


def compare_sensitivity(primary: Mapping[str, Any], alternate: Mapping[str, Any], *,
                         primary_judgments: Mapping[str, Mapping[str, Any]] | None = None,
                         alternate_judgments: Mapping[str, Mapping[str, Any]] | None = None,
                         requirements_comparable: bool = True) -> dict:
    """Compare systems on exactly the same predeclared complete panel."""
    if set(primary["systems"]) != set(alternate["systems"]):
        raise EvaluationError("Sensitivity arms must compare identical systems")
    complete = bool(primary["complete"] and alternate["complete"] and
                    primary["complete_query_ids"] == alternate["complete_query_ids"])
    report = {"complete": complete, "query_ids": primary["complete_query_ids"] if complete else [],
              "requirements_comparable": requirements_comparable, "comparisons": {}, "agreement": None}
    if not complete:
        return report
    runs = sorted(primary["systems"])
    for index, a in enumerate(runs):
        for b in runs[index + 1:]:
            p = primary["systems"][b]["full_split"]["ndcg_at_5"] - primary["systems"][a]["full_split"]["ndcg_at_5"]
            alt = alternate["systems"][b]["full_split"]["ndcg_at_5"] - alternate["systems"][a]["full_split"]["ndcg_at_5"]
            report["comparisons"][f"{b} - {a}"] = {"primary_delta": p, "alternate_delta": alt,
                "delta_swing": abs(p - alt), "material_instability": p * alt < 0 and abs(p - alt) >= 0.02}
    if primary_judgments is not None and alternate_judgments is not None:
        pairs = []
        for aid in sorted(set(primary_judgments) & set(alternate_judgments)):
            p, alt = _valid_judgment(primary_judgments[aid], aid), _valid_judgment(alternate_judgments[aid], aid)
            if p is not None and alt is not None:
                pairs.append((p, alt))
        if pairs:
            req_compared = req_disagreed = 0
            if requirements_comparable:
                for p, alt in pairs:
                    ps = {row["id"]: row["status"] for row in p["requirements"]}
                    ats = {row["id"]: row["status"] for row in alt["requirements"]}
                    for rid in set(ps) & set(ats):
                        req_compared += 1
                        req_disagreed += ps[rid] != ats[rid]
            report["agreement"] = {"pair_count": len(pairs),
                "exact_grade_agreement": sum(p["grade"] == a["grade"] for p, a in pairs) / len(pairs),
                "mean_absolute_grade_difference": sum(abs(p["grade"] - a["grade"]) for p, a in pairs) / len(pairs),
                "useful_threshold_flip_rate": sum((p["grade"] >= 2) != (a["grade"] >= 2) for p, a in pairs) / len(pairs),
                "requirement_disagreement_rate": req_disagreed / req_compared if req_compared else None,
                "requirement_comparison_count": req_compared}
    return report


def _comparison(report: Mapping[str, Any], candidate: str, incumbent: str) -> dict:
    forward = report["comparisons"].get(f"{candidate} - {incumbent}")
    if forward:
        return dict(forward)
    backward = report["comparisons"].get(f"{incumbent} - {candidate}")
    if not backward:
        raise EvaluationError(f"Missing comparison: {candidate} versus {incumbent}")
    result = dict(backward)
    result["mean_delta"] *= -1
    result["ci95"] = [-backward["ci95"][1], -backward["ci95"][0]]
    result["wins"], result["losses"] = backward["losses"], backward["wins"]
    return result


def select_configuration(report: Mapping[str, Any], checks: Mapping[str, Mapping[str, Any]], *,
                         incumbent: str = "semantic-small", semantic_incumbent: str = "semantic-small",
                         hybrid_incumbent: str = "hybrid-small") -> dict:
    """Apply the frozen initial comparison policy; do not infer missing checks."""
    required_checks = {"repeat", "prompt", "model", "intent_card"}
    required_runs = set(INITIAL_RUNS)
    selection = {"serving": incumbent, "semantic_finalist": semantic_incumbent,
                 "hybrid_finalist": hybrid_incumbent, "promoted": False,
                 "origin": "llm", "calibration_status": "not_performed", "reasons": [],
                 "final_evaluation_released": False}
    if not report.get("complete") or not required_runs <= set(report.get("systems", {})):
        selection["reasons"].append("Incomplete development comparison; retain the previously accepted incumbent.")
        return selection
    if not required_checks <= set(checks) or not all(checks[name].get("complete") for name in required_checks):
        selection["reasons"].append("Required sensitivity checks incomplete; retain the previously accepted incumbent.")
        return selection

    def stable(a: str, b: str) -> bool:
        for check in checks.values():
            comp = check["comparisons"].get(f"{a} - {b}", check["comparisons"].get(f"{b} - {a}"))
            if comp is None or comp["material_instability"]:
                return False
        return True

    def no_contradiction_regression(a: str, b: str) -> bool:
        av = report["systems"][a]["full_split"]["contradiction_rate"]
        bv = report["systems"][b]["full_split"]["contradiction_rate"]
        return av is None and bv is None or av is not None and bv is not None and av <= bv

    def eligible(a: str, b: str) -> bool:
        return stable(a, b) and no_contradiction_regression(a, b)

    for family, current in (("semantic", semantic_incumbent), ("hybrid", hybrid_incumbent)):
        challenger = f"{family}-large" if current.endswith("small") else f"{family}-small"
        if eligible(challenger, current) and _comparison(report, challenger, current)["ci95"][0] > 0:
            selection[f"{family}_finalist"] = challenger
    semantic, hybrid = selection["semantic_finalist"], selection["hybrid_finalist"]
    sem_candidates = [semantic] + [run for run in ("semantic-small", "semantic-large")
                                   if run != semantic and eligible(run, semantic)]
    strongest = max(sem_candidates, key=lambda run: (report["systems"][run]["full_split"]["ndcg_at_5"], run == semantic))
    advantage = _comparison(report, hybrid, strongest)
    if eligible(hybrid, semantic) and stable(hybrid, strongest) and advantage["mean_delta"] >= 0.02 and advantage["ci95"][0] > 0:
        selection["serving"] = hybrid
        selection["reasons"].append("Hybrid meets the frozen gain, uncertainty, contradiction and stability rules.")
    else:
        selection["serving"] = semantic
        selection["reasons"].append("Hybrid promotion threshold not met; retain the selected semantic approach.")
    selection["promoted"] = selection["serving"] != incumbent
    selection["strongest_eligible_semantic_comparator"] = strongest
    selection["same_model_fusion"] = {size: _comparison(report, f"hybrid-{size}", f"semantic-{size}") for size in ("small", "large")}
    selection["evidence_statement"] = "Provisional LLM preference under this protocol; not human-validated superiority."
    return selection


def assert_final_release(selection: Mapping[str, Any], *, release_final: bool,
                         round4_complete: bool = False, iteration_declined: bool = False) -> None:
    """Procedural guard; this is deliberately not described as a security boundary."""
    if not release_final or not (round4_complete or iteration_declined):
        raise EvaluationError("Final judging stays locked until Round 4 ends or the user declines iteration")
    required = ("semantic_finalist", "hybrid_finalist", "retrieval_hash", "judge_hash", "split_hash", "coverage_hash")
    if any(not selection.get(key) for key in required):
        raise EvaluationError("Final release requires frozen finalists, retrieval, judge, split and coverage hashes")
    if not selection.get("selection_frozen"):
        raise EvaluationError("Freeze selection before releasing final judgments")


def card_differences(primary: Mapping[str, Any], alternate: Mapping[str, Any]) -> dict:
    """Report interpretation changes without pretending requirement IDs align."""
    def clauses(card: Mapping[str, Any]) -> set[tuple[str, str, str]]:
        return {(row["quote"], row["kind"], row["interpretation"]) for row in card["requirements"]}
    p, a = clauses(primary), clauses(alternate)
    return {"primary_only": [list(row) for row in sorted(p - a)],
            "alternate_only": [list(row) for row in sorted(a - p)],
            "mandatory_clauses_changed": {(q, text) for q, kind, text in p if kind == "mandatory"}
                                         != {(q, text) for q, kind, text in a if kind == "mandatory"},
            "requirement_denominators_directly_comparable": p == a}
