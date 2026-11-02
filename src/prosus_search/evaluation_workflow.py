"""Offline orchestration of the frozen evaluation; search never imports this.

Every actual judgment still passes through JudgeClient and the persistent ledger.
An unavailable key, card or judgment produces explicit incompleteness, not a score.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence
import unicodedata

from .evaluation import (
    INITIAL_RUNS, SEED, EvaluationError, assert_final_release, build_pool,
    canonical_hash, card_differences, compare_sensitivity, evaluate_runs,
    group_mapping, select_configuration,
)
from .judge import JudgeClient, JudgeConfig, JudgePreflightError, evidence_view, protocol_hash
from .store import BudgetExceededError


FROZEN_SETTINGS = {"primary_model": "gpt-6-luna", "secondary_model": "gpt-6-sol",
                   "input_token_cap": 3072, "output_token_cap": 512,
                   "bootstrap_replicates": 2000, "seed": SEED}
TERMINAL_PROVIDER_STOPS = {"authentication_or_access", "quota_exhausted", "usage_missing",
                          "provider_token_cap_overrun", "provider_request_failed", "transient_provider_failure"}


def _query_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text", value.get("query", value.get("original_query")))
    else:
        text = getattr(value, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise EvaluationError("Query mapping must contain original nonblank text")
    return text


def _card(result: Mapping[str, Any] | None) -> dict | None:
    return result.get("value") if result and result.get("status") == "valid" else None


def _missing(reason: str) -> dict:
    return {"status": "missing", "value": None, "reason": reason,
            "origin": "llm", "calibration_status": "not_performed"}


def query_requests_price(query: str) -> bool:
    """Identify explicit price/currency wording without inventing a threshold.

    Broad quality, health or portion aspirations do not qualify. Numeric values
    alone are not prices; currency semantics remain unknown even when exposed.
    """
    text = "".join(char for char in unicodedata.normalize("NFKD", query.casefold())
                   if not unicodedata.combining(char))
    return bool(re.search(r"\b(?:precos?|custos?|custa|custam|barat[oa]s?|economic[oa]s?|reais|euros?|dolares?)\b|[$€£]|\bvalor\s+(?:(?:de|ate)\s+)?\d", text))


def run_evaluation(*, rankings: Sequence[Mapping[str, Any]], queries: Mapping[str, Any],
                   metadata: Mapping[str, Mapping[str, Any]], split: Mapping[str, Any],
                   store: Any, token_counter: Callable[[str], int], allow_api: bool = False,
                   phase: str = "development", selection: Mapping[str, Any] | None = None,
                   release_final: bool = False, round4_complete: bool = False,
                   iteration_declined: bool = False,
                   evaluation_config: Mapping[str, Any] | None = None,
                   primary_provider: Any = None, secondary_provider: Any = None) -> dict:
    """Run/replay development or explicitly released final evaluation.

    ``queries`` maps IDs to original strings (or Query objects); ``metadata`` maps
    item IDs to the immutable release's original metadata. Providers are injectable
    solely for offline mechanics tests. Default execution makes no network calls.
    Local/nonfrozen run sets produce an explicitly unjudged operational report;
    they cannot accidentally trigger or win the five-run hosted comparison.
    """
    phase = "development" if phase == "dev" else phase
    if phase not in {"development", "final"}:
        raise EvaluationError("Evaluation phase must be development or final")
    supplied = dict(evaluation_config or {})
    if set(supplied) - set(FROZEN_SETTINGS):
        raise EvaluationError("Unknown evaluation settings; no configuration is silently ignored")
    settings = {**FROZEN_SETTINGS, **supplied}
    if settings != FROZEN_SETTINGS:
        raise EvaluationError("Evaluation settings differ from the frozen protocol; record a versioned amendment first")
    split_body = {key: value for key, value in split.items() if key != "manifest_hash"}
    if canonical_hash(split_body) != split.get("manifest_hash"):
        raise EvaluationError("Split checksum mismatch")
    dev, final = list(split["development"]), list(split["final"])
    if len(dev) != 30 or len(final) != 70 or len(set(dev + final)) != 100:
        raise EvaluationError("Expected the frozen 30/70 development/final split")
    groups = group_mapping(split)
    if set(groups) != set(dev + final) or set(groups[q] for q in dev) & set(groups[q] for q in final):
        raise EvaluationError("Intent groups must cover the split without crossing it")
    texts = {qid: _query_text(value) for qid, value in queries.items()}
    if set(texts) != set(dev + final):
        raise EvaluationError("Query mapping must match all 100 frozen query IDs")
    primary_config, secondary_config = JudgeConfig(), JudgeConfig.secondary()
    combined_judge_hash = canonical_hash({"primary": protocol_hash(primary_config),
                                         "secondary": protocol_hash(secondary_config), "settings": settings,
                                         "workflow_version": "evaluation-workflow-v1",
                                         "price_evidence_policy": "explicit_price_words_or_currency_v1"})
    sorted_rows = sorted((dict(row) for row in rankings), key=lambda row: (row["run_id"], row["query_id"], row["rank"]))
    hashes = {"retrieval_hash": canonical_hash(sorted_rows), "judge_hash": combined_judge_hash,
              "split_hash": split["manifest_hash"], "query_hash": canonical_hash(texts),
              "evidence_hash": canonical_hash(metadata)}
    runs_present = sorted({row["run_id"] for row in rankings})
    frozen_comparison = set(INITIAL_RUNS) == set(runs_present)
    if set(INITIAL_RUNS) < set(runs_present):
        raise EvaluationError("Extra runs require an explicitly expanded pool and budget protocol; no run is silently omitted")
    authorization = {"selection": dict(selection or {}), "release_final": release_final,
                     "round4_complete": round4_complete, "iteration_declined": iteration_declined}
    if phase == "final":
        assert_final_release(selection or {}, release_final=release_final,
                             round4_complete=round4_complete, iteration_declined=iteration_declined)
        for name, expected in hashes.items():
            if (selection or {}).get(name) != expected:
                raise EvaluationError(f"Final release {name} differs from the frozen selection")
        runs = ["lexical", selection["semantic_finalist"], selection["hybrid_finalist"]]
        query_ids = final
    else:
        runs = list(INITIAL_RUNS) if frozen_comparison else runs_present
        query_ids = dev
    if not runs:
        raise EvaluationError("No ranking runs supplied")
    pool = build_pool(rankings, query_ids, runs=runs, seed=settings["seed"])
    if any(pair["item_id"] not in metadata for pair in pool["pairs"]):
        raise EvaluationError("A pooled item is absent from the immutable evidence snapshot")
    if phase == "final":
        if selection.get("coverage_hash") != pool["pool_hash"]:
            raise EvaluationError("Final query/pool coverage differs from the frozen selection")
        # A single stored record prevents a second final release under changed
        # hashes. Resumption under the same frozen record remains allowed.
        final_lock = {**hashes, "coverage_hash": pool["pool_hash"], "runs": runs}
        def check_final_lock() -> None:
            previous = store.lookup("evaluation:final-release-v1")
            if previous is not None and previous != final_lock:
                raise EvaluationError("Final evaluation has already started under a different frozen selection")
            if allow_api and previous is None:
                store.put("evaluation:final-release-v1", final_lock)
        if allow_api:
            with store.api_writer():
                check_final_lock()
        else:
            check_final_lock()
    panel_ids = list(split["panels"]["development" if phase == "development" else "final"])
    secondary_ids = list(split["panels"]["secondary"]) if phase == "development" else []
    if not set(panel_ids) <= set(query_ids) or not set(secondary_ids) <= set(panel_ids):
        raise EvaluationError("Diagnostic panel membership crosses the declared split")
    panel_pool = build_pool(rankings, panel_ids, runs=runs)
    secondary_pool = build_pool(rankings, secondary_ids, runs=runs) if secondary_ids else None
    panel_pairs, secondary_pairs = len(panel_pool["pairs"]), len(secondary_pool["pairs"]) if secondary_pool else 0
    if len(pool["pairs"]) > (750 if phase == "development" else 1050) or panel_pairs > (100 if phase == "development" else 30) or secondary_pairs > 50:
        raise EvaluationError("Pooled workload exceeds the frozen initial experiment bounds")
    primary_calls = len(query_ids) + len(pool["pairs"]) + panel_pairs
    secondary_calls = 0
    if phase == "development":
        primary_calls += panel_pairs + secondary_pairs
        secondary_calls = len(secondary_ids) + secondary_pairs
    planned_micro = primary_calls * primary_config.reservation_micro_usd + secondary_calls * secondary_config.reservation_micro_usd
    bucket = "development" if phase == "development" else "final"
    bucket_limit = store.summary()["buckets"][bucket]["limit"]
    if planned_micro > bucket_limit:
        raise BudgetExceededError("Complete primary and diagnostic workload does not fit its protected budget bucket")
    preflight = {"primary_calls_before_retries": primary_calls, "secondary_calls_before_retries": secondary_calls,
                 "upper_cost_micro_usd_before_retries": planned_micro, "budget_bucket": bucket,
                 "bucket_limit_micro_usd": bucket_limit, "input_token_cap": 3072, "output_token_cap": 512,
                 "protected_final_micro_usd": store.summary()["buckets"]["final"]["limit"],
                 "policy": "Before each uncached task, preserve the full remaining planned workload plus one possible retry; each attempt still reserves separately."}
    primary = JudgeClient(store, config=primary_config, allow_api=allow_api, provider=primary_provider,
                          token_counter=token_counter, final_authorization=authorization)
    secondary = JudgeClient(store, config=secondary_config, allow_api=allow_api, provider=secondary_provider,
                            token_counter=token_counter, final_authorization=authorization)
    cards, judgments = {}, {}
    check_labels: dict[str, dict] = {"repeat": {}, "prompt": {}, "model": {}, "intent_card": {}}
    alternate_cards: dict[str, dict] = {}
    progress_key = "evaluation-progress:" + canonical_hash({**hashes, "pool": pool["pool_hash"], "phase": phase})
    progress = store.lookup(progress_key) or {"completed_caps": {}}
    stop_reason = None

    def task(task_id: str, evaluator: JudgeClient, invoke: Callable[[], dict]) -> dict:
        nonlocal stop_reason
        if stop_reason:
            return _missing(stop_reason)
        was_allowed = evaluator.allow_api
        evaluator.allow_api = False
        try:
            cached = invoke()
        finally:
            evaluator.allow_api = was_allowed
        terminal = cached.get("status") == "valid" or cached.get("retry_allowed") is False
        if terminal:
            progress["completed_caps"][task_id] = evaluator.config.reservation_micro_usd
            if allow_api:
                store.put(progress_key, progress)
            if cached.get("reason") in TERMINAL_PROVIDER_STOPS:
                stop_reason = cached["reason"]
            return cached
        if not allow_api:
            return cached
        remaining_planned = planned_micro - sum(progress["completed_caps"].values())
        available = store.summary()["buckets"][bucket]["remaining"]
        required = max(0, remaining_planned) + evaluator.config.reservation_micro_usd
        if available < required:
            stop_reason = "insufficient_budget_for_remaining_workload_and_retry"
            return _missing(stop_reason)
        result = invoke()
        progress["completed_caps"][task_id] = evaluator.config.reservation_micro_usd
        store.put(progress_key, progress)
        if result.get("reason") in TERMINAL_PROVIDER_STOPS:
            stop_reason = result["reason"]
        return result

    def make_primary_card(qid: str) -> dict:
        return task(f"card:{qid}", primary, lambda: primary.make_card(qid, texts[qid], bucket=bucket))

    def judge_pairs(pairs: Sequence[Mapping[str, Any]], destination: dict, *, evaluator: JudgeClient = primary,
                    arm: str = "primary", arm_cards: Mapping[str, Any] | None = None) -> None:
        nonlocal stop_reason
        card_source = cards if arm_cards is None else arm_cards
        for pair in pairs:
            aid, qid = pair["assessment_id"], pair["query_id"]
            intent = _card(card_source.get(qid))
            if intent is None:
                destination[aid] = _missing("intent_card_unavailable")
                continue
            result = task(f"{arm}:{aid}", evaluator, lambda pair=pair, intent=intent: evaluator.judge_pair(
                pair["assessment_id"], texts[pair["query_id"]], intent,
                evidence_view(metadata[pair["item_id"]], include_price=query_requests_price(texts[pair["query_id"]])),
                replicate_id=arm, alternate_prompt=arm == "prompt", bucket=bucket))
            destination[aid] = result
            if phase == "development" and arm == "primary" and qid in split["pilot"]:
                if any(attempt.get("validation_error") for attempt in result.get("attempts", [])):
                    stop_reason = "mechanical_pilot_structural_failure_requires_versioned_review"
                    break

    # Missing hosted comparators are a declared incomplete experiment. No local
    # retrieval run is silently substituted for them or sent to the hosted judge.
    if phase == "development" and not frozen_comparison:
        if allow_api:
            raise EvaluationError("Frozen hosted comparison is incomplete; local runs cannot trigger its paid evaluation")
        cards = {qid: _missing("frozen_hosted_comparators_unavailable") for qid in query_ids}
        judgments = {pair["assessment_id"]: _missing("not_judged") for pair in pool["pairs"]}
        stop_reason = "frozen_hosted_comparators_unavailable"
    else:
        # The pilot stops on structural failures before the main workload. Every
        # card remains query-only and is generated before its candidate judgments.
        ordered_queries = list(split["pilot"]) + [qid for qid in query_ids if qid not in split["pilot"]] if phase == "development" else query_ids
        for qid in ordered_queries:
            cards[qid] = make_primary_card(qid)
            if phase == "development" and qid in split["pilot"] and any(
                    attempt.get("validation_error") for attempt in cards[qid].get("attempts", [])):
                stop_reason = "mechanical_pilot_structural_failure_requires_versioned_review"
            judge_pairs([pair for pair in pool["pairs"] if pair["query_id"] == qid], judgments)
            if stop_reason:
                break
    for qid in query_ids:
        cards.setdefault(qid, _missing(stop_reason or "not_completed"))
    for pair in pool["pairs"]:
        judgments.setdefault(pair["assessment_id"], _missing(stop_reason or "not_completed"))
    report = evaluate_runs(rankings, pool, judgments, cards, groups=groups,
                           bootstrap_replicates=settings["bootstrap_replicates"], seed=settings["seed"])
    checks = {}
    if report["complete"] and not stop_reason:
        judge_pairs(panel_pool["pairs"], check_labels["repeat"], arm="repeat")
        if phase == "development":
            judge_pairs(panel_pool["pairs"], check_labels["prompt"], arm="prompt")
            judge_pairs(secondary_pool["pairs"], check_labels["model"], evaluator=secondary, arm="model")
            for qid in secondary_ids:
                alternate_cards[qid] = task(f"independent-card:{qid}", secondary, lambda qid=qid: secondary.make_card(
                    qid, texts[qid], replicate_id="independent-secondary-card", bucket=bucket))
            judge_pairs(secondary_pool["pairs"], check_labels["intent_card"], arm="intent_card", arm_cards=alternate_cards)
    arm_names = ("repeat", "prompt", "model", "intent_card") if phase == "development" else ("repeat",)
    for arm in arm_names:
        arm_pool = secondary_pool if arm in {"model", "intent_card"} else panel_pool
        arm_cards = alternate_cards if arm == "intent_card" else cards
        baseline = evaluate_runs(rankings, arm_pool, judgments, cards, groups=groups)
        alternate = evaluate_runs(rankings, arm_pool, check_labels[arm], arm_cards, groups=groups)
        differences = {qid: card_differences(_card(cards[qid]), _card(alternate_cards[qid]))
                       for qid in secondary_ids if _card(cards.get(qid)) and _card(alternate_cards.get(qid))} if arm == "intent_card" else {}
        comparable = all(item["requirement_denominators_directly_comparable"] for item in differences.values()) if differences else arm != "intent_card"
        aids = {pair["assessment_id"] for pair in arm_pool["pairs"]}
        checks[arm] = compare_sensitivity(baseline, alternate,
            primary_judgments={aid: judgments[aid] for aid in aids if aid in judgments},
            alternate_judgments={aid: check_labels[arm][aid] for aid in aids if aid in check_labels[arm]},
            requirements_comparable=comparable)
        checks[arm].update(primary_metrics=baseline, alternate_metrics=alternate, card_differences=differences)
    if phase == "development" and not frozen_comparison:
        chosen = {"serving": None, "semantic_finalist": None, "hybrid_finalist": None,
                  "promoted": False, "selection_frozen": False, "status": "unjudged_local_default",
                  "protocol_incumbent": "semantic-small", "local_selection_performed": False,
                  "coverage_hash": None, "development_pool_hash": pool["pool_hash"], **hashes,
                  "reasons": ["The local search default is operational, not selected by hosted relevance evidence; the frozen hosted comparison remains pending."]}
    elif phase == "development":
        chosen = select_configuration(report, checks,
            incumbent=(selection or {}).get("serving", "semantic-small"),
            semantic_incumbent=(selection or {}).get("semantic_finalist", "semantic-small"),
            hybrid_incumbent=(selection or {}).get("hybrid_finalist", "hybrid-small"))
        chosen.update(hashes)
        if frozen_comparison:
            final_pool = build_pool(rankings, final, runs=["lexical", chosen["semantic_finalist"], chosen["hybrid_finalist"]])
            chosen["coverage_hash"] = final_pool["pool_hash"]
        else:
            chosen["coverage_hash"] = None
        chosen["selection_frozen"] = False
        chosen["development_pool_hash"] = pool["pool_hash"]
    else:
        chosen = dict(selection)
        chosen["final_evaluation_released"] = True
        chosen["reselection_performed"] = False
    complete = report["complete"] and all(check["complete"] for check in checks.values())
    return {"status": "complete" if complete else "incomplete", "phase": phase,
            "comparison_status": "frozen_hosted_comparison" if frozen_comparison else "unjudged_local_or_nonfrozen_runs",
            "origin": "llm", "calibration_status": "not_performed", "human_metrics": None,
            "stop_reason": stop_reason, "preflight": preflight, "hashes": hashes,
            "pool": pool, "cards": cards, "judgments": judgments,
            "diagnostic_judgments": check_labels, "independent_cards": alternate_cards,
            "report": report, "checks": checks, "selection": chosen,
            "cost": store.summary(), "pilot_query_ids": list(split["pilot"]) if phase == "development" else [],
            "limitations": report["limitations"]}
