"""Validated structured-model requests with caching and cumulative accounting."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import json
import time
from threading import Lock
from typing import Any, Callable, Mapping
from .runtime import canonical_hash

PROTOCOL_VERSION = "structured-model-v1"

def protocol_hash(config=None):
    return canonical_hash({"config": (config or ModelConfig()).__dict__})

class ModelValidationError(ValueError):
    """The output cannot be used as a valid structured result."""


class ModelPreflightError(ValueError):
    """The input cannot safely be sent under the frozen protocol."""


@dataclass(frozen=True)
class ModelConfig:
    model: str = "openai/gpt-4.1-mini"
    input_micro_usd_per_token: str = "0.40"
    output_micro_usd_per_token: str = "1.60"
    input_token_cap: int = 3072
    output_token_cap: int = 1024
    token_overhead: int = 128
    timeout_seconds: float = 60
    protocol_version: str = PROTOCOL_VERSION
    base_url: str = "https://api.openai.com/v1/"


    @classmethod
    def gpt41(cls, model: str, **kwargs) -> "ModelConfig":
        rates = {'openai/gpt-4.1-nano': ('0.10','0.40'),
                 'openai/gpt-4.1-mini': ('0.40','1.60'),
                 'openai/gpt-4.1': ('2','8')}
        if model not in rates:
            raise ValueError('This execution permits only the user-authorized GPT-4.1 models')
        inp, out = rates[model]
        kwargs['protocol_version']=kwargs.get('protocol_version',PROTOCOL_VERSION)+'-chat-configured-base-v2'
        return cls(model=model,input_micro_usd_per_token=inp,output_micro_usd_per_token=out,**kwargs)

    def cost(self, input_tokens: int, output_tokens: int) -> int:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        amount = Decimal(self.input_micro_usd_per_token) * input_tokens + Decimal(self.output_micro_usd_per_token) * output_tokens
        return int(amount.to_integral_value(rounding=ROUND_CEILING))

    @property
    def reservation_micro_usd(self) -> int:
        return self.cost(self.input_token_cap, self.output_token_cap)


def _payload(config: ModelConfig, instructions: str, data: dict, schema: dict, name: str) -> dict:
    payload = {"model": config.model, "instructions": instructions,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)}]}],
            "text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
            "reasoning": {"effort": "none"}, "max_output_tokens": config.output_token_cap,
            "store": False, "service_tier": "default"}
    if config.model.removeprefix('openai/').startswith('gpt-4.1'):
        payload.pop('reasoning')  # non-reasoning models do not accept reasoning.effort
    return payload


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _usage(response: Any) -> tuple[int, int] | None:
    usage = _get(response, "usage")
    inp, out = _get(usage, "input_tokens"), _get(usage, "output_tokens")
    return (inp, out) if type(inp) is int and inp >= 0 and type(out) is int and out >= 0 else None


def _visible_output(response: Any) -> str:
    if _get(response, "status") != "completed":
        raise ModelValidationError("Response not completed")
    for item in _get(response, "output", []) or []:
        for content in _get(item, "content", []) or []:
            if _get(content, "type") == "refusal":
                raise ModelValidationError("Response refused")
    value = _get(response, "output_text")
    if value is None:
        value = "".join(_get(content, "text", "") for item in _get(response, "output", []) or []
                        for content in _get(item, "content", []) or [] if _get(content, "type") == "output_text")
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError("No visible structured output")
    return value


def _parse_json(text: str) -> dict:
    def distinct(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ModelValidationError("Duplicate JSON field")
            result[key] = value
        return result
    try:
        result = json.loads(text, object_pairs_hook=distinct, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError) as exc:
        raise ModelValidationError("Invalid structured JSON") from exc
    if not isinstance(result, dict):
        raise ModelValidationError("Structured output must be an object")
    return result


def _safe_error(exc: Exception) -> tuple[str, bool, float]:
    """Never persist exception strings, headers, request bodies or SDK dumps."""
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    name = type(exc).__name__
    if status in (401, 403) or name in {"AuthenticationError", "PermissionDeniedError"}:
        return "authentication_or_access", False, 0
    if code == "insufficient_quota":
        return "quota_exhausted", False, 0
    if status == 429 or type(status) is int and status >= 500 or name in {"TimeoutError", "APITimeoutError", "APIConnectionError", "ConnectionError"}:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {})
        try:
            delay = max(0.0, float(headers.get("retry-after", 0)))
        except (ValueError, TypeError):
            delay = 0.0
        return "transient_provider_failure", delay <= 60, delay
    return "provider_request_failed", False, 0


class ModelClient:
    """Bounded attempts, exact input caching and reserve-before-request accounting.

    ``provider`` is an explicit callable accepting a request dictionary.
    ``token_counter`` counts the full serialized request; an extra allowance is
    included for transport framing. No network is used unless allow_api=True.
    """
    def __init__(self, store: Any, *, config: ModelConfig | None = None, allow_api: bool = False,
                 provider: Any = None, token_counter: Callable[[str], int] | None = None,
                 sleeper: Callable[[float], None] = time.sleep,
                 max_attempts: int = 1):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 2:
            raise ValueError("max_attempts must be 1 or 2")
        self.store = store
        self.config = config or ModelConfig()
        self.allow_api = allow_api
        self.provider = provider
        self.token_counter = token_counter
        self.sleeper = sleeper
        self.max_attempts = max_attempts
        self._provider_lock = Lock()



    def preflight(self, payload: Mapping[str, Any]) -> dict:
        if self.token_counter is None:
            raise ModelPreflightError("Provide a local tokenizer before any model API call")
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        token_count = self.token_counter(serialized) + self.config.token_overhead
        if type(token_count) is not int or token_count < 1 or token_count > self.config.input_token_cap:
            raise ModelPreflightError("Full model payload exceeds input cap; no evidence is silently truncated")
        return {"estimated_input_tokens": token_count, "reserved_input_tokens": self.config.input_token_cap,
                "output_token_cap": self.config.output_token_cap, "reservation_micro_usd": self.config.reservation_micro_usd}

    def _request(self, payload: dict) -> Any:
        if not callable(self.provider):
            raise ModelPreflightError("Configure an explicit provider before API use")
        return self.provider(payload)

    def _execute(self, payload: dict, replicate_id: str, bucket: str, purpose: str,
                 validator: Callable[[dict], dict]) -> dict:
        if not replicate_id.strip() or bucket not in {"development", "final", "round4", "recovery"}:
            raise ModelPreflightError("Invalid operation or budget bucket")
        key = "model:" + canonical_hash({"payload": payload, "replicate_id": replicate_id,
                                         "protocol": protocol_hash(self.config), "purpose": purpose})
        cached = self.store.lookup(key)
        if cached and cached.get("status") == "valid":
            validator(cached["value"])
            return {**cached, "cache_hit": True}
        if not self.allow_api:
            return {**(cached or {}), "status": "missing", "reason": "offline_cache_miss" if not cached else cached.get("reason", "cached_missing"),
                    "cache_key": key, "cache_hit": bool(cached), "origin": "llm", "calibration_status": "not_performed"}
        preflight = self.preflight(payload)
        # Missing local credentials are a proven pre-transmission failure, so no
        # reservation or logical attempt is created for that condition.
        if self.provider is None:
            raise ModelPreflightError("Configure an explicit provider before API use")
        with self.store.api_writer(scope=key):
            record = self.store.lookup(key) or {"status": "missing", "value": None, "attempts": [],
                "cache_key": key, "payload": payload, "payload_hash": canonical_hash(payload),
                "protocol_hash": protocol_hash(self.config), "replicate_id": replicate_id,
                "origin": "llm", "calibration_status": "not_performed", "preflight": preflight}
            if record.get("status") == "valid":
                validator(record["value"])
                return {**record, "cache_hit": True}
            if record.get("retry_allowed") is False:
                return {**record, "cache_hit": True}
            while len(record["attempts"]) < self.max_attempts:
                charge_id = self.store.reserve(bucket, self.config.reservation_micro_usd,
                    metadata={"purpose": purpose, "model": self.config.model, "payload_hash": record["payload_hash"],
                              "replicate_id": replicate_id, "attempt": len(record["attempts"]) + 1})
                attempt = {"request_id": charge_id, "state": "submitted_or_interrupted", "usage": None,
                           "visible_output": None, "validation_error": None}
                record["attempts"].append(attempt)
                record["retry_allowed"] = True
                self.store.put(key, record)
                started = time.perf_counter()
                try:
                    response = self._request(payload)
                except Exception as exc:
                    self.store.mark_uncertain(charge_id)
                    reason, retry_allowed, delay = _safe_error(exc)
                    status_code = getattr(exc, "status_code", None)
                    body=getattr(exc,'body',{})
                    # Fixed diagnostic categories only, never provider text.
                    message=str(body.get('message',body.get('error',{})) if isinstance(body,dict) else '').lower()
                    signals=[s for s in ('model','deployment','route','permission','not found','not_found','resource','api version') if s in message]
                    attempt.update(state="uncertain", elapsed_seconds=time.perf_counter() - started, error_code=reason,
                                   http_status=status_code if type(status_code) is int else None,
                                   error_type=type(exc).__name__)
                    attempt['error_message_signals']=signals
                    record.update(reason=reason, retry_allowed=retry_allowed)
                    self.store.put(key, record)
                    if retry_allowed and len(record["attempts"]) < self.max_attempts:
                        if delay:
                            self.sleeper(delay)
                        continue
                    break
                usage = _usage(response)
                attempt.update(elapsed_seconds=time.perf_counter() - started,
                               response_id=_get(response, "id"), response_model=_get(response, "model"))
                if usage is None:
                    self.store.mark_uncertain(charge_id)
                    attempt["state"] = "uncertain"
                    record.update(reason="usage_missing", retry_allowed=False)
                    self.store.put(key, record)
                    break
                attempt["usage"] = {"input_tokens": usage[0], "output_tokens": usage[1]}
                actual = self.config.cost(*usage)
                attempt["cost_micro_usd"] = actual
                attempt["state"] = "confirmed"
                if usage[0] > self.config.input_token_cap or usage[1] > self.config.output_token_cap:
                    record.update(reason="provider_token_cap_overrun", retry_allowed=False)
                    self.store.settle(charge_id, actual, usage=attempt["usage"], cache_entries={key: record})
                    break
                try:
                    visible = _visible_output(response)
                    attempt["visible_output"] = visible
                    parsed = validator(_parse_json(visible))
                    record.update(status="valid", value=parsed, reason=None, retry_allowed=False)
                except ModelValidationError as exc:
                    # Validation messages are our fixed strings; no provider
                    # exception text, headers, hidden reasoning or SDK dump.
                    attempt["validation_error"] = str(exc)
                    record.update(reason="invalid_structured_output", retry_allowed=True)
                self.store.settle(charge_id, actual, usage=attempt["usage"], cache_entries={key: record})
                if record["status"] == "valid":
                    break
            if record["status"] != "valid" and len(record["attempts"]) >= self.max_attempts:
                record["retry_allowed"] = False
                self.store.put(key, record)
            return {**record, "cache_hit": False}
