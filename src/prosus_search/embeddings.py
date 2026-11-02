"""Cached hosted embeddings with explicit network opt-in and bounded spending."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
import time
from typing import Any, Callable, Sequence

import numpy as np

from .store import Store, content_hash, cost_micro_usd, utc_now


class EmbeddingError(RuntimeError):
    pass


class OfflineCacheMissError(EmbeddingError):
    pass


class EmbeddingResponseError(EmbeddingError):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dimensions: int
    input_rate_per_million: float
    epoch: str = "v1"
    provider: str = "openai"
    batch_size: int = 64
    batch_token_limit: int = 32_768
    build_timeout: float = 60.0
    interactive_timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.model or not self.epoch or self.provider != "openai":
            raise ValueError("Hosted embeddings require an explicit OpenAI model and cache epoch")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if not 1 <= self.batch_size <= 64 or not 1 <= self.batch_token_limit <= 32_768:
            raise ValueError("Embedding batches exceed the frozen operational limits")
        if self.build_timeout <= 0 or self.interactive_timeout <= 0:
            raise ValueError("Embedding timeouts must be positive")
        cost_micro_usd(0, self.input_rate_per_million)


def embedding_cache_key(text: str, config: EmbeddingConfig, representation_version: str) -> str:
    return "embedding:" + content_hash({
        "schema": 1, "provider": config.provider, "model": config.model,
        "dimensions": config.dimensions, "epoch": config.epoch,
        "representation_version": representation_version,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    })


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _normalize(matrix: Any, rows: int, dimensions: int) -> np.ndarray:
    try:
        array = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise EmbeddingResponseError("Embedding response contains invalid numbers") from None
    if array.shape != (rows, dimensions) or not np.isfinite(array).all():
        raise EmbeddingResponseError("Embedding response shape or finite-value validation failed")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise EmbeddingResponseError("Embedding response contains zero or invalid vectors")
    return np.ascontiguousarray(array / norms[:, None], dtype=np.float32)


def _response_vectors(response: Any, count: int, dimensions: int) -> np.ndarray:
    data = _field(response, "data")
    if not isinstance(data, (list, tuple)) or len(data) != count:
        raise EmbeddingResponseError("Embedding response is incomplete")
    vectors: list[Any] = [None] * count
    seen: set[int] = set()
    for entry in data:
        index = _field(entry, "index")
        if type(index) is not int or not 0 <= index < count or index in seen:
            raise EmbeddingResponseError("Embedding response has duplicate or invalid indexes")
        seen.add(index)
        vectors[index] = _field(entry, "embedding")
    return _normalize(vectors, count, dimensions)


def _usage_tokens(response: Any) -> int | None:
    usage = _field(response, "usage")
    value = _field(usage, "total_tokens", _field(usage, "prompt_tokens"))
    return value if type(value) is int and value >= 0 else None


def _error_kind(error: Exception) -> tuple[str, bool, float]:
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", {})
    if isinstance(body, dict):
        body = body.get("error", body)
    code = body.get("code") if isinstance(body, dict) else None
    code = code or getattr(error, "code", None)
    if status in (401, 403):
        return "authentication or model access denied", False, 0
    if code in ("insufficient_quota", "billing_hard_limit_reached"):
        return "account quota exhausted", False, 0
    retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
    retryable = retryable or isinstance(error, TimeoutError) or type(error).__name__ == "APITimeoutError"
    delay = 1.0
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    try:
        delay = max(0.0, float(headers.get("retry-after", "1")))
    except (ValueError, TypeError, AttributeError):
        pass
    if not np.isfinite(delay) or delay > 60:
        return "provider retry delay exceeds the 60-second bound", False, 0
    return ("temporary provider failure" if retryable else "provider request failed"), retryable, delay


class EmbeddingClient:
    def __init__(self, store: Store, config: EmbeddingConfig, token_counter: Callable[[str], int],
                 allow_api: bool = False, provider: Any = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.store, self.config, self.token_counter = store, config, token_counter
        self.allow_api, self._provider, self._sleep = allow_api, provider, sleep
        self.last_run: dict[str, Any] = {}

    def _get_provider(self) -> Any:
        if self._provider is not None:
            return self._provider
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if (not key or any(marker in key.casefold() for marker in ("placeholder", "your_api_key", "your-openai-api-key", "sk-your-key", "replace_me"))):
            raise EmbeddingError("Set a real OPENAI_API_KEY securely in the environment before enabling API calls")
        try:
            from openai import OpenAI
        except ImportError:
            raise EmbeddingError("Install the project's pinned OpenAI dependency before using hosted embeddings") from None
        # SDK import can enable debug logging through environment settings.
        # Never emit request bodies or headers from this application.
        for logger in ("openai", "httpx", "httpcore", "httpx2", "httpcore2"):
            logging.getLogger(logger).setLevel(logging.WARNING)
        try:
            self._provider = OpenAI(api_key=key, max_retries=0, timeout=self.config.build_timeout)
        except Exception:
            raise EmbeddingError("Cannot initialize the OpenAI client; check the installed SDK and environment configuration") from None
        return self._provider

    def _request(self, provider: Any, texts: Sequence[str], interactive: bool) -> Any:
        kwargs = {"model": self.config.model, "input": list(texts), "dimensions": self.config.dimensions,
                  "encoding_format": "float", "timeout": self.config.interactive_timeout if interactive else self.config.build_timeout}
        if callable(provider) and not hasattr(provider, "embeddings"):
            return provider(**kwargs)
        return provider.embeddings.create(**kwargs)

    def embed(self, texts: Sequence[str], representation_version: str = "query-v1",
              interactive: bool = False, bypass_cache: bool = False,
              bucket: str = "embeddings") -> np.ndarray:
        if isinstance(texts, str) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be a sequence of nonblank strings")
        started = time.perf_counter()
        self.last_run = {"model": self.config.model, "inputs": len(texts), "unique_inputs": 0,
                         "cache_hits": 0, "api_requests": 0, "retries": 0, "input_tokens": 0,
                         "bypass_cache": bypass_cache, "elapsed_seconds": 0.0}
        if not texts:
            return np.empty((0, self.config.dimensions), dtype=np.float32)
        keys = [embedding_cache_key(text, self.config, representation_version) for text in texts]
        unique = dict(zip(keys, texts))
        self.last_run["unique_inputs"] = len(unique)
        found: dict[str, np.ndarray] = {}
        missing: list[tuple[str, str, int]] = []
        for key, text in unique.items():
            vector = None if bypass_cache else self.store.get_embedding(key, self.config.dimensions)
            if vector is not None:
                found[key] = vector
                self.last_run["cache_hits"] += 1
            else:
                count = self.token_counter(text)
                if type(count) is not int or count <= 0 or count > self.config.batch_token_limit:
                    raise EmbeddingError("Embedding token count is invalid or exceeds the request bound; input was not truncated")
                missing.append((key, text, count))
        if missing and not self.allow_api:
            raise OfflineCacheMissError(f"Offline embedding cache miss for {len(missing)} unique inputs; build with explicit --allow-api and a configured key")
        if missing:
            # Validate credentials/SDK before reserving anything. Holding the OS
            # lock, recheck the cache in case a prior writer completed meanwhile.
            provider = self._get_provider()
            with self.store.api_writer():
                remaining = []
                for key, text, count in missing:
                    vector = None if bypass_cache else self.store.get_embedding(key, self.config.dimensions)
                    if vector is not None:
                        found[key] = vector
                        self.last_run["cache_hits"] += 1
                    else:
                        remaining.append((key, text, count))
                batches: list[list[tuple[str, str, int]]] = []
                batch: list[tuple[str, str, int]] = []
                tokens = 0
                for entry in remaining:
                    if batch and (len(batch) == self.config.batch_size or tokens + entry[2] > self.config.batch_token_limit):
                        batches.append(batch)
                        batch, tokens = [], 0
                    batch.append(entry)
                    tokens += entry[2]
                if batch:
                    batches.append(batch)
                for batch in batches:
                    batch_tokens = sum(entry[2] for entry in batch)
                    reservation = cost_micro_usd(batch_tokens, self.config.input_rate_per_million)
                    attempts = 1 if interactive else 2
                    for attempt in range(attempts):
                        request_id = self.store.reserve(bucket, reservation, metadata={
                            "kind": "embedding", "model": self.config.model, "dimensions": self.config.dimensions,
                            "input_count": len(batch), "input_tokens_bound": batch_tokens,
                            "payload_hash": content_hash([entry[0] for entry in batch]), "attempt": attempt + 1,
                        })
                        self.last_run["api_requests"] += 1
                        try:
                            response = self._request(provider, [entry[1] for entry in batch], interactive)
                        except Exception as error:
                            self.store.mark_uncertain(request_id)
                            message, retryable, delay = _error_kind(error)
                            if retryable and attempt + 1 < attempts:
                                self.last_run["retries"] += 1
                                self._sleep(delay)
                                continue
                            raise EmbeddingError(f"Embedding {message}; possible cost retained; retry later from cache") from None
                        usage_tokens = _usage_tokens(response)
                        try:
                            vectors = _response_vectors(response, len(batch), self.config.dimensions)
                        except EmbeddingResponseError:
                            if usage_tokens is None:
                                self.store.mark_uncertain(request_id)
                            else:
                                self.store.settle(request_id, cost_micro_usd(usage_tokens, self.config.input_rate_per_million),
                                                  usage={"input_tokens": usage_tokens})
                            raise
                        if usage_tokens is None:
                            self.store.mark_uncertain(request_id)
                            raise EmbeddingResponseError("Embedding response has no valid usage; possible charge retained, vectors not cached")
                        metadata = {"requested_model": self.config.model, "response_model": _field(response, "model", self.config.model),
                                    "epoch": self.config.epoch, "representation_version": representation_version, "created_at": utc_now()}
                        entries = [(entry[0], vectors[index], metadata) for index, entry in enumerate(batch)]
                        self.store.settle(request_id, cost_micro_usd(usage_tokens, self.config.input_rate_per_million),
                                          usage={"input_tokens": usage_tokens}, embedding_entries=[] if bypass_cache else entries)
                        self.last_run["input_tokens"] += usage_tokens
                        found.update({entry[0]: vectors[index] for index, entry in enumerate(batch)})
                        break
        self.last_run["elapsed_seconds"] = time.perf_counter() - started
        return np.stack([found[key] for key in keys]).astype(np.float32, copy=False)
