"""Small, strict project configuration; all paths resolve beside the TOML file."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any


class ConfigError(ValueError):
    pass


_KEYS = {
    "paths": {"catalog", "queries", "state", "artifacts", "model_dir", "split", "outputs", "tokenizer_cache"},
    "data": {"catalog_sha256", "queries_sha256"},
    "representation": {"version", "item_token_cap", "query_token_cap"},
    "retrieval": {"k1", "b", "rrf_k", "rrf_depth"},
    "local": {"model", "revision", "dimensions", "batch_size", "threads"},
    "budget": {"global_micro_usd", "embeddings", "development", "final", "round4", "recovery"},
    "evaluation": {"seed", "bootstrap_replicates", "primary_model", "secondary_model", "input_token_cap", "output_token_cap"},
}
_MODEL_KEYS = {"model", "dimensions", "input_rate_per_million", "epoch"}


@dataclass(frozen=True)
class Config:
    path: Path
    values: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def resolve(self, name: str) -> Path:
        return (self.path.parent / self.values["paths"][name]).resolve()

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.values, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()


def load_config(path: str | Path = "config.toml") -> Config:
    source = Path(path).resolve()
    try:
        values = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read configuration: {source.name} ({type(exc).__name__})") from None
    if set(values) != set(_KEYS) | {"models"}:
        raise ConfigError("Configuration sections differ from the documented schema")
    for section, keys in _KEYS.items():
        if not isinstance(values[section], dict) or set(values[section]) != keys:
            raise ConfigError(f"Missing or unknown keys in [{section}]")
    if set(values["models"]) != {"small", "large"}:
        raise ConfigError("Expected exactly small and large hosted model configurations")
    for name, model in values["models"].items():
        if set(model) != _MODEL_KEYS:
            raise ConfigError(f"Missing or unknown keys in [models.{name}]")
        if type(model["dimensions"]) is not int or model["dimensions"] <= 0:
            raise ConfigError("Model dimensions must be a positive integer")
        try:
            rate = Decimal(str(model["input_rate_per_million"]))
        except Exception:
            raise ConfigError("Model rate must be a finite nonnegative decimal") from None
        if not rate.is_finite() or rate < 0 or not model["model"] or not model["epoch"]:
            raise ConfigError("Invalid hosted model configuration")
    if any(not isinstance(v, str) or not v for v in values["paths"].values()):
        raise ConfigError("All configured paths must be nonempty strings")
    for digest in values["data"].values():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ConfigError("Dataset fingerprints must be SHA256 hex strings")
    integer_fields = {
        "representation": ("item_token_cap", "query_token_cap"),
        "local": ("dimensions", "batch_size", "threads"),
        "retrieval": ("rrf_k", "rrf_depth"),
        "evaluation": ("seed", "bootstrap_replicates", "input_token_cap", "output_token_cap"),
        "budget": tuple(_KEYS["budget"]),
    }
    for section, names in integer_fields.items():
        for name in names:
            value = values[section][name]
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{section}.{name} must be a positive integer")
    lexical = values["retrieval"]
    if not 0 < lexical["k1"] < 100 or not 0 <= lexical["b"] <= 1:
        raise ConfigError("BM25 requires positive k1 and b in [0,1]")
    budget = values["budget"]
    if budget["global_micro_usd"] > 4_500_000:
        raise ConfigError("The untouched $0.50 reserve cannot be allocated")
    if sum(value for key, value in budget.items() if key != "global_micro_usd") > budget["global_micro_usd"]:
        raise ConfigError("Bucket allocations exceed the global ceiling")
    return Config(source, values)


def token_counter(config: Config, *, allow_download: bool = False):
    """Initialize cached cl100k_base without a surprise first-use download."""
    cache = config.resolve("tokenizer_cache")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    # tiktoken keys its downloaded asset by the URL SHA1; verify before loading.
    asset = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cached = cache / hashlib.sha1(asset.encode()).hexdigest()
    if not cached.is_file() and not allow_download:
        raise ConfigError("Tokenizer cache is missing. Run `food-search setup` once with network access.")
    expected = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
    if cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest() != expected:
        raise ConfigError("Tokenizer cache checksum failed; remove the corrupt asset and rerun setup.")
    cache.mkdir(parents=True, exist_ok=True)
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text, disallowed_special=()))
