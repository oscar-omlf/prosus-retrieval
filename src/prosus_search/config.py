"""Strict application configuration; paths resolve beside the configuration file."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    path: Path
    values: dict[str, Any]

    def __getitem__(self, key):
        return self.values[key]

    def resolve(self, name):
        return (self.path.parent / self.values['paths'][name]).resolve()

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.values, sort_keys=True,
            separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def load_config(path='config.toml'):
    source = Path(path).resolve()
    try:
        values = tomllib.loads(source.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ConfigError('Cannot read valid configuration') from None
    required = {
        'paths': {'catalog', 'queries', 'state', 'artifacts', 'model_dir', 'tokenizer_cache'},
        'data': {'catalog_sha256', 'queries_sha256'},
        'representation': {'version', 'item_token_cap', 'query_token_cap'},
        'retrieval': {'k1', 'b', 'rrf_k', 'rrf_depth'},
        'local': {'model', 'revision', 'dimensions', 'batch_size', 'threads'},
        'models': {'query', 'ranker'},
        'budget': {'global_micro_usd', 'embeddings', 'development', 'final', 'round4', 'recovery'},
    }
    if set(values) != set(required) or any(set(values[k]) != fields for k, fields in required.items()):
        raise ConfigError('Configuration fields differ from the application schema')
    if any(not isinstance(v, str) or not v for v in values['paths'].values()):
        raise ConfigError('Paths must be nonempty strings')
    if any(not re.fullmatch(r'[0-9a-f]{64}', v) for v in values['data'].values()):
        raise ConfigError('Data hashes must be SHA-256 strings')
    local = values['local']
    if local['model'] != 'intfloat/multilingual-e5-small' or local['dimensions'] != 384:
        raise ConfigError('The local adapter requires multilingual-e5-small, 384 dimensions')
    if not re.fullmatch(r'[0-9a-f]{40}', local['revision']):
        raise ConfigError('Pin the local model to a commit revision')
    allowed = {'openai/gpt-4.1-mini', 'openai/gpt-4.1'}
    if any(model not in allowed for model in values['models'].values()):
        raise ConfigError('Unsupported query or ranking model')
    integers = {'local': ('dimensions', 'batch_size', 'threads'),
        'representation': ('item_token_cap', 'query_token_cap'),
        'retrieval': ('rrf_k', 'rrf_depth'), 'budget': tuple(required['budget'])}
    for section, fields in integers.items():
        if any(type(values[section][f]) is not int or values[section][f] <= 0 for f in fields):
            raise ConfigError('Integer settings must be positive')
    if not 0 < values['retrieval']['k1'] < 100 or not 0 <= values['retrieval']['b'] <= 1:
        raise ConfigError('Invalid lexical retrieval parameters')
    budget = values['budget']
    if (budget['global_micro_usd'] > 4_500_000 or
            sum(v for k, v in budget.items() if k != 'global_micro_usd') > budget['global_micro_usd']):
        raise ConfigError('Budget exceeds the cumulative $4.50 ceiling')
    return Config(source, values)


def token_counter(config, *, allow_download=False):
    cache = config.resolve('tokenizer_cache')
    os.environ['TIKTOKEN_CACHE_DIR'] = str(cache)
    url = 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'
    asset = cache / hashlib.sha1(url.encode()).hexdigest()
    if not asset.is_file() and not allow_download:
        raise ConfigError('Tokenizer missing; run python scripts/main.py setup first')
    expected = '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'
    if asset.is_file() and hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
        raise ConfigError('Tokenizer checksum failed')
    cache.mkdir(parents=True, exist_ok=True)
    import tiktoken
    encoder = tiktoken.get_encoding('cl100k_base')
    return lambda text: len(encoder.encode(text, disallowed_special=()))
