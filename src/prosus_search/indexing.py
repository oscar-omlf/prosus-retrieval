"""Build the local, immutable catalog index without evaluation dependencies."""
from __future__ import annotations

import time

from .artifacts import file_hash, publish_current, publish_release
from .catalog import load_catalog
from .config import token_counter
from .representation import prepare_catalog
from .runtime import local_client, open_store


def build(config):
    started = time.perf_counter()
    if file_hash(config.resolve('catalog')) != config['data']['catalog_sha256']:
        raise ValueError('Catalog differs from its configured SHA-256; version the data before rebuilding')
    catalog = load_catalog(config.resolve('catalog'))
    representation = config['representation']
    prepared = prepare_catalog(catalog, token_counter(config),
        representation['item_token_cap'], representation['version'])
    client = local_client(config, open_store(config))
    texts = [item.text for item in prepared.items]
    if max(client.count_tokens(text, 'passage') for text in texts) > 512:
        raise ValueError('Catalog exceeds the model token limit; no silent truncation')
    vectors = client.embed(texts, representation_version=prepared.representation_version,
                           kind='passage')
    model = config['local']
    release = publish_release(config.resolve('artifacts'), prepared, vectors,
        model['model'], model['dimensions'], model['revision'],
        metadata={'config_hash': config.digest, 'retrieval': config['retrieval']})
    publish_current(config.resolve('artifacts'), {'semantic-local': release},
                    default_mode='semantic-local')
    return {'release_id': release, 'items': len(catalog.records),
            'matrix_bytes': int(vectors.nbytes),
            'elapsed_seconds': time.perf_counter() - started, 'api_calls': 0}
