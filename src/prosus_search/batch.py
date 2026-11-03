"""Generate ranked IDs from a query CSV; outputs are local runtime files."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import uuid

from .catalog import load_queries


def export(service, queries, output, *, k=5, overwrite=False):
    destination = Path(output).resolve()
    if destination.suffix.lower() != '.csv':
        raise ValueError('Batch output must have a .csv extension')
    if destination.exists() and not overwrite:
        raise ValueError('Output exists; use --overwrite to replace it explicitly')
    if destination == Path(queries).resolve():
        raise ValueError('Output cannot overwrite the query input')
    if destination == service.config.resolve('catalog'):
        raise ValueError('Output cannot overwrite the catalog input')
    rows = []
    for query in load_queries(queries):
        response = service.search(query.text, k)
        rows.extend({'query_id': query.query_id, 'rank': item['rank'],
            'item_id': item['item_id'], 'score': item['score'],
            'score_kind': item['score_kind'], 'mode': response['mode'],
            'release_id': response['release_id']} for item in response['results'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['query_id', 'rank', 'item_id',
                'score', 'score_kind', 'mode', 'release_id'], lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'output': str(destination), 'rows': len(rows),
            'queries': len({row['query_id'] for row in rows})}
