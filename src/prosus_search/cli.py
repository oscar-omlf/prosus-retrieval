"""One command interface; each operation delegates to a focused module."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config, token_counter
from .runtime import open_store

MODES = ('semantic-local', 'hybrid-local', 'lexical', 'constraint-hybrid', 'contextual')


def parser(default_config='config.toml'):
    result = argparse.ArgumentParser(description='Portuguese marketplace search')
    result.add_argument('--config', default=str(default_config))
    commands = result.add_subparsers(dest='command', required=True)
    commands.add_parser('setup', help='download and validate the pinned local model/tokenizer')
    commands.add_parser('build', help='build a versioned local catalog index')
    query = commands.add_parser('search', help='rank offers for one Portuguese query')
    query.add_argument('query')
    query.add_argument('--json', action='store_true')
    batch = commands.add_parser('batch', help='generate ranked IDs for a query CSV')
    batch.add_argument('--queries', help='query CSV; defaults to the configured assignment input')
    batch.add_argument('--output', default='outputs/results.csv')
    batch.add_argument('--overwrite', action='store_true')
    server = commands.add_parser('serve', help='serve localhost JSON search')
    server.add_argument('--port', type=int, default=8766)
    for command in (query, batch, server):
        command.add_argument('--mode', choices=MODES, default='semantic-local')
        command.add_argument('--allow-api', action='store_true',
            help='allow budgeted official model calls in contextual mode only')
    for command in (query, batch):
        command.add_argument('--k', type=int, default=5)
    commands.add_parser('budget', help='inspect cumulative API accounting')
    return result


def make_service(config, mode, allow_api=False):
    if allow_api and mode != 'contextual':
        raise ValueError('--allow-api is supported only for contextual mode')
    if mode == 'contextual':
        from .quality import QualitySearch
        return QualitySearch(config, allow_api=allow_api)
    from .service import SearchService
    return SearchService(config, mode=mode, allow_api=False)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None, *, default_config='config.toml'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    args = parser(default_config).parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == 'setup':
            from .local_embeddings import download_model
            token_counter(config, allow_download=True)
            emit(download_model(config.resolve('model_dir'), config['local']['revision']))
        elif args.command == 'build':
            from .indexing import build
            emit(build(config))
        elif args.command == 'budget':
            emit(open_store(config).summary())
        elif args.command == 'serve':
            from .server import serve
            serve(config, port=args.port, mode=args.mode, allow_api=args.allow_api)
        else:
            service = make_service(config, args.mode, args.allow_api)
            if args.command == 'batch':
                from .batch import export
                emit(export(service, args.queries or config.resolve('queries'),
                    args.output, k=args.k, overwrite=args.overwrite))
            else:
                response = service.search(args.query, args.k)
                if args.json:
                    emit(response)
                else:
                    print(f"{response['mode']} | {len(response['results'])} results")
                    for item in response['results']:
                        print(f"{item['rank']}. {item['name']} [{item['item_id']}]")
    except (ValueError, RuntimeError, OSError) as error:
        # Transport errors are sanitized by the model boundary, never SDK dumps.
        print(f'Error: {error}', file=sys.stderr)
        return 2
    return 0
