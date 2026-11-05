"""Local search playground and JSON API on the same loopback-only server."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from threading import RLock
import time
from urllib.parse import parse_qs, urlparse

from .catalog import load_catalog, load_queries
from .cli import MODES, make_service
from .config import load_config
from .runtime import open_store
from .service import SearchBusyError


MODE_INFO = {
    'contextual': ('Contextual', 'Hybrid retrieval, ingredient checks and GPT-4.1-mini reranking.'),
    'constraint-hybrid': ('Constraint-aware', 'Local hybrid ranking with ingredient exclusions. Complex requests need a cached interpretation.'),
    'semantic-local': ('Semantic', 'Local E5 embeddings match meaning across different wording.'),
    'hybrid-local': ('Hybrid', 'E5 semantic retrieval and BM25 keywords, combined with rank fusion.'),
    'lexical': ('Keyword', 'BM25 matches product names, descriptions and exact wording.'),
}
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/assets/app.css': ('app.css', 'text/css; charset=utf-8'),
          '/assets/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/assets/prosus-logo.png': ('prosus-logo.png', 'image/png')}
IMAGE_BASE = 'https://static.ifood-static.com.br/image/upload/t_low/pratos/'


def image_url(images):
    """Only catalog-relative iFood image paths become browser URLs."""
    if not isinstance(images, (tuple, list)):
        return None
    for value in images:
        if (isinstance(value, str) and value and not value.startswith(('/', '\\'))
                and not any(c in value for c in (':', '?', '#', '\\'))
                and all(p not in {'', '.', '..'} for p in value.split('/'))):
            from urllib.parse import quote
            return IMAGE_BASE + quote(value, safe='/')
    return None


class Playground:
    """Reuse services lazily; serialize demo requests, never change rankers."""
    def __init__(self, config='config.toml', *, mode='constraint-hybrid', allow_api=False):
        if mode not in MODES:
            raise ValueError('Unsupported search mode')
        if allow_api and mode != 'contextual':
            raise ValueError('Server API opt-in is supported only for contextual quality mode')
        self.config = load_config(config) if isinstance(config, str) else config
        self.default_mode, self.allow_api = mode, allow_api
        self.lock = RLock()
        self.services = {mode: make_service(self.config, mode, allow_api)}
        self.catalog = load_catalog(self.config.resolve('catalog'))
        self.products = {r.item_id: r.metadata for r in self.catalog.records}
        self.examples = [q.text for q in load_queries(self.config.resolve('queries'))][:6]
        self.store = open_store(self.config)

    def health(self):
        return {'status': 'ready', 'mode': self.default_mode,
                'release_id': self.services[self.default_mode].release_id,
                'remote_api_enabled': self.allow_api,
                'usage': '/search?q=hamburguer%20sem%20queijo&k=5'}

    def metadata(self):
        ledger = self.store.summary()
        return {**self.health(), 'catalog_count': len(self.products), 'examples': self.examples,
                'modes': [{'id': key, 'name': MODE_INFO[key][0], 'description': MODE_INFO[key][1]}
                          for key in ('contextual', 'constraint-hybrid', 'semantic-local', 'hybrid-local', 'lexical')],
                'budget': {'accounted_usd': ledger['effective_spend'] / 1e6,
                           'ceiling_usd': ledger['global_limit'] / 1e6}}

    def search(self, query, k=5, mode=None):
        chosen = mode or self.default_mode
        if chosen not in MODES:
            raise ValueError('Unsupported search mode')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Enter a Portuguese food query')
        if len(query) > 2048:
            raise ValueError('Query exceeds demo character limit')
        if not self.lock.acquire(blocking=False):
            raise SearchBusyError('Search busy; retry later')
        try:
            started = time.perf_counter()
            if chosen not in self.services:
                self.services[chosen] = make_service(self.config, chosen, self.allow_api and chosen == 'contextual')
            response = self.services[chosen].search(query, k)
            enriched = []
            for result in response['results']:
                metadata = self.products.get(result['item_id'], {})
                enriched.append({**result, 'category': metadata.get('category_name') or None,
                                 'price': metadata.get('price'), 'currency': 'BRL',
                                 'image_url': image_url(metadata.get('images'))})
            return {**response, 'results': enriched,
                    'http_server_ms': (time.perf_counter() - started) * 1000,
                    'remote_api_enabled': self.allow_api and chosen == 'contextual'}
        finally:
            self.lock.release()


def make_handler(playground):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Queries and request paths never enter the access log.

        def send_bytes(self, status, data, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "img-src 'self' https://static.ifood-static.com.br; connect-src 'self'; "
                             "font-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def respond(self, status, value):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                            'application/json; charset=utf-8')

        def do_GET(self):
            # A remote page must not spend the local API budget through a GET.
            port = self.server.server_address[1]
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            if (self.headers.get('Host') not in hosts
                    or self.headers.get('Sec-Fetch-Site') == 'cross-site'
                    or self.headers.get('Origin') not in {None, *('http://' + h for h in hosts)}):
                self.respond(403, {'error': 'Use the local playground origin'})
                return
            route = urlparse(self.path)
            if route.path in ASSETS:
                name, content_type = ASSETS[route.path]
                self.send_bytes(200, files('prosus_search').joinpath('web', name).read_bytes(), content_type)
                return
            requested_mode = playground.default_mode
            try:
                if route.path == '/health':
                    self.respond(200, playground.health())
                elif route.path == '/api/meta':
                    self.respond(200, playground.metadata())
                elif route.path == '/search':
                    args = parse_qs(route.query)
                    requested_mode = args.get('mode', [playground.default_mode])[0]
                    self.respond(200, playground.search(args.get('q', [''])[0],
                                 int(args.get('k', ['5'])[0]), args.get('mode', [None])[0]))
                else:
                    self.respond(404, {'error': 'Unknown route'})
            except SearchBusyError:
                self.respond(429, {'error': 'Search busy; retry later'})
            except (ValueError, RuntimeError) as exc:
                self.respond(400, {'error': str(exc),
                                  'remote_api_enabled': playground.allow_api and requested_mode == 'contextual'})

    return Handler


def serve(config='config.toml', *, port=8765, mode='constraint-hybrid', allow_api=False):
    playground = Playground(config, mode=mode, allow_api=allow_api)
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(playground))
    print(f'Listening on http://127.0.0.1:{port} | {mode} | API '
          f'{"enabled with ledger ceiling" if allow_api else "disabled"}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
