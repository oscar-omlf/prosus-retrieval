"""Localhost JSON demo; reused runtime, with explicit budgeted quality API opt-in."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import parse_qs, urlparse

from .service import SearchService, SearchBusyError


def serve(config='config.toml', *, port=8765, mode='constraint-hybrid', allow_api=False):
    if allow_api and mode!='contextual':raise ValueError('Server API opt-in is supported only for contextual quality mode')
    if mode=='contextual':
        from .quality import QualitySearch
        service=QualitySearch(config,allow_api=allow_api)
    else:
        service = SearchService(config, mode=mode, allow_api=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # do not log request text or query strings

        def respond(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = urlparse(self.path)
            if route.path in {'/', '/health'}:
                self.respond(200, {'status': 'ready', 'mode': mode, 'release_id': service.release_id,
                    'remote_api_enabled': allow_api, 'usage': '/search?q=hamburguer%20sem%20queijo&k=5'})
                return
            if route.path != '/search':
                self.respond(404, {'error': 'Unknown route'})
                return
            try:
                args = parse_qs(route.query)
                query = args.get('q', [''])[0]
                if len(query) > 2048:
                    raise ValueError('Query exceeds demo character limit')
                result = service.search(query, int(args.get('k', ['5'])[0]))
                self.respond(200, result)
            except SearchBusyError:
                self.respond(429, {'error': 'Search busy; retry later'})
            except (ValueError, RuntimeError) as exc:
                self.respond(400, {'error': str(exc), 'remote_api_enabled': allow_api})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Listening on http://127.0.0.1:{port} | {mode} | API {"enabled with ledger ceiling" if allow_api else "disabled"}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
