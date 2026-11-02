"""The online path: load once, embed a query, score, return top k."""
from __future__ import annotations

from dataclasses import asdict
import time

from .artifacts import load_current, load_release
from .benchmark import expected_space, make_embedder, open_store, restored_prepared
from .catalog import load_catalog
from .config import Config, ConfigError, load_config, token_counter
from .representation import prepare_catalog, prepare_query
from .retrieval import SearchEngine


class SearchService:
    """Reusable process-local search instance; no judge or rewrite in this path.

    Local search never calls a remote model. Hosted cache misses fail unless the
    caller explicitly enables APIs. Construct once to amortize model/index load.
    """

    def __init__(self, config: Config | str = "config.toml", *, mode: str | None = None,
                 release_id: str | None = None, allow_api: bool = False):
        self.config = load_config(config) if isinstance(config, str) else config
        self.counter = token_counter(self.config)
        self.release_id = release_id
        self.embedder = None
        if mode == "lexical":
            if release_id:
                release = load_release(self.config.resolve("artifacts"), release_id)
                prepared = restored_prepared(release)
            else:
                catalog = load_catalog(self.config.resolve("catalog"))
                rep = self.config["representation"]
                prepared = prepare_catalog(catalog, self.counter, rep["item_token_cap"], rep["version"])
            self.engine = SearchEngine(prepared, **self.config["retrieval"])
            self.family, self.mode, self.retrieval_mode = None, "lexical", "lexical"
            return
        pointer = None
        if mode is None or release_id is None:
            pointer = load_current(self.config.resolve("artifacts"))
        self.mode = mode or pointer["default_mode"]
        parts = self.mode.split("-", 1)
        if len(parts) != 2 or parts[0] not in {"semantic", "hybrid"} or parts[1] not in {"local", "small", "large"}:
            raise ConfigError("Use lexical, semantic-{local,small,large}, or hybrid-{local,small,large}")
        self.retrieval_mode, self.family = parts
        if self.release_id is None:
            self.release_id = pointer["models"].get("semantic-" + self.family)
            if self.release_id is None:
                raise ConfigError(f"No built release for {self.family}; run the matching build command first")
        model, dimensions, epoch = expected_space(self.config, self.family)
        release = load_release(self.config.resolve("artifacts"), self.release_id, model, dimensions, epoch)
        if release.manifest["metadata"].get("retrieval") != self.config["retrieval"]:
            raise ConfigError("Retrieval settings differ from release; record a versioned build before changing them")
        self.engine = SearchEngine(restored_prepared(release), release.vectors, **self.config["retrieval"])
        self.embedder = make_embedder(self.config, open_store(self.config), self.family, self.counter, allow_api=allow_api)

    def search(self, query: str, k: int = 5, *, bypass_query_cache: bool = False) -> dict:
        start = time.perf_counter()
        if type(k) is not int or not 1 <= k <= len(self.engine.ids):
            raise ConfigError(f"k must be an integer between 1 and {len(self.engine.ids)}")
        if self.retrieval_mode == "hybrid" and k > min(len(self.engine.ids), self.engine.rrf_depth):
            raise ConfigError(f"Interactive hybrid search supports k up to {min(len(self.engine.ids), self.engine.rrf_depth)} at its fixed candidate depth")
        text = prepare_query(query, self.counter, self.config["representation"]["query_token_cap"])
        vector = None
        embed_start = time.perf_counter()
        if self.embedder is not None:
            options = {"kind": "query"} if self.family == "local" else {"interactive": True}
            vector = self.embedder.embed([text], representation_version="query-v1",
                                         bypass_cache=bypass_query_cache, **options)[0]
        embedding_seconds = time.perf_counter() - embed_start
        score_start = time.perf_counter()
        results = self.engine.search(text, vector, mode=self.retrieval_mode, k=k)
        scoring_seconds = time.perf_counter() - score_start
        return {"query": query, "mode": self.mode, "release_id": self.release_id,
                "results": [asdict(item) for item in results],
                "timing_ms": {"embedding": embedding_seconds * 1000,
                              "scoring": scoring_seconds * 1000,
                              "total": (time.perf_counter() - start) * 1000},
                "scores_are_probabilities": False}


def search(query: str, k: int = 5, *, config: str = "config.toml", mode: str | None = None,
           allow_api: bool = False) -> dict:
    """Convenience call. Reuse SearchService for repeated queries and lower latency."""
    return SearchService(config, mode=mode, allow_api=allow_api).search(query, k)
