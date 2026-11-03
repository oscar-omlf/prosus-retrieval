"""Exact semantic retrieval, a lexical baseline, and fixed reciprocal-rank fusion.

No provider calls or judgments occur here. Scores rank catalog evidence; they
are not probabilities of relevance. All ties resolve by canonical item ID.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from threading import Lock
from typing import Collection, Sequence

import numpy as np

from .representation import PreparedCatalog, lexical_tokens, normalize_text
from .snapshots import freeze_json


class RetrievalError(ValueError):
    """Invalid vectors, ranking configuration, or unavailable retrieval inputs."""


@dataclass(frozen=True)
class ScoreBundle:
    scores: np.ndarray
    score_kind: str
    dense_scores: np.ndarray | None = None
    lexical_scores: np.ndarray | None = None
    dense_ranks: np.ndarray | None = None
    lexical_ranks: np.ndarray | None = None
    candidate_count: int = 0
    eligible_rows: np.ndarray | None = None


@dataclass(frozen=True)
class RankedItem:
    item_id: str
    merchant_id: str
    rank: int
    score: float
    score_kind: str
    name: str
    description: str
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    lexical_evidence: bool | None = None


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RetrievalError(f"{name} must be a positive integer")


def _validated_scores(ids: Sequence[str], scores: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(scores)
    if array.ndim != 1 or len(array) != len(ids):
        raise RetrievalError("scores and item IDs must be aligned one-dimensional arrays")
    if not len(ids) or len(set(ids)) != len(ids) or any(not isinstance(value, str) or not value for value in ids):
        raise RetrievalError("item IDs must be nonempty unique strings")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)) or np.iscomplexobj(array):
        raise RetrievalError("scores must be finite real numbers")
    return array


def stable_rank(ids: Sequence[str], scores: Sequence[float] | np.ndarray, k: int | None = None) -> np.ndarray:
    """Return row indices ordered by unrounded score descending, then ID."""
    array = _validated_scores(ids, scores)
    if k is None:
        k = len(ids)
    _positive_integer(k, "k")
    if k > len(ids):
        raise RetrievalError(f"k={k} exceeds catalog size {len(ids)}")
    return np.lexsort((np.asarray(ids, dtype=str), -array))[:k]


def _topk_canonical(scores: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k for already validated, canonically ID-sorted engine rows.

    Partition only finds a score boundary. Include strictly better rows and the
    first IDs tied at that boundary, then stably sort the small selected set.
    Unlike a bare argpartition, this preserves deterministic ties at k.
    """
    if k == len(scores):
        return np.argsort(-scores, kind="stable")
    boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
    better = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)[:k - len(better)]
    candidates = np.concatenate((better, tied))
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def normalize_vectors(vectors: np.ndarray | Sequence, expected_dimensions: int | None = None) -> np.ndarray:
    """Validate shape/finite/nonzero values and return float32 unit vectors."""
    source = np.asarray(vectors)
    if np.iscomplexobj(source) or not np.issubdtype(source.dtype, np.number):
        raise RetrievalError("vectors must contain real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(source, dtype=np.float32)
    if array.ndim not in (1, 2) or any(dimension == 0 for dimension in array.shape):
        raise RetrievalError("vectors must be a nonempty vector or two-dimensional matrix")
    if expected_dimensions is not None and array.shape[-1] != expected_dimensions:
        raise RetrievalError(f"vector dimensions {array.shape[-1]} do not match expected {expected_dimensions}")
    if not np.all(np.isfinite(array)):
        raise RetrievalError("vectors contain non-finite values or float32 overflow")
    # Float64 norm computation prevents overflow/underflow from finite float32
    # inputs; the stored vectors and dot products remain float32.
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise RetrievalError("zero vectors cannot be normalized")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def dense_scores(normalized_matrix: np.ndarray, query_vector: np.ndarray | Sequence[float]) -> np.ndarray:
    """Score a previously validated unit-row matrix with a normalized query.

    Matrix validation belongs to release loading/engine construction, avoiding
    another full matrix scan on every warm query.
    """
    if normalized_matrix.ndim != 2 or normalized_matrix.shape[0] == 0:
        raise RetrievalError("catalog vectors must be a nonempty matrix")
    query = normalize_vectors(query_vector, normalized_matrix.shape[1])
    if query.ndim != 1:
        raise RetrievalError("query vector must be one-dimensional")
    scores = np.asarray(normalized_matrix @ query, dtype=np.float32)
    if not np.all(np.isfinite(scores)):
        raise RetrievalError("dense scoring produced non-finite values")
    return scores


def rrf_scores(
    ids: Sequence[str],
    dense: Sequence[float] | np.ndarray,
    lexical: Sequence[float] | np.ndarray,
    *,
    rank_constant: int = 60,
    depth: int = 100,
) -> np.ndarray:
    """Equal branch weights and one-based ranks; zero lexical scores abstain."""
    _positive_integer(rank_constant, "rank_constant")
    _positive_integer(depth, "depth")
    dense = _validated_scores(ids, dense)
    lexical = _validated_scores(ids, lexical)
    if np.any(lexical < 0):
        raise RetrievalError("Lucene lexical scores must be nonnegative")
    dense_order = stable_rank(ids, dense, min(depth, len(ids)))
    lexical_order = stable_rank(ids, lexical)
    lexical_order = lexical_order[lexical[lexical_order] > 0][:depth]
    scores = np.zeros(len(ids), dtype=np.float64)
    for order in (dense_order, lexical_order):
        scores[order] += 1.0 / (rank_constant + np.arange(1, len(order) + 1, dtype=np.float64))
    return scores


class LexicalIndex:
    """BM25S Lucene with our shared, deterministic Portuguese token view."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise RetrievalError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        if not corpus_tokens:
            raise RetrievalError("lexical corpus must not be empty")
        tokens = [list(document) for document in corpus_tokens]
        if any(not isinstance(token, str) or not token for document in tokens for token in document):
            raise RetrievalError("lexical tokens must be nonempty strings")
        self.size = len(tokens)
        self._index = None
        vocabulary = {token: index for index, token in enumerate(sorted({token for document in tokens for token in document}))}
        if not vocabulary:
            return
        try:
            import bm25s
        except ImportError as exc:
            raise RetrievalError("lexical/hybrid retrieval requires the pinned bm25s dependency") from exc
        self._index = bm25s.BM25(k1=k1, b=b, method="lucene", backend="numpy", csc_backend="numpy", dtype="float32")
        token_ids = [[vocabulary[token] for token in document] for document in tokens]
        # Explicit sorted vocabulary removes Python set/hash ordering from the
        # sparse-index construction. We handle empty/OOV queries ourselves.
        self._index.index((token_ids, vocabulary), create_empty_token=False, show_progress=False)

    def score(self, query: str | Sequence[str]) -> np.ndarray:
        tokens = list(lexical_tokens(query) if isinstance(query, str) else query)
        if any(not isinstance(token, str) for token in tokens):
            raise RetrievalError("query tokens must be strings")
        if self._index is None or not tokens:
            return np.zeros(self.size, dtype=np.float32)
        known_tokens = [token for token in tokens if token in self._index.vocab_dict]
        if not known_tokens:
            return np.zeros(self.size, dtype=np.float32)
        scores = np.asarray(self._index.get_scores(known_tokens), dtype=np.float32)
        if scores.shape != (self.size,) or not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise RetrievalError("BM25 returned invalid scores")
        return scores


class SearchEngine:
    """Loaded in-memory catalog; lexical index is created only when requested."""

    def __init__(
        self,
        prepared: PreparedCatalog,
        vectors: np.ndarray | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        rrf_depth: int = 100,
    ):
        # A frozen dataclass alone does not freeze its nested metadata. Detach
        # the caller's JSON once so names/evidence cannot drift from vectors.
        self.prepared = replace(prepared, items=tuple(
            replace(item, record=replace(item.record, metadata=freeze_json(item.record.metadata)))
            for item in prepared.items))
        self.ids = tuple(item.record.item_id for item in self.prepared.items)
        if not self.ids or self.ids != tuple(sorted(set(self.ids))):
            raise RetrievalError("prepared item IDs must be unique and in canonical sorted order")
        _positive_integer(rrf_k, "rrf_k")
        _positive_integer(rrf_depth, "rrf_depth")
        self.vectors = None
        if vectors is not None:
            matrix = normalize_vectors(vectors)
            if matrix.ndim != 2 or matrix.shape[0] != len(self.ids):
                raise RetrievalError("catalog vectors and item IDs are not aligned")
            # A writable owning array can undo setflags(write=False). A bytes
            # backing keeps this snapshot read-only, including through views.
            self.vectors = np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)
        self.k1, self.b, self.rrf_k, self.rrf_depth = k1, b, rrf_k, rrf_depth
        self._lexical: LexicalIndex | None = None
        self._lexical_lock = Lock()

    def _ranks(self, scores: np.ndarray, *, depth: int | None = None,
               positive_only: bool = False, eligible_rows: np.ndarray | None = None) -> np.ndarray:
        if eligible_rows is None:
            order = _topk_canonical(scores, min(depth or len(self.ids), len(self.ids)))
        elif len(eligible_rows):
            order = eligible_rows[_topk_canonical(scores[eligible_rows], min(depth or len(eligible_rows), len(eligible_rows)))]
        else:
            order = eligible_rows
        if positive_only:
            order = order[scores[order] > 0]
        ranks = np.zeros(len(self.ids), dtype=np.int32)
        ranks[order] = np.arange(1, len(order) + 1)
        return ranks

    def _eligible_rows(self, eligible_ids: Collection[str] | None) -> np.ndarray | None:
        if eligible_ids is None:
            return None
        if isinstance(eligible_ids, str):
            raise RetrievalError("Eligible IDs must be a collection, not text")
        try:
            allowed = frozenset(eligible_ids)
        except TypeError:
            raise RetrievalError("Eligible IDs must be a collection of item IDs") from None
        if not allowed.issubset(self.ids):
            raise RetrievalError("Eligible IDs are absent from this catalog")
        # Canonical order preserves ID tie-breaking within the eligible universe.
        return np.fromiter((i for i, item_id in enumerate(self.ids) if item_id in allowed), dtype=np.int64)

    def score(self, query: str, query_vector: np.ndarray | None = None, mode: str = "semantic",
              *, include_ranks: bool = True, eligible_ids: Collection[str] | None = None) -> ScoreBundle:
        if mode not in {"semantic", "lexical", "hybrid"}:
            raise RetrievalError(f"unsupported retrieval mode {mode!r}")
        if not normalize_text(query):
            raise RetrievalError("query must not be blank")
        eligible_rows = self._eligible_rows(eligible_ids)
        candidate_count = len(self.ids) if eligible_rows is None else len(eligible_rows)
        dense = lexical = None
        if mode in {"semantic", "hybrid"}:
            if self.vectors is None or query_vector is None:
                raise RetrievalError("semantic/hybrid retrieval requires catalog and query vectors; no fallback")
            dense = dense_scores(self.vectors, query_vector)
        if mode in {"lexical", "hybrid"}:
            if self._lexical is None:
                with self._lexical_lock:
                    if self._lexical is None:
                        self._lexical = LexicalIndex([item.lexical_tokens for item in self.prepared.items], k1=self.k1, b=self.b)
            lexical = self._lexical.score(query)
        if mode == "semantic":
            return ScoreBundle(dense, "cosine", dense_scores=dense,
                               dense_ranks=self._ranks(dense, eligible_rows=eligible_rows) if include_ranks else None,
                               candidate_count=candidate_count, eligible_rows=eligible_rows)
        if mode == "lexical":
            return ScoreBundle(lexical, "bm25_lucene", lexical_scores=lexical,
                               lexical_ranks=self._ranks(lexical, eligible_rows=eligible_rows) if include_ranks else None,
                               candidate_count=candidate_count, eligible_rows=eligible_rows)
        dense_ranks = self._ranks(dense, depth=self.rrf_depth, eligible_rows=eligible_rows)
        lexical_ranks = self._ranks(lexical, positive_only=True, depth=self.rrf_depth, eligible_rows=eligible_rows)
        scores = np.zeros(len(self.ids), dtype=np.float64)
        for ranks in (dense_ranks, lexical_ranks):
            present = ranks > 0
            scores[present] += 1.0 / (self.rrf_k + ranks[present].astype(np.float64))
        return ScoreBundle(
            scores, "rrf", dense, lexical,
            dense_ranks, lexical_ranks,
            int(np.count_nonzero(scores)), eligible_rows,
        )

    def search(
        self, query: str, query_vector: np.ndarray | None = None, mode: str = "semantic", k: int = 5,
        *, eligible_ids: Collection[str] | None = None,
    ) -> tuple[RankedItem, ...]:
        _positive_integer(k, "k")
        if k > len(self.ids):
            raise RetrievalError(f"k={k} exceeds catalog size {len(self.ids)}")
        bundle = self.score(query, query_vector, mode, include_ranks=False, eligible_ids=eligible_ids)
        if k > bundle.candidate_count and eligible_ids is None:
            raise RetrievalError(f"k={k} exceeds {bundle.candidate_count} candidates at fixed RRF depth {self.rrf_depth}")
        if bundle.eligible_rows is None:
            order = _topk_canonical(bundle.scores, k)
        else:
            count = min(k, bundle.candidate_count)
            if count == 0:
                return ()
            candidates = bundle.eligible_rows
            if mode == "hybrid":
                candidates = candidates[bundle.scores[candidates] > 0]
            order = candidates[_topk_canonical(bundle.scores[candidates], count)]
        results = []
        for rank, index in enumerate(order, start=1):
            record = self.prepared.items[index].record
            dense_rank = None if bundle.dense_ranks is None else int(bundle.dense_ranks[index]) or None
            lexical_rank = None if bundle.lexical_ranks is None else int(bundle.lexical_ranks[index]) or None
            if mode == "semantic":
                dense_rank = rank
            elif mode == "lexical":
                lexical_rank = rank
            results.append(RankedItem(
                record.item_id, record.merchant_id, rank, float(bundle.scores[index]), bundle.score_kind,
                record.metadata["name"], record.metadata.get("description", ""),
                None if bundle.dense_scores is None else float(bundle.dense_scores[index]),
                None if bundle.lexical_scores is None else float(bundle.lexical_scores[index]),
                dense_rank, lexical_rank,
                None if bundle.lexical_scores is None else bool(bundle.lexical_scores[index] > 0),
            ))
        return tuple(results)
