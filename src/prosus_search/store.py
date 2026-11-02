"""Persistent cache and conservative, assignment-wide API cost accounting.

Money is integer micro-USD. A request is reserved before transmission and remains
charged conservatively after a crash or an unknown billing outcome. Cache cleanup
must never delete this ledger. This is a local application guard, not an account
balance service.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping, Sequence
import uuid

import numpy as np


DEFAULT_BUCKET_LIMITS = {
    "embeddings": 500_000,
    "development": 1_400_000,
    "final": 850_000,
    "round4": 1_000_000,
    "recovery": 750_000,
}
DEFAULT_GLOBAL_LIMIT = 4_500_000


class StoreError(RuntimeError):
    """An actionable cache/accounting error without provider payloads."""


class BudgetExceededError(StoreError):
    pass


class WriterBusyError(StoreError):
    pass


class CacheIntegrityError(StoreError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def cost_micro_usd(input_tokens: int, input_rate_per_million: float | str,
                   output_tokens: int = 0, output_rate_per_million: float | str = 0) -> int:
    """Convert USD/million-token tariffs to micro-USD, rounding upward once."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (input_tokens, output_tokens)):
        raise ValueError("Token counts must be nonnegative integers")
    rates = [Decimal(str(input_rate_per_million)), Decimal(str(output_rate_per_million))]
    if any(not rate.is_finite() or rate < 0 for rate in rates):
        raise ValueError("Token tariffs must be finite and nonnegative")
    return int((input_tokens * rates[0] + output_tokens * rates[1]).to_integral_value(rounding=ROUND_CEILING))


def _safe_json(value: Any) -> str:
    """Reject credential-bearing metadata keys, including nested SDK dumps."""
    forbidden = {"api_key", "apikey", "authorization", "access_token", "password", "cookie", "headers", "http_headers"}

    def check(obj: Any) -> None:
        if isinstance(obj, Mapping):
            for key, child in obj.items():
                if str(key).casefold() in forbidden:
                    raise StoreError("Credential/header fields cannot be persisted")
                check(child)
        elif isinstance(obj, (list, tuple)):
            for child in obj:
                check(child)

    check(value)
    return canonical_json(value)


def _validated_vector(vector: Any, dimensions: int | None = None) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or (dimensions is not None and array.shape != (dimensions,)):
        raise CacheIntegrityError("Embedding has an unexpected shape")
    if not len(array) or not np.isfinite(array).all() or float(np.linalg.norm(array.astype(np.float64))) == 0:
        raise CacheIntegrityError("Embedding must be finite and nonzero")
    return np.ascontiguousarray(array, dtype="<f4")


class Store:
    def __init__(self, path: str | Path, bucket_limits: Mapping[str, int] | None = None,
                 global_limit: int = DEFAULT_GLOBAL_LIMIT):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        self.bucket_limits = dict(DEFAULT_BUCKET_LIMITS if bucket_limits is None else bucket_limits)
        self.global_limit = global_limit
        if not self.bucket_limits or any(type(v) is not int or v < 0 for v in self.bucket_limits.values()):
            raise ValueError("Budget limits must be nonnegative integer micro-USD")
        if type(global_limit) is not int or global_limit < 0:
            raise ValueError("Global budget must be nonnegative integer micro-USD")
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL, checksum TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY, dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
                    checksum TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ledger (
                    request_id TEXT PRIMARY KEY, bucket TEXT NOT NULL, reserved INTEGER NOT NULL,
                    actual INTEGER, status TEXT NOT NULL, metadata TEXT NOT NULL, usage TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            """)
            settings = canonical_json({"schema_version": 1, "bucket_limits": self.bucket_limits, "global_limit": global_limit})
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT value FROM settings WHERE key='budget'").fetchone()
            if existing is None:
                connection.execute("INSERT INTO settings VALUES ('budget', ?)", (settings,))
            elif existing[0] != settings:
                raise StoreError("Existing ledger budget differs; use its recorded configuration, never reset the ledger")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def api_writer(self) -> Iterator[None]:
        """One API writer across processes. No SQLite transaction spans the call."""
        with self.lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            acquired = False
            try:
                if os.name == "nt":
                    import msvcrt
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    except OSError:
                        raise WriterBusyError("Another process is using the API writer; retry after it finishes") from None
                else:
                    import fcntl
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise WriterBusyError("Another process is using the API writer; retry after it finishes") from None
                acquired = True
                yield
            finally:
                if acquired:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def lookup(self, key: str) -> Any | None:
        with self._connection() as connection:
            row = connection.execute("SELECT value,checksum FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row["value"].encode("utf-8")).hexdigest() != row["checksum"]:
            raise CacheIntegrityError("Cached JSON checksum mismatch")
        return json.loads(row["value"])

    @staticmethod
    def _put_json(connection: sqlite3.Connection, key: str, value: Any) -> None:
        serialized = _safe_json(value)
        connection.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                           (key, serialized, hashlib.sha256(serialized.encode("utf-8")).hexdigest(), utc_now()))

    def put(self, key: str, value: Any) -> None:
        with self._connection() as connection:
            self._put_json(connection, key, value)

    def get_embedding(self, key: str, dimensions: int) -> np.ndarray | None:
        with self._connection() as connection:
            row = connection.execute("SELECT dimensions,vector,checksum FROM embeddings WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if row["dimensions"] != dimensions or hashlib.sha256(row["vector"]).hexdigest() != row["checksum"]:
            raise CacheIntegrityError("Cached embedding dimensions/checksum mismatch")
        return _validated_vector(np.frombuffer(row["vector"], dtype="<f4").copy(), dimensions)

    @staticmethod
    def _put_embedding(connection: sqlite3.Connection, key: str, vector: Any, metadata: Any) -> None:
        array = _validated_vector(vector)
        blob = array.tobytes()
        connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
                           (key, len(array), blob, hashlib.sha256(blob).hexdigest(), _safe_json(metadata), utc_now()))

    def put_embedding(self, key: str, vector: Any, metadata: Mapping[str, Any] | None = None) -> None:
        with self._connection() as connection:
            self._put_embedding(connection, key, vector, metadata or {})

    @staticmethod
    def _totals(connection: sqlite3.Connection) -> dict[str, int]:
        rows = connection.execute("""SELECT bucket, SUM(CASE
            WHEN status='confirmed' THEN actual WHEN status IN ('pending','uncertain') THEN reserved ELSE 0 END) AS spent
            FROM ledger GROUP BY bucket""").fetchall()
        return {row["bucket"]: int(row["spent"]) for row in rows}

    def reserve(self, bucket: str, amount_micro_usd: int, request_id: str | None = None,
                metadata: Mapping[str, Any] | None = None) -> str:
        if bucket not in self.bucket_limits:
            raise StoreError(f"Unknown budget bucket: {bucket}")
        if type(amount_micro_usd) is not int or amount_micro_usd < 0:
            raise ValueError("Reservation must be nonnegative integer micro-USD")
        serialized = _safe_json(metadata or {})
        request_id = request_id or uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            totals = self._totals(connection)
            if totals.get(bucket, 0) + amount_micro_usd > self.bucket_limits[bucket]:
                raise BudgetExceededError(f"Budget bucket '{bucket}' cannot cover this reservation")
            if sum(totals.values()) + amount_micro_usd > self.global_limit:
                raise BudgetExceededError("The assignment-wide $4.50 spending stop would be exceeded")
            try:
                connection.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?)",
                                   (request_id, bucket, amount_micro_usd, None, "pending", serialized, None, utc_now(), utc_now()))
            except sqlite3.IntegrityError:
                raise StoreError("Request reservation ID already exists; never reuse a charged attempt") from None
        return request_id

    def settle(self, request_id: str, actual_micro_usd: int, usage: Mapping[str, Any] | None = None,
               cache_entries: Mapping[str, Any] | None = None,
               embedding_entries: Sequence[tuple[str, Any, Mapping[str, Any]]] | None = None) -> None:
        """Settle known usage and successful cache entries in the same transaction.

        A provider overrun is recorded truthfully before raising. It is never
        clipped to the reservation to make the displayed balance look safe.
        """
        if type(actual_micro_usd) is not int or actual_micro_usd < 0:
            raise ValueError("Actual charge must be nonnegative integer micro-USD")
        serialized_usage = _safe_json(usage or {})
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM ledger WHERE request_id=?", (request_id,)).fetchone()
            if row is None or row["status"] not in ("pending", "uncertain"):
                raise StoreError("Only an outstanding reservation can be settled")
            connection.execute("UPDATE ledger SET actual=?,status='confirmed',usage=?,updated_at=? WHERE request_id=?",
                               (actual_micro_usd, serialized_usage, utc_now(), request_id))
            for key, value in (cache_entries or {}).items():
                self._put_json(connection, key, value)
            for key, vector, metadata in (embedding_entries or ()):
                self._put_embedding(connection, key, vector, metadata)
            overrun = actual_micro_usd > row["reserved"]
        if overrun:
            raise BudgetExceededError("Provider usage exceeded its reservation; actual cost recorded; stop and review rates/token bounds")

    def mark_uncertain(self, request_id: str) -> None:
        with self._connection() as connection:
            result = connection.execute("UPDATE ledger SET status='uncertain',updated_at=? WHERE request_id=? AND status='pending'",
                                        (utc_now(), request_id))
            if result.rowcount != 1:
                raise StoreError("Only a pending request can become uncertain")

    def release_unsubmitted(self, request_id: str) -> None:
        """Release ONLY when the caller proves that no transmission occurred."""
        with self._connection() as connection:
            result = connection.execute("UPDATE ledger SET status='released',updated_at=? WHERE request_id=? AND status='pending'",
                                        (utc_now(), request_id))
            if result.rowcount != 1:
                raise StoreError("An uncertain/submitted request cannot be released")

    def summary(self) -> dict[str, Any]:
        with self._connection() as connection:
            totals = self._totals(connection)
            rows = connection.execute("SELECT status,COUNT(*) AS count,SUM(CASE WHEN status='confirmed' THEN actual ELSE reserved END) AS amount FROM ledger GROUP BY status").fetchall()
        return {
            "currency": "USD", "unit": "micro_usd", "global_limit": self.global_limit,
            "effective_spend": sum(totals.values()), "remaining": self.global_limit - sum(totals.values()),
            "buckets": {key: {"limit": value, "effective_spend": totals.get(key, 0), "remaining": value - totals.get(key, 0)}
                        for key, value in self.bucket_limits.items()},
            "states": {row["status"]: {"count": row["count"], "amount": row["amount"]} for row in rows},
        }

    def requests(self) -> list[dict[str, Any]]:
        """Sanitized accounting records, useful for an audit or cache resumption."""
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM ledger ORDER BY created_at,request_id").fetchall()
        return [dict(row) | {"metadata": json.loads(row["metadata"]), "usage": json.loads(row["usage"] or "{}")} for row in rows]
