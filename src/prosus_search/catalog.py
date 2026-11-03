"""Strict ingestion of the supplied catalog and query snapshot.

The raw CSV hash identifies a dataset version. Duplicate offers collapse only
when their parsed metadata, profile and merchant agree; row provenance survives.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any


class CatalogError(ValueError):
    """Malformed or internally conflicting input, with source context."""


@dataclass(frozen=True)
class CatalogRecord:
    item_id: str
    merchant_id: str
    metadata: dict[str, Any]
    source_ids: tuple[str, ...]
    source_lines: tuple[int, ...]
    profile_hash: str = ""
    historical_options: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Catalog:
    records: tuple[CatalogRecord, ...]
    source_hash: str
    raw_count: int


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def canonical_json(value: Any) -> str:
    """Stable JSON for fingerprints; rejects non-finite numeric values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    return hash_bytes(Path(path).read_bytes())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _parse_object(value: str, context: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        # A numeric exponent such as 1e999 bypasses parse_constant.
        canonical_json(parsed)
    except (ValueError, TypeError) as exc:
        raise CatalogError(f"{context}: invalid JSON ({exc})") from exc
    if not isinstance(parsed, dict):
        raise CatalogError(f"{context}: expected a JSON object")
    return parsed


def _read_rows(path: str | Path, required: set[str]) -> tuple[bytes, list[tuple[int, dict[str, str]]]]:
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise CatalogError(f"{path.name}: invalid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    try:
        header = reader.fieldnames
        if not header or len(set(header)) != len(header):
            raise CatalogError(f"{path.name}: missing or duplicate CSV header")
        missing = required - set(header)
        if missing:
            raise CatalogError(f"{path.name}: missing columns {sorted(missing)}")
        rows = []
        while True:
            start_line = reader.line_num + 1
            try:
                row = next(reader)
            except StopIteration:
                break
            if None in row or any(value is None for value in row.values()):
                raise CatalogError(f"{path.name}:{start_line}: row width does not match CSV header")
            rows.append((start_line, row))
    except csv.Error as exc:
        raise CatalogError(f"{path.name}:{reader.line_num}: invalid CSV ({exc})") from exc
    if not rows:
        raise CatalogError(f"{path.name}: no data rows")
    return raw, rows


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{context}: expected a nonblank string")
    if value != value.strip():
        raise CatalogError(f"{context}: identifier has surrounding whitespace")
    return value


def _validate_metadata(metadata: dict[str, Any], context: str) -> None:
    if not isinstance(metadata.get("name"), str) or not metadata["name"].strip():
        raise CatalogError(f"{context}.name: expected a nonblank string")
    for key in ("category_name", "description"):
        if key in metadata and not isinstance(metadata[key], str):
            raise CatalogError(f"{context}.{key}: expected a string")
    for key in ("vegan", "lacFree", "organic"):
        if key in metadata and not isinstance(metadata[key], bool):
            raise CatalogError(f"{context}.{key}: expected a Boolean")
    if "price" in metadata:
        price = metadata["price"]
        if isinstance(price, bool) or not isinstance(price, (float, int)) or not math.isfinite(price):
            raise CatalogError(f"{context}.price: expected a finite number")
    if "images" in metadata:
        images = metadata["images"]
        if not isinstance(images, list) or any(not isinstance(value, str) for value in images):
            raise CatalogError(f"{context}.images: expected a list of strings")
    if "taxonomy" in metadata:
        taxonomy = metadata["taxonomy"]
        if not isinstance(taxonomy, dict) or any(not isinstance(value, str) for value in taxonomy.values()):
            raise CatalogError(f"{context}.taxonomy: expected an object with string values")
    if "tags" in metadata:
        tags = metadata["tags"]
        if not isinstance(tags, list):
            raise CatalogError(f"{context}.tags: expected a list")
        for index, tag in enumerate(tags):
            if not isinstance(tag, dict) or not isinstance(tag.get("key"), str):
                raise CatalogError(f"{context}.tags[{index}]: expected key/value object")
            values = tag.get("value")
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise CatalogError(f"{context}.tags[{index}].value: expected a list of strings")


def load_catalog(path: str | Path) -> Catalog:
    """Load canonical offers, failing on malformed rows or conflicting IDs."""
    path = Path(path)
    raw, rows = _read_rows(path, {"_id", "itemId", "itemMetadata", "itemProfile", "merchantId"})
    records: dict[str, CatalogRecord] = {}
    signatures: dict[str, str] = {}
    for line, row in rows:
        context = f"{path.name}:{line}"
        item_id = _identifier(row["itemId"], f"{context}.itemId")
        merchant_id = _identifier(row["merchantId"], f"{context}.merchantId")
        source_id = _identifier(row["_id"], f"{context}._id")
        metadata = _parse_object(row["itemMetadata"], f"{context}.itemMetadata")
        profile = _parse_object(row["itemProfile"], f"{context}.itemProfile")
        _validate_metadata(metadata, f"{context}.itemMetadata")
        signature = canonical_json([merchant_id, metadata, profile])
        if item_id in records:
            previous = records[item_id]
            if signatures[item_id] != signature:
                raise CatalogError(
                    f"{context}: conflicting duplicate itemId {item_id!r}; first at line {previous.source_lines[0]}"
                )
            records[item_id] = replace(
                previous, source_ids=previous.source_ids + (source_id,), source_lines=previous.source_lines + (line,)
            )
        else:
            records[item_id] = CatalogRecord(
                item_id, merchant_id, metadata, (source_id,), (line,), hash_bytes(canonical_json(profile).encode("utf-8")),
                tuple(profile.get("metrics", {}).get("customizations", [])),
            )
            signatures[item_id] = signature
    return Catalog(tuple(records[item_id] for item_id in sorted(records)), hash_bytes(raw), len(rows))


def load_queries(path: str | Path) -> tuple[Query, ...]:
    """Preserve original wording and assign IDs from source order, not content."""
    path = Path(path)
    _, rows = _read_rows(path, {"search_term_pt"})
    queries = []
    for index, (line, row) in enumerate(rows, start=1):
        text = row["search_term_pt"]
        if not text.strip():
            raise CatalogError(f"{path.name}:{line}.search_term_pt: expected a nonblank query")
        queries.append(Query(f"q{index:04d}", text))
    return tuple(queries)
