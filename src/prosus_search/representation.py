"""One source-backed product representation shared by every retrieval run."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Callable, Iterable

from .catalog import Catalog, CatalogRecord, canonical_json, hash_bytes


class RepresentationError(ValueError):
    """Input cannot be represented under the declared comparison settings."""


@dataclass(frozen=True)
class PortionClaim:
    people: int
    field_path: str
    raw_value: str


@dataclass(frozen=True)
class PreparedItem:
    record: CatalogRecord
    text: str
    text_hash: str
    token_count: int
    lexical_tokens: tuple[str, ...]


@dataclass(frozen=True)
class PreparedCatalog:
    items: tuple[PreparedItem, ...]
    source_hash: str
    representation_version: str
    prepared_hash: str


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise RepresentationError("text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Fold accents/case only for lexical matching; retain negation and digits."""
    decomposed = unicodedata.normalize("NFD", normalize_text(text).casefold())
    folded = "".join(character for character in decomposed if not unicodedata.combining(character))
    return tuple(re.findall(r"[^\W_]+", folded, flags=re.UNICODE))


def portion_claims(record: CatalogRecord) -> tuple[PortionClaim, ...]:
    """Expose every recognized claim with its original JSON field path.

    Conflicting claims remain distinct. NOT_APPLICABLE and missing tags do not
    imply a one-person portion or any unobserved suitability attribute.
    """
    claims = []
    for tag_index, tag in enumerate(record.metadata.get("tags", [])):
        if tag["key"] != "PORTION_SIZE":
            continue
        for value_index, value in enumerate(tag["value"]):
            if value in {"SERVES_1", "SERVES_2", "SERVES_3", "SERVES_4"}:
                claims.append(PortionClaim(int(value[-1]), f"tags[{tag_index}].value[{value_index}]", value))
    return tuple(claims)


def item_text(record: CatalogRecord) -> str:
    metadata = record.metadata
    parts = [f"Nome: {normalize_text(metadata['name'])}"]
    category = normalize_text(metadata.get("category_name", ""))
    if category:
        parts.append(f"Categoria: {category}")
    taxonomy = metadata.get("taxonomy", {})
    for level in ("l0", "l1", "l2"):
        value = normalize_text(taxonomy.get(level, "").replace("_", " "))
        if value:
            parts.append(f"Taxonomia {level}: {value}")
    description = normalize_text(metadata.get("description", ""))
    if description:
        parts.append(f"Descrição: {description}")
    # Deduplicate equal claims without combining distinct counts into a range.
    for people in sorted({claim.people for claim in portion_claims(record)}):
        noun = "pessoa" if people == 1 else "pessoas"
        parts.append(f"Porção declarada no catálogo: serve {people} {noun}")
    return "\n".join(parts)


def _count(text: str, counter: Callable[[str], int], cap: int, context: str) -> int:
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise RepresentationError("token cap must be a positive integer")
    count = counter(text)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise RepresentationError(f"{context}: tokenizer must return a positive integer count")
    if count > cap:
        raise RepresentationError(f"{context}: {count} tokens exceeds cap {cap}; no text was truncated")
    return count


def prepare_query(text: str, token_counter: Callable[[str], int], query_token_cap: int = 128) -> str:
    prepared = normalize_text(text)
    if not prepared:
        raise RepresentationError("query must not be blank")
    _count(prepared, token_counter, query_token_cap, "query")
    return prepared


def prepare_catalog(
    catalog: Catalog,
    token_counter: Callable[[str], int],
    item_token_cap: int = 512,
    representation_version: str = "catalog-v1-portion",
) -> PreparedCatalog:
    if not isinstance(representation_version, str) or not representation_version.strip():
        raise RepresentationError("representation_version must not be blank")
    if not catalog.records:
        raise RepresentationError("catalog must contain at least one item")
    ids = tuple(record.item_id for record in catalog.records)
    if ids != tuple(sorted(set(ids))):
        raise RepresentationError("catalog IDs must be unique and in canonical sorted order")
    items = []
    for record in catalog.records:
        text = item_text(record)
        count = _count(text, token_counter, item_token_cap, f"item {record.item_id}")
        items.append(PreparedItem(record, text, hash_bytes(text.encode("utf-8")), count, lexical_tokens(text)))
    fingerprint = {
        "source_hash": catalog.source_hash,
        "representation_version": representation_version,
        "items": [[item.record.item_id, item.text_hash] for item in items],
    }
    return PreparedCatalog(
        tuple(items), catalog.source_hash, representation_version, hash_bytes(canonical_json(fingerprint).encode("utf-8"))
    )


def prepared_catalog_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source_hash: str,
    prepared_hash: str,
    representation_version: str,
) -> PreparedCatalog:
    """Restore checked release rows without tokenizer downloads or API access.

    Artifact checksums are verified by the release loader. These additional
    checks protect ID/text/token alignment when using the library directly.
    """
    items = []
    try:
        for row in rows:
            record = CatalogRecord(
                row["item_id"], row["merchant_id"], row["metadata"],
                tuple(row["source_ids"]), tuple(row["source_lines"]), row.get("profile_hash", ""),
            )
            text = row["text"]
            text_hash = hash_bytes(text.encode("utf-8"))
            tokens = tuple(row["lexical_tokens"])
            if text_hash != row["text_hash"] or tokens != lexical_tokens(text):
                raise RepresentationError(f"item {record.item_id}: release text/hash/tokens disagree")
            if text != item_text(record):
                raise RepresentationError(f"item {record.item_id}: release representation disagrees with source metadata")
            count = row["token_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise RepresentationError(f"item {record.item_id}: invalid stored token count")
            items.append(PreparedItem(record, text, text_hash, count, tokens))
    except (KeyError, TypeError, AttributeError) as exc:
        raise RepresentationError("malformed prepared release row") from exc
    ids = tuple(item.record.item_id for item in items)
    if not ids or ids != tuple(sorted(set(ids))):
        raise RepresentationError("release item IDs must be nonempty, unique and canonically ordered")
    fingerprint = {
        "source_hash": source_hash,
        "representation_version": representation_version,
        "items": [[item.record.item_id, item.text_hash] for item in items],
    }
    actual_hash = hash_bytes(canonical_json(fingerprint).encode("utf-8"))
    if actual_hash != prepared_hash:
        raise RepresentationError("prepared catalog fingerprint does not match release manifest")
    return PreparedCatalog(tuple(items), source_hash, representation_version, prepared_hash)
