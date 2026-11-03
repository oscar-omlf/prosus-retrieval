"""Immutable, expiring allowlists supplied by an authoritative eligibility caller.

This contract does not infer inventory, delivery area, merchant state or dietary
safety from catalog text. The caller must obtain the correct list for its request
context and must not reuse an expired list, including during catalog rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Collection


class EligibilityError(ValueError):
    """Eligibility cannot be established for this release/request."""


@dataclass(frozen=True)
class EligibilitySnapshot:
    release_id: str
    revision: str
    context_id: str
    item_ids: frozenset[str]
    valid_until: float

    def __post_init__(self):
        if not isinstance(self.release_id, str) or not re.fullmatch(r"[0-9a-f]{64}", self.release_id):
            raise EligibilityError("Eligibility requires a complete catalog release ID")
        for value in (self.revision, self.context_id):
            if not isinstance(value, str) or not value.strip():
                raise EligibilityError("Eligibility revision and context must be nonblank")
        if isinstance(self.item_ids, str):
            raise EligibilityError("Eligible item IDs must be a collection, not text")
        try:
            ids = frozenset(self.item_ids)
        except TypeError:
            raise EligibilityError("Eligible item IDs must be a collection of strings") from None
        if any(not isinstance(item_id, str) or not item_id for item_id in ids):
            raise EligibilityError("Eligible item IDs must be nonblank strings")
        object.__setattr__(self, "item_ids", ids)
        if type(self.valid_until) not in (int, float) or not math.isfinite(self.valid_until):
            raise EligibilityError("Eligibility expiry must be a finite UTC Unix timestamp")

    def assert_fresh(self) -> None:
        if time.time() >= self.valid_until:
            raise EligibilityError("Eligibility has expired; obtain a fresh authoritative snapshot")

    def validate_for(self, release_id: str | None, catalog_ids: Collection[str]) -> frozenset[str]:
        self.assert_fresh()
        if release_id != self.release_id:
            raise EligibilityError("Eligibility belongs to a different catalog release")
        if not self.item_ids.issubset(catalog_ids):
            raise EligibilityError("Eligibility contains IDs absent from the catalog release")
        return self.item_ids
