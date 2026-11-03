"""Detached, read-only JSON containers for loaded serving snapshots.

Keep JSON serialization and dict/list equality unchanged. These protect normal
Python container operations, like frozen dataclasses protect field assignment;
they are not a security boundary against deliberate base-class mutation.
"""
from __future__ import annotations

from typing import Any


def _immutable(*args, **kwargs):
    raise TypeError("Loaded snapshot data is immutable; build a new snapshot")


class _Frozen:
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _FrozenDict(_Frozen, dict):
    __setitem__ = __delitem__ = __ior__ = _immutable
    clear = pop = popitem = setdefault = update = _immutable


class _FrozenList(_Frozen, list):
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _immutable
    append = clear = extend = insert = pop = remove = reverse = sort = _immutable


def freeze_json(value: Any) -> Any:
    """Copy a JSON tree once, recursively preventing subsequent mutation."""
    if isinstance(value, (_FrozenDict, _FrozenList)):
        return value
    if isinstance(value, dict):
        return _FrozenDict((key, freeze_json(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenList(freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Snapshot values must contain only JSON-compatible data")
