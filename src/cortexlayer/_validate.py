"""Argument validation shared by ``CortexClient`` and ``Memory`` so both
surfaces accept and reject exactly the same inputs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .errors import InvalidRequestError

_TAG_SCALARS = (str, int, float, bool)


def need_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequestError(f"{name} must be a non-empty string")
    return value


def need_int(value: Any, name: str, lo: int, hi: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRequestError(f"{name} must be an integer")
    if value < lo or (hi is not None and value > hi):
        rng = f">= {lo}" if hi is None else f"{lo}..{hi}"
        raise InvalidRequestError(f"{name} must be {rng}")
    return value


def need_tags(value: Any, name: str) -> Optional[Dict[str, Any]]:
    """``None``, or a flat ``dict[str, str | int | float | bool]`` — provenance
    metadata (task 0090), never a filter. Rejected outright rather than
    silently coerced: nested structures, non-scalar values, non-string keys.
    An empty dict is treated the same as ``None`` (nothing to store)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidRequestError(f"{name} must be a dict or None")
    for k, v in value.items():
        if not isinstance(k, str) or not k:
            raise InvalidRequestError(f"{name} keys must be non-empty strings")
        if not isinstance(v, _TAG_SCALARS):
            raise InvalidRequestError(f"{name}[{k!r}] must be a str, int, float, or bool")
    return value or None
