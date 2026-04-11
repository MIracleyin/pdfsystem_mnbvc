"""Generic, stdlib-only dataclass ↔ JSON-compatible dict conversion.

Used by :class:`pdfsys_core.cache.LayoutCache` and anywhere else a pdfsys
dataclass needs to roundtrip through JSON without pulling in pydantic or
msgspec. Supports:

* Nested dataclasses (including frozen + slotted).
* ``Enum`` subclasses (serialized by ``.value``).
* ``tuple[T, ...]``, ``list[T]``, ``dict[K, V]`` generics.
* ``Optional[T]`` / ``T | None`` unions.
* Primitives (str, int, float, bool, None) and ``Any`` pass-through.

The conversion functions are intentionally lenient: unknown fields in the
input are ignored on ``from_dict``, which lets us evolve the schema forward
without breaking old cache files.
"""

from __future__ import annotations

import dataclasses
import enum
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints


def to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass tree to JSON-compatible primitives."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (tuple, list)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    return obj


def from_dict(cls: Any, data: Any) -> Any:
    """Recursively reconstruct a typed object from JSON-compatible data.

    ``cls`` is inspected with :func:`typing.get_type_hints` so forward
    references inside a module using ``from __future__ import annotations``
    are resolved correctly.
    """
    if data is None:
        return None

    # Enum — dispatch by value.
    if isinstance(cls, type) and issubclass(cls, enum.Enum):
        return cls(data)

    origin = get_origin(cls)

    # Union / Optional — try each non-None member until one decodes.
    if origin is Union or origin is UnionType:
        args = [a for a in get_args(cls) if a is not NoneType]
        for arg in args:
            try:
                return from_dict(arg, data)
            except Exception:
                continue
        raise ValueError(f"Cannot decode {data!r} as any of {args}")

    # tuple[X, ...] or tuple[X, Y, Z]
    if origin is tuple:
        args = get_args(cls)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_dict(args[0], v) for v in data)
        return tuple(from_dict(a, v) for a, v in zip(args, data))

    # list[X]
    if origin is list:
        (arg,) = get_args(cls)
        return [from_dict(arg, v) for v in data]

    # dict[K, V]
    if origin is dict:
        k_type, v_type = get_args(cls)
        return {from_dict(k_type, k): from_dict(v_type, v) for k, v in data.items()}

    # Dataclass — recurse through fields.
    if isinstance(cls, type) and dataclasses.is_dataclass(cls):
        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in data:
                kwargs[f.name] = from_dict(hints.get(f.name, Any), data[f.name])
        return cls(**kwargs)

    # Primitive, Any, or unknown — pass through unchanged.
    return data
