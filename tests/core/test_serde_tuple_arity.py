"""A fixed-length tuple type says how many elements there are.

``from_dict`` decoded ``tuple[X, Y, Z]`` with a bare ``zip``, which stops at
the shorter side. So a ``Block.bbox`` — ``tuple[float, float, float, float]``
— read back from three values became a 3-tuple, and nothing anywhere said so:
no exception, no warning, just a rectangle missing an edge written into the
shard. Found by ruff's B905 while clearing the lint backlog.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pdfsys_core.serde import from_dict


@dataclass
class _HasBbox:
    bbox: tuple[float, float, float, float]


def test_a_full_bbox_round_trips():
    assert from_dict(_HasBbox, {"bbox": [1.0, 2.0, 3.0, 4.0]}).bbox == (1.0, 2.0, 3.0, 4.0)


@pytest.mark.parametrize("values", [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
def test_a_bbox_of_the_wrong_length_is_refused(values):
    """Both directions: zip stopped at the shorter side either way, so a
    five-element bbox was silently truncated to four and looked correct."""
    with pytest.raises(ValueError):
        from_dict(_HasBbox, {"bbox": values})


def test_variable_length_tuples_still_take_any_number():
    """tuple[X, ...] declares no arity, so nothing to check."""

    @dataclass
    class _HasMany:
        xs: tuple[int, ...]

    assert from_dict(_HasMany, {"xs": [1, 2, 3, 4, 5]}).xs == (1, 2, 3, 4, 5)
    assert from_dict(_HasMany, {"xs": []}).xs == ()
