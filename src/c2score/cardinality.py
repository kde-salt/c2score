"""Cardinality intervals and their similarity (D_card).

An interval is a pair ``(lower, upper)`` with ``lower`` in ``{0, 1}`` and
``upper`` in ``{1, "*"}``. Only the four intervals ``0..1``, ``0..*``,
``1..1`` and ``1..*`` exist; observed degree bounds are always rounded to one
of them.
"""

from __future__ import annotations

from typing import NamedTuple, Union

STAR = "*"

Bound = Union[int, str]
Interval = tuple[int, Bound]

ZERO_ONE: Interval = (0, 1)
ZERO_MANY: Interval = (0, STAR)
ONE_ONE: Interval = (1, 1)
ONE_MANY: Interval = (1, STAR)

_BY_NAME: dict[str, Interval] = {
    "0..1": ZERO_ONE,
    "0..*": ZERO_MANY,
    "1..1": ONE_ONE,
    "1..*": ONE_MANY,
}
_NAME_OF: dict[Interval, str] = {interval: name for name, interval in _BY_NAME.items()}


class Cardinality(NamedTuple):
    """The out-degree and in-degree intervals of an edge type."""

    out: Interval
    in_: Interval

    def __str__(self) -> str:
        return f"out {format_interval(self.out)} / in {format_interval(self.in_)}"


#: The cardinality assumed for an edge type that has none.
DEFAULT_CARDINALITY = Cardinality(ZERO_MANY, ZERO_MANY)


def parse_interval(text: str) -> Interval:
    """Parse one of ``"0..1"``, ``"0..*"``, ``"1..1"``, ``"1..*"``."""
    try:
        return _BY_NAME[text]
    except (KeyError, TypeError):
        allowed = ", ".join(repr(name) for name in _BY_NAME)
        raise ValueError(f"invalid interval {text!r}; expected one of {allowed}") from None


def format_interval(interval: Interval) -> str:
    """Return the text form of one of the four intervals."""
    try:
        return _NAME_OF[tuple(interval)]
    except (KeyError, TypeError):
        raise ValueError(f"not one of the four intervals: {interval!r}") from None


def round_observed(min_degree: int, max_degree: int) -> Interval:
    """Round observed degree bounds ``[min, max]`` to one of the four intervals.

    The lower bound becomes 0 if ``min`` is 0 and 1 otherwise; the upper bound
    becomes 1 if ``max`` is at most 1 and ``"*"`` otherwise.
    """
    if min_degree < 0 or max_degree < min_degree or max_degree < 1:
        raise ValueError(f"invalid observed degree bounds [{min_degree}, {max_degree}]")
    lower = 0 if min_degree == 0 else 1
    upper: Bound = 1 if max_degree <= 1 else STAR
    return (lower, upper)


def bd(x: Bound, y: Bound) -> float:
    """Similarity of two bounds (Eq. 33)."""
    x_star = x == STAR
    y_star = y == STAR
    if x_star and y_star:
        return 1.0
    if x_star or y_star:
        return 0.0
    return (min(x, y) + 1) / (max(x, y) + 1)


def d_int(a: Interval, b: Interval) -> float:
    """Similarity of two intervals (Eq. 32)."""
    return (bd(a[0], b[0]) + bd(a[1], b[1])) / 2


def d_card(c: Cardinality, c2: Cardinality) -> float:
    """Similarity of two cardinalities (Eq. 31)."""
    return (d_int(c.out, c2.out) + d_int(c.in_, c2.in_)) / 2
