"""Coverage: how well the schema covers the instance types (Section 4.7, Eqs. 35-36)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np

from .flatten import FlatSchema
from .similarity import Similarity

_T = TypeVar("_T")
_S = TypeVar("_S")


@dataclass(frozen=True)
class Match(Generic[_T, _S]):
    """The best schema type for one instance type; ``schema_type`` is ``None`` if every similarity is 0."""

    instance_type: _T
    schema_type: Optional[_S]
    similarity: float


def _coverage(
    instance_types: Sequence[_T], candidates: Sequence[_S], scores: Callable[[_S], np.ndarray]
) -> tuple[float, list[Match[_T, _S]]]:
    """The mean best similarity. An empty instance side is fully covered."""
    if not instance_types:
        return 1.0, []
    if not candidates:
        return 0.0, [Match(t, None, 0.0) for t in instance_types]
    # One column per candidate. argmax returns the first maximum, so among
    # equal similarities the candidate that comes first in order wins.
    matrix = np.column_stack([scores(c) for c in candidates])
    best = matrix.argmax(axis=1)
    values = matrix[np.arange(len(instance_types)), best].tolist()
    matches = [
        Match(t, candidates[i] if value > 0 else None, value)
        for t, i, value in zip(instance_types, best.tolist(), values)
    ]
    return sum(values) / len(instance_types), matches


def cov_v(flat: FlatSchema, similarity: Similarity) -> tuple[float, list[Match]]:
    """Cov_V and the best match of each instance node type."""
    return _coverage(similarity.instance.node_types, flat.node_types, similarity.node_scores)


def cov_e(flat: FlatSchema, similarity: Similarity) -> tuple[float, list[Match]]:
    """Cov_E and the best match of each instance edge type."""
    return _coverage(similarity.instance.edge_types, flat.edge_types, similarity.edge_scores)
