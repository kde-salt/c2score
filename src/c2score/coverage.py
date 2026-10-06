"""Coverage: how well the schema covers the instance types (Section 4.7, Eqs. 35-36).

The node similarities form a dense matrix (instance node types x V(S+)). The
edge similarities are kept sparse: an index over E(S+) finds, for each
instance edge type, the elements whose sim_E can be non-zero (the edge
labels and both endpoints' labels overlap), and only those are computed.
Every other similarity is exactly 0 by Section 4.5, so the row maxima are
the same as over the dense matrix.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Generic, Optional, TypeVar

import numpy as np

from .flatten import FlatEdgeType, FlatSchema
from .similarity import Similarity, _edge_score, dice, label_keys

_T = TypeVar("_T")
_S = TypeVar("_S")


@dataclass(frozen=True)
class Match(Generic[_T, _S]):
    """The best schema type for one instance type; ``schema_type`` is ``None`` if every similarity is 0."""

    instance_type: _T
    schema_type: Optional[_S]
    similarity: float


def _mean(values: list[float]) -> float:
    # The same summation everywhere, so that equal lists give equal coverages.
    return sum(values) / len(values)


class CoverageTables:
    """The similarities of the instance types to one flattened schema, with the best match per row.

    Besides Cov_V and Cov_E, it gives the coverages after some schema types
    are removed, as long as the remaining types are unchanged (Conciseness).
    """

    def __init__(self, flat: FlatSchema, similarity: Similarity) -> None:
        self.flat = flat
        self.similarity = similarity
        self._node_table()
        self._edge_table()

    # -- nodes ---------------------------------------------------------------

    def _node_table(self) -> None:
        instance_types = self.similarity.instance.node_types
        candidates = self.flat.node_types
        self.node_matrix = np.zeros((len(instance_types), len(candidates)))
        self.node_values: list[float] = []
        self.node_best: list[int] = []
        if instance_types and candidates:
            self.node_matrix = np.column_stack([self.similarity.node_scores(s) for s in candidates])
            # argmax returns the first maximum, so among equal similarities
            # the candidate that comes first in order wins.
            best = self.node_matrix.argmax(axis=1)
            self.node_values = self.node_matrix[np.arange(len(instance_types)), best].tolist()
            self.node_best = best.tolist()
        self._node_top: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.cov_v_value = self._node_coverage()

    def _node_coverage(self) -> float:
        if not self.similarity.instance.node_types:
            return 1.0
        if not self.flat.node_types:
            return 0.0
        return _mean(self.node_values)

    def cov_v(self) -> tuple[float, list[Match]]:
        """Cov_V and the best match of each instance node type."""
        candidates = self.flat.node_types
        if not candidates:
            return self.cov_v_value, [Match(t, None, 0.0) for t in self.similarity.instance.node_types]
        return self.cov_v_value, [
            Match(t, candidates[i] if value > 0 else None, value)
            for t, i, value in zip(self.similarity.instance.node_types, self.node_best, self.node_values)
        ]

    def cov_v_without(self, column: int) -> float:
        """Cov_V when the node type in ``column`` is removed and the others stay as they are."""
        if not self.similarity.instance.node_types:
            return 1.0
        if len(self.flat.node_types) == 1:
            return 0.0
        if self._node_top is None:
            n = self.node_matrix.shape[1]
            part = np.partition(self.node_matrix, n - 2, axis=1)
            self._node_top = (part[:, -1], part[:, -2])
        top1, top2 = self._node_top
        # Without the column, a row keeps its maximum unless the column held
        # it; then the next largest value (equal to it on a tie) takes over.
        values = np.where(self.node_matrix[:, column] < top1, top1, top2).tolist()
        return _mean(values)

    # -- edges ---------------------------------------------------------------

    def _edge_table(self) -> None:
        similarity = self.similarity
        instance = similarity.instance
        params = similarity.params
        elements = self.flat.edge_types

        index: dict[tuple[object, object, object], list[int]] = {}
        for j, s in enumerate(elements):
            for key in product(label_keys(s.labels), label_keys(s.source.labels), label_keys(s.target.labels)):
                bucket = index.get(key)
                if bucket is None:
                    index[key] = [j]
                else:
                    bucket.append(j)

        # The sim_V column of each flattened node type, as a list of floats.
        columns: dict[int, list[float]] = {}

        def column(s_node: object) -> list[float]:
            found = columns.get(id(s_node))
            if found is None:
                found = similarity.node_scores(s_node).tolist()  # type: ignore[arg-type]
                columns[id(s_node)] = found
            return found

        node_index = similarity.node_index
        self.edge_columns: list[list[int]] = []
        self.edge_scores: list[list[float]] = []
        self.edge_values: list[float] = []
        self.edge_best: list[int] = []
        self.rows_by_best: dict[int, list[int]] = {}
        for i, t in enumerate(instance.edge_types):
            keys = list(product(label_keys(t.labels), label_keys(t.source_labels), label_keys(t.target_labels)))
            if len(keys) == 1:
                found: Sequence[int] = index.get(keys[0], ())
            else:
                union: set[int] = set()
                for key in keys:
                    union.update(index.get(key, ()))
                found = sorted(union)
            source_row = node_index[instance.node_type(t.source_labels)]
            target_row = node_index[instance.node_type(t.target_labels)]
            scores = []
            best_value, best = 0.0, -1
            for j in found:
                s = elements[j]
                value = _edge_score(
                    t,
                    s,
                    dice(t.labels, s.labels),
                    column(s.source)[source_row],
                    column(s.target)[target_row],
                    params,
                )
                scores.append(value)
                # Columns are in ascending order: the first maximum wins.
                if value > best_value:
                    best_value, best = value, j
            self.edge_columns.append(list(found))
            self.edge_scores.append(scores)
            self.edge_values.append(best_value)
            self.edge_best.append(best)
            if best >= 0:
                self.rows_by_best.setdefault(best, []).append(i)
        self.cov_e_value = self._edge_coverage()

    def _edge_coverage(self) -> float:
        if not self.similarity.instance.edge_types:
            return 1.0
        if not self.flat.edge_types:
            return 0.0
        return _mean(self.edge_values)

    def cov_e(self) -> tuple[float, list[Match]]:
        """Cov_E and the best match of each instance edge type."""
        elements: Sequence[FlatEdgeType] = self.flat.edge_types
        return self.cov_e_value, [
            Match(t, elements[j] if j >= 0 else None, value)
            for t, j, value in zip(self.similarity.instance.edge_types, self.edge_best, self.edge_values)
        ]

    def cov_e_without(self, removed: Collection[int]) -> float:
        """Cov_E when the elements in the columns ``removed`` are removed and the others stay as they are."""
        if not self.similarity.instance.edge_types:
            return 1.0
        if len(self.flat.edge_types) == len(removed):
            return 0.0
        changed = []
        for j in removed:
            for i in self.rows_by_best.get(j, ()):
                value = max(
                    (v for c, v in zip(self.edge_columns[i], self.edge_scores[i]) if c not in removed),
                    default=0.0,
                )
                if value != self.edge_values[i]:
                    changed.append((i, value, self.edge_values[i]))
        if not changed:
            return self.cov_e_value
        values = self.edge_values
        try:
            for i, value, _ in changed:
                values[i] = value
            return _mean(values)
        finally:
            for i, _, old in changed:
                values[i] = old


def cov_v(flat: FlatSchema, similarity: Similarity) -> tuple[float, list[Match]]:
    """Cov_V and the best match of each instance node type."""
    return CoverageTables(flat, similarity).cov_v()


def cov_e(flat: FlatSchema, similarity: Similarity) -> tuple[float, list[Match]]:
    """Cov_E and the best match of each instance edge type."""
    return CoverageTables(flat, similarity).cov_e()
