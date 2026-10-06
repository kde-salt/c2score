"""Similarity of instance types to schema types (Sections 4.4 and 4.5)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import AbstractSet, Optional

import numpy as np

from .cardinality import DEFAULT_CARDINALITY, d_card
from .datatypes import DataType
from .flatten import FlatEdgeType, FlatNodeType
from .model import Instance, InstanceEdgeType, InstanceNodeType, PropertySpec

Properties = Mapping[str, PropertySpec]


@dataclass(frozen=True)
class SimilarityParams:
    """The weights and switches that the similarity depends on.

    ``label_weight`` is alpha (Eqs. 28-29) and ``edge_weights`` is
    (alpha_A, alpha_E, alpha_C) (Eq. 34).
    """

    label_weight: float = 0.5
    edge_weights: tuple[float, float, float] = (0.5, 0.5, 0.0)
    use_types: bool = True
    allow_any: bool = True


def dice(a: AbstractSet[str], b: AbstractSet[str]) -> float:
    """The Dice coefficient; two empty sets are identical (1) and one empty set gives 0."""
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


def _keys(properties: Properties, required: bool) -> frozenset[str]:
    return frozenset(key for key, spec in properties.items() if spec.required == required)


def d_req(p: Properties, q: Properties) -> float:
    """Eq. 23."""
    return dice(_keys(p, True), _keys(q, True))


def d_opt(p: Properties, q: Properties) -> float:
    """Eq. 24."""
    return dice(_keys(p, False), _keys(q, False))


def d_flag(p: Properties, q: Properties) -> float:
    """Eq. 22."""
    return (d_req(p, q) + d_opt(p, q)) / 2


def delta(instance_type: DataType, schema_type: DataType, allow_any: bool) -> int:
    """Whether the types of one key match."""
    if allow_any:
        return int(instance_type == schema_type or DataType.ANY in (instance_type, schema_type))
    return int(instance_type == schema_type and instance_type != DataType.ANY)


def d_type(p: Properties, q: Properties, allow_any: bool) -> float:
    """Eq. 25, where ``p`` is the instance side and ``q`` the schema side."""
    common = p.keys() & q.keys()
    if not common:
        return 1.0 if not (p.keys() | q.keys()) else 0.0
    return sum(delta(p[k].type, q[k].type, allow_any) for k in common) / len(common)


def d_prop(p: Properties, q: Properties, params: SimilarityParams) -> float:
    """Eq. 26; without types it is D_flag alone."""
    if not params.use_types:
        return d_flag(p, q)
    return (d_flag(p, q) + d_type(p, q, params.allow_any)) / 2


def _attributes(d_lab: float, p: Properties, q: Properties, params: SimilarityParams) -> float:
    """alpha * D_lab + (1 - alpha) * D_prop (Eqs. 28-29)."""
    alpha = params.label_weight
    return alpha * d_lab + (1 - alpha) * d_prop(p, q, params)


def sim_v(t: InstanceNodeType, s: FlatNodeType, params: SimilarityParams) -> float:
    """Eq. 28: 0 if the labels do not overlap."""
    d_lab = dice(t.labels, s.labels)
    if d_lab == 0:
        return 0.0
    return _attributes(d_lab, t.properties, s.properties, params)


def _edge_score(
    t: InstanceEdgeType,
    s: FlatEdgeType,
    d_lab: float,
    source_sim: float,
    target_sim: float,
    params: SimilarityParams,
) -> float:
    """Eq. 34 for a pair that passed the label check and the endpoint filter.

    ``source_sim`` and ``target_sim`` are sim_V of the endpoints (for D_end).
    """
    d_attr = _attributes(d_lab, t.properties, s.properties, params)
    w_attr, w_end, w_card = params.edge_weights
    d_end = (source_sim + target_sim) / 2
    score = w_attr * d_attr + w_end * d_end
    if w_card != 0:
        score += w_card * d_card(t.cardinality, s.cardinality or DEFAULT_CARDINALITY)
    return score


def sim_e(
    t: InstanceEdgeType,
    s: FlatEdgeType,
    instance: Instance,
    params: SimilarityParams,
    node_sim: Optional[Callable[[InstanceNodeType, FlatNodeType], float]] = None,
) -> float:
    """Eq. 34: 0 if the labels do not overlap or the endpoint filter fails (Section 4.5).

    ``node_sim`` computes sim_V for D_end; it defaults to :func:`sim_v`.
    """
    d_lab = dice(t.labels, s.labels)
    if d_lab == 0:
        return 0.0
    if dice(t.source_labels, s.source.labels) == 0 or dice(t.target_labels, s.target.labels) == 0:
        return 0.0
    if node_sim is None:
        node_sim = lambda u, v: sim_v(u, v, params)  # noqa: E731
    return _edge_score(
        t,
        s,
        d_lab,
        node_sim(instance.node_type(t.source_labels), s.source),
        node_sim(instance.node_type(t.target_labels), s.target),
        params,
    )


#: The key that stands for an empty label set in the label indexes: two empty
#: sets have a Dice coefficient of 1, but share no label.
NO_LABEL = None


def label_keys(labels: AbstractSet[str]) -> AbstractSet[Optional[str]]:
    """The keys under which a label set is indexed: its labels, or :data:`NO_LABEL` if it is empty.

    Two label sets have a positive Dice coefficient exactly when their keys overlap.
    """
    return labels if labels else (NO_LABEL,)  # type: ignore[return-value]


class Similarity:
    """sim_V and sim_E for one instance and one set of parameters.

    sim_V is computed per flattened node type for all instance node types at
    once (a column) and cached by content, so the cache stays valid across
    the reduced schemas that Conciseness flattens. Only the instance node
    types that pass the label check are computed; the others are 0 (Eq. 28).
    """

    def __init__(self, instance: Instance, params: SimilarityParams) -> None:
        self.instance = instance
        self.params = params
        self.node_index = {t: i for i, t in enumerate(instance.node_types)}
        self._rows_by_label: dict[Optional[str], list[int]] = {}
        for i, t in enumerate(instance.node_types):
            for key in label_keys(t.labels):
                self._rows_by_label.setdefault(key, []).append(i)
        self._node_scores: dict[FlatNodeType, np.ndarray] = {}

    def node_scores(self, s: FlatNodeType) -> np.ndarray:
        """sim_V(t, s) for every instance node type t."""
        scores = self._node_scores.get(s)
        if scores is None:
            scores = np.zeros(len(self.instance.node_types), dtype=float)
            rows: set[int] = set()
            for key in label_keys(s.labels):
                rows.update(self._rows_by_label.get(key, ()))
            types = self.instance.node_types
            for i in rows:
                scores[i] = sim_v(types[i], s, self.params)
            self._node_scores[s] = scores
        return scores

    def node(self, t: InstanceNodeType, s: FlatNodeType) -> float:
        return float(self.node_scores(s)[self.node_index[t]])

    def edge(self, t: InstanceEdgeType, s: FlatEdgeType) -> float:
        return sim_e(t, s, self.instance, self.params, self.node)
