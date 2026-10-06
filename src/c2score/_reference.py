"""The straightforward implementation of the evaluation, kept as a reference.

This is the implementation that passed the hand-computed examples and the
comparison with the earlier implementation before the evaluation was made
fast. It
flattens the schema again for every removed definition and computes every
similarity densely. It is not part of the public API: the tests compare the
fast :func:`c2score.evaluate` with :func:`evaluate` here.

The similarity rules themselves (:mod:`c2score.similarity`) are shared.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Optional, TypeVar

import numpy as np

from .conciseness import ConcisenessResult, Removal
from .coverage import Match
from .errors import SchemaError
from .evaluate import EvaluationResult, _check_params, _result
from .flatten import FlatEdgeType, FlatNodeType, FlatSchema
from .model import EdgeTypeDef, FrozenDict, Instance, InstanceEdgeType, InstanceNodeType, PropertySpec, Schema, TypeDef
from .similarity import SimilarityParams, sim_e, sim_v

_T = TypeVar("_T")
_S = TypeVar("_S")


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def _merged_labels(chain: Iterable[TypeDef]) -> frozenset[str]:
    return frozenset().union(*(definition.labels for definition in chain))


def _merged_properties(chain: Iterable[TypeDef]) -> FrozenDict[str, PropertySpec]:
    merged: dict[str, PropertySpec] = {}
    for definition in chain:
        for key, spec in definition.properties.items():
            merged.setdefault(key, spec)
    return FrozenDict(merged)


def _chain(schema: Schema, definition: TypeDef) -> list[TypeDef]:
    try:
        return [schema[type_id] for type_id in schema.precedence(definition.id)]
    except KeyError as e:
        raise SchemaError(f"{definition.id!r}: unknown supertype id {e.args[0]!r}") from None


def inherited_endpoint(schema: Schema, edge: EdgeTypeDef, end: str) -> Optional[str]:
    for definition in _chain(schema, edge):
        endpoint = getattr(definition, end)
        if endpoint is not None:
            return endpoint
    return None


def _descendants(schema: Schema) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {node.id: [] for node in schema.node_types}
    for node in schema.node_types:
        for parent in node.supertypes:
            if parent in children:
                children[parent].append(node.id)
    closure: dict[str, list[str]] = {}
    for node_id in children:
        order = [node_id]
        seen = {node_id}
        for current in order:
            for child in children[current]:
                if child not in seen:
                    seen.add(child)
                    order.append(child)
        closure[node_id] = order
    return closure


def flatten(schema: Schema) -> FlatSchema:
    flat_nodes: dict[str, FlatNodeType] = {}
    for node in schema.node_types:
        chain = _chain(schema, node)
        flat_nodes[node.id] = FlatNodeType(node.id, _merged_labels(chain), _merged_properties(chain))

    descendants = _descendants(schema)
    elements: dict[tuple[object, ...], FlatEdgeType] = {}
    origins: dict[tuple[object, ...], set[str]] = {}
    for edge in schema.edge_types:
        chain = _chain(schema, edge)
        labels = _merged_labels(chain)
        properties = _merged_properties(chain)
        cardinality = next((d.cardinality for d in chain if d.cardinality is not None), None)
        ends = []
        for end in ("source", "target"):
            endpoint = inherited_endpoint(schema, edge, end)
            if endpoint is None:
                raise SchemaError(f"{edge.id!r}: {end} is not determined")
            if endpoint not in flat_nodes:
                raise SchemaError(f"{edge.id!r}.{end}: {endpoint!r} is not a node type")
            ends.append(endpoint)
        source, target = ends
        for source_id in descendants[source]:
            for target_id in descendants[target]:
                key = (labels, frozenset(properties.items()), cardinality, source_id, target_id)
                if key not in elements:
                    elements[key] = FlatEdgeType(
                        labels, properties, cardinality, flat_nodes[source_id], flat_nodes[target_id]
                    )
                    origins[key] = set()
                origins[key].add(edge.id)

    edge_types = [
        FlatEdgeType(e.labels, e.properties, e.cardinality, e.source, e.target, frozenset(origins[key]))
        for key, e in elements.items()
    ]
    edge_types.sort(key=lambda e: (e.schema_type, e.source.id, e.target.id))
    node_types = sorted(flat_nodes.values(), key=lambda n: n.id)
    return FlatSchema(tuple(node_types), tuple(edge_types))


# ---------------------------------------------------------------------------
# Similarity with a dense cache
# ---------------------------------------------------------------------------


class Similarity:
    """sim_V and sim_E against every instance type, cached per flattened type (by content)."""

    def __init__(self, instance: Instance, params: SimilarityParams) -> None:
        self.instance = instance
        self.params = params
        self._node_index = {t: i for i, t in enumerate(instance.node_types)}
        self._edge_index = {t: i for i, t in enumerate(instance.edge_types)}
        self._node_scores: dict[FlatNodeType, np.ndarray] = {}
        self._edge_scores: dict[FlatEdgeType, np.ndarray] = {}

    def node_scores(self, s: FlatNodeType) -> np.ndarray:
        scores = self._node_scores.get(s)
        if scores is None:
            scores = np.array([sim_v(t, s, self.params) for t in self.instance.node_types], dtype=float)
            self._node_scores[s] = scores
        return scores

    def edge_scores(self, s: FlatEdgeType) -> np.ndarray:
        scores = self._edge_scores.get(s)
        if scores is None:
            scores = np.array(
                [sim_e(t, s, self.instance, self.params, self.node) for t in self.instance.edge_types],
                dtype=float,
            )
            self._edge_scores[s] = scores
        return scores

    def node(self, t: InstanceNodeType, s: FlatNodeType) -> float:
        return float(self.node_scores(s)[self._node_index[t]])

    def edge(self, t: InstanceEdgeType, s: FlatEdgeType) -> float:
        return float(self.edge_scores(s)[self._edge_index[t]])


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _coverage(
    instance_types: Sequence[_T], candidates: Sequence[_S], scores: Callable[[_S], np.ndarray]
) -> tuple[float, list[Match[_T, _S]]]:
    if not instance_types:
        return 1.0, []
    if not candidates:
        return 0.0, [Match(t, None, 0.0) for t in instance_types]
    matrix = np.column_stack([scores(c) for c in candidates])
    best = matrix.argmax(axis=1)
    values = matrix[np.arange(len(instance_types)), best].tolist()
    matches = [
        Match(t, candidates[i] if value > 0 else None, value)
        for t, i, value in zip(instance_types, best.tolist(), values)
    ]
    return sum(values) / len(instance_types), matches


def cov_v(flat: FlatSchema, similarity: Similarity) -> tuple[float, list[Match]]:
    return _coverage(similarity.instance.node_types, flat.node_types, similarity.node_scores)


def cov_e(flat: FlatSchema, similarity: Similarity) -> tuple[float, list[Match]]:
    return _coverage(similarity.instance.edge_types, flat.edge_types, similarity.edge_scores)


# ---------------------------------------------------------------------------
# Conciseness
# ---------------------------------------------------------------------------


def reduce_schema(schema: Schema, type_id: str) -> Schema:
    removed = {type_id}
    while True:
        reduced = Schema(
            node_types=tuple(
                replace(n, supertypes=tuple(p for p in n.supertypes if p not in removed))
                for n in schema.node_types
                if n.id not in removed
            ),
            edge_types=tuple(
                replace(e, supertypes=tuple(p for p in e.supertypes if p not in removed))
                for e in schema.edge_types
                if e.id not in removed
            ),
        )
        broken = {
            edge.id
            for edge in reduced.edge_types
            for end in ("source", "target")
            if inherited_endpoint(reduced, edge, end) in (None, *removed)
        }
        if not broken:
            return reduced
        removed |= broken


def _ratio(redundant: int, total: int) -> float:
    return 1.0 if total == 0 else 1 - redundant / total


def conciseness(
    schema: Schema,
    flat: FlatSchema,
    base_cov_v: float,
    base_cov_e: float,
    similarity: Similarity,
    redundancy_ratio: float,
) -> ConcisenessResult:
    n_v, n_e = len(flat.node_types), len(flat.edge_types)
    theta_v = redundancy_ratio * base_cov_v / n_v if n_v else 0.0
    theta_e = redundancy_ratio * base_cov_e / n_e if n_e else 0.0

    def removal(type_id: str) -> Removal:
        reduced = flatten(reduce_schema(schema, type_id))
        delta_v = base_cov_v - cov_v(reduced, similarity)[0]
        delta_e = base_cov_e - cov_e(reduced, similarity)[0]
        redundant = (delta_v < theta_v or delta_v <= 0) and (delta_e < theta_e or delta_e <= 0)
        return Removal(type_id, delta_v, delta_e, redundant)

    node_removals = tuple(removal(n.id) for n in schema.node_types)
    edge_removals = tuple(removal(e.id) for e in schema.edge_types)
    return ConcisenessResult(
        con_v=_ratio(sum(r.redundant for r in node_removals), len(node_removals)),
        con_e=_ratio(sum(r.redundant for r in edge_removals), len(edge_removals)),
        theta_v=theta_v,
        theta_e=theta_e,
        node_removals=node_removals,
        edge_removals=edge_removals,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceRun:
    """The public result together with the intermediate values, for the tests."""

    result: EvaluationResult
    flat: FlatSchema
    conciseness: ConcisenessResult


def run(instance: Instance, schema: Schema, **kwargs: object) -> ReferenceRun:
    """Like :func:`c2score.evaluate` (same keyword arguments), with the intermediate values."""
    checked = _check_params(instance, schema, **kwargs)  # type: ignore[arg-type]
    similarity = Similarity(instance, checked.similarity)
    flat = flatten(schema)
    cov_v_value, node_matches = cov_v(flat, similarity)
    cov_e_value, edge_matches = cov_e(flat, similarity)
    con = conciseness(schema, flat, cov_v_value, cov_e_value, similarity, checked.redundancy_ratio)
    result = _result(checked, cov_v_value, cov_e_value, node_matches, edge_matches, con)
    return ReferenceRun(result, flat, con)


def evaluate(instance: Instance, schema: Schema, **kwargs: object) -> EvaluationResult:
    """The reference counterpart of :func:`c2score.evaluate`."""
    return run(instance, schema, **kwargs).result

