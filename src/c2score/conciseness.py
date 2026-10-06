"""Conciseness: which written definitions are redundant (Section 4.7).

Each definition is removed in turn, together with everything that depends on
it, and the coverage of the reduced schema is compared with the original.

When no other definition inherits from the removed one (and, for a node
definition, no edge definition that goes with it), the remaining definitions
flatten exactly as before: S+ of the reduced schema is S+ minus the elements
that the removal takes away. The coverage is then read off the original
similarities. Otherwise the reduced schema is flattened again.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import Optional, TypeVar

from .coverage import CoverageTables
from .flatten import FlatSchema, flatten, inherited_endpoint
from .model import EdgeTypeDef, NodeTypeDef, Schema
from .similarity import Similarity

_D = TypeVar("_D", NodeTypeDef, EdgeTypeDef)


@dataclass(frozen=True)
class Removal:
    """The effect of removing one definition."""

    type_id: str
    delta_v: float
    delta_e: float
    redundant: bool


@dataclass(frozen=True)
class ConcisenessResult:
    con_v: float
    con_e: float
    theta_v: float
    theta_e: float
    node_removals: tuple[Removal, ...]
    edge_removals: tuple[Removal, ...]
    #: The definitions whose removal needed a new flattening (the others were incremental).
    reflattened: tuple[str, ...] = ()


def reduce_schema(schema: Schema, type_id: str) -> Schema:
    """S - d: remove a definition and, repeatedly, what no longer holds without it.

    Removed definitions are dropped from the remaining ``supertypes``. An edge
    definition whose endpoint (possibly inherited) is removed or no longer
    determined is removed as well.
    """
    removed = {type_id}

    def kept(definition: _D) -> _D:
        if not any(parent in removed for parent in definition.supertypes):
            return definition
        return replace(definition, supertypes=tuple(p for p in definition.supertypes if p not in removed))

    while True:
        reduced = Schema(
            node_types=tuple(kept(n) for n in schema.node_types if n.id not in removed),
            edge_types=tuple(kept(e) for e in schema.edge_types if e.id not in removed),
        )
        broken = set()
        for edge in reduced.edge_types:
            for end in ("source", "target"):
                endpoint = inherited_endpoint(reduced, edge, end)
                if endpoint is None or endpoint in removed:
                    broken.add(edge.id)
        if not broken:
            return reduced
        removed |= broken


def _ratio(redundant: int, total: int) -> float:
    """1 - redundant / total, or 1 if there is nothing to remove."""
    return 1.0 if total == 0 else 1 - redundant / total


class _Shortcuts:
    """What a removal takes away from S+, for the removals that leave the rest of S+ unchanged."""

    def __init__(self, schema: Schema, flat: FlatSchema) -> None:
        children: dict[str, list[str]] = {}
        for definition in (*schema.node_types, *schema.edge_types):
            for parent in definition.supertypes:
                children.setdefault(parent, []).append(definition.id)
        attached: dict[str, set[str]] = {}
        for edge in schema.edge_types:
            for end in ("source", "target"):
                endpoint = inherited_endpoint(schema, edge, end)
                if endpoint is not None:
                    attached.setdefault(endpoint, set()).add(edge.id)
        self._children = children
        self._attached = attached
        self._node_column = {s.id: j for j, s in enumerate(flat.node_types)}
        self._touching: dict[str, set[int]] = {}
        self._only_from: dict[str, set[int]] = {}
        for j, e in enumerate(flat.edge_types):
            self._touching.setdefault(e.source.id, set()).add(j)
            self._touching.setdefault(e.target.id, set()).add(j)
            if len(e.origins) == 1:
                self._only_from.setdefault(next(iter(e.origins)), set()).add(j)

    def node(self, type_id: str) -> Optional[tuple[int, Collection[int]]]:
        """For a node definition: its column in V(S+) and the columns of E(S+) that go with it.

        ``None`` if the removal changes other elements of S+. That happens if
        some definition inherits from it, or from an edge definition removed
        with it (one whose endpoint, possibly inherited, is this node).
        Otherwise the removed edge definitions only produce elements whose
        source or target is this node (it has no descendants), and every
        other element survives with the same content.
        """
        if self._children.get(type_id):
            return None
        removed_edges = self._attached.get(type_id, set())
        for edge_id in removed_edges:
            if any(child not in removed_edges for child in self._children.get(edge_id, ())):
                return None
        return self._node_column[type_id], self._touching.get(type_id, set())

    def edge(self, type_id: str) -> Optional[Collection[int]]:
        """For an edge definition: the columns of E(S+) produced by it alone.

        ``None`` if some definition inherits from it.
        """
        if self._children.get(type_id):
            return None
        return self._only_from.get(type_id, set())


def conciseness(
    schema: Schema,
    flat: FlatSchema,
    base_cov_v: float,
    base_cov_e: float,
    similarity: Similarity,
    redundancy_ratio: float,
    *,
    tables: Optional[CoverageTables] = None,
) -> ConcisenessResult:
    """Con_V and Con_E. ``flat``, the base coverages and ``tables`` belong to the original schema."""
    n_v, n_e = len(flat.node_types), len(flat.edge_types)
    theta_v = redundancy_ratio * base_cov_v / n_v if n_v else 0.0
    theta_e = redundancy_ratio * base_cov_e / n_e if n_e else 0.0
    if tables is None:
        tables = CoverageTables(flat, similarity)
    shortcuts = _Shortcuts(schema, flat)
    reflattened: list[str] = []

    def judge(type_id: str, cov_v_after: float, cov_e_after: float) -> Removal:
        delta_v = base_cov_v - cov_v_after
        delta_e = base_cov_e - cov_e_after
        redundant = (delta_v < theta_v or delta_v <= 0) and (delta_e < theta_e or delta_e <= 0)
        return Removal(type_id, delta_v, delta_e, redundant)

    def again(type_id: str) -> Removal:
        reflattened.append(type_id)
        reduced = CoverageTables(flatten(reduce_schema(schema, type_id)), similarity)
        return judge(type_id, reduced.cov_v_value, reduced.cov_e_value)

    def node_removal(type_id: str) -> Removal:
        taken = shortcuts.node(type_id)
        if taken is None:
            return again(type_id)
        column, edges = taken
        return judge(type_id, tables.cov_v_without(column), tables.cov_e_without(edges))

    def edge_removal(type_id: str) -> Removal:
        edges = shortcuts.edge(type_id)
        if edges is None:
            return again(type_id)
        # The node types do not change, so Cov_V is computed from the same values.
        return judge(type_id, tables.cov_v_value, tables.cov_e_without(edges))

    node_removals = tuple(node_removal(n.id) for n in schema.node_types)
    edge_removals = tuple(edge_removal(e.id) for e in schema.edge_types)
    return ConcisenessResult(
        con_v=_ratio(sum(r.redundant for r in node_removals), len(node_removals)),
        con_e=_ratio(sum(r.redundant for r in edge_removals), len(edge_removals)),
        theta_v=theta_v,
        theta_e=theta_e,
        node_removals=node_removals,
        edge_removals=edge_removals,
        reflattened=tuple(reflattened),
    )
