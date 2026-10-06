"""Conciseness: which written definitions are redundant (Section 4.7).

Each definition is removed in turn, together with everything that depends on
it, and the coverage of the reduced schema is compared with the original.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .coverage import cov_e, cov_v
from .flatten import FlatSchema, flatten, inherited_endpoint
from .model import Schema
from .similarity import Similarity


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


def reduce_schema(schema: Schema, type_id: str) -> Schema:
    """S - d: remove a definition and, repeatedly, what no longer holds without it.

    Removed definitions are dropped from the remaining ``supertypes``. An edge
    definition whose endpoint (possibly inherited) is removed or no longer
    determined is removed as well.
    """
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
    """1 - redundant / total, or 1 if there is nothing to remove."""
    return 1.0 if total == 0 else 1 - redundant / total


def conciseness(
    schema: Schema,
    flat: FlatSchema,
    base_cov_v: float,
    base_cov_e: float,
    similarity: Similarity,
    redundancy_ratio: float,
) -> ConcisenessResult:
    """Con_V and Con_E. ``flat`` and the base coverages belong to the original schema."""
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
