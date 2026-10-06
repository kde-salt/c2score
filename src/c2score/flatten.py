"""Flattening a schema: S+ = Flat(S) (Section 4.2).

Inheritance is resolved with the precedence list ``prec(t)``: labels are the
union over the list, and every other component (each property, the
cardinality, the source and the target) comes from the first definition in
the list that writes it. Edge types are then expanded to every pair of
(source or a descendant, target or a descendant).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional

from .cardinality import Cardinality
from .errors import SchemaError
from .model import EdgeTypeDef, FrozenDict, PropertySpec, Schema, TypeDef


@dataclass(frozen=True)
class FlatNodeType:
    """A node type of S+: one node definition with its inheritance resolved."""

    id: str
    labels: frozenset[str]
    properties: Mapping[str, PropertySpec]
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.id, self.labels, self.properties)))

    def __hash__(self) -> int:
        # Flattened types are cache keys in the similarity, looked up many times.
        return self._hash


@dataclass(frozen=True)
class FlatEdgeType:
    """An element of E(S+): an edge type with resolved inheritance and fixed endpoints.

    ``cardinality`` is ``None`` when no definition in the precedence list
    writes one. ``origins`` are the ids of the edge definitions that produce
    this element; they do not take part in equality.
    """

    labels: frozenset[str]
    properties: Mapping[str, PropertySpec]
    cardinality: Optional[Cardinality]
    source: FlatNodeType
    target: FlatNodeType
    origins: frozenset[str] = field(default=frozenset(), compare=False)
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        key = (self.labels, self.properties, self.cardinality, self.source, self.target)
        object.__setattr__(self, "_hash", hash(key))

    def __hash__(self) -> int:
        return self._hash

    @property
    def schema_type(self) -> str:
        """The smallest id among the definitions that produce this element."""
        return min(self.origins)


@dataclass(frozen=True)
class FlatSchema:
    """S+: node types in id order, edge elements in (smallest origin, source, target) order."""

    node_types: tuple[FlatNodeType, ...]
    edge_types: tuple[FlatEdgeType, ...]


def _merged_labels(chain: Sequence[TypeDef]) -> frozenset[str]:
    if len(chain) == 1:
        return chain[0].labels
    return frozenset().union(*(definition.labels for definition in chain))


def _merged_properties(chain: Sequence[TypeDef]) -> Mapping[str, PropertySpec]:
    """For each key, the description written by the first definition in the chain."""
    if len(chain) == 1:
        # Already a FrozenDict; sharing it keeps its cached hash.
        return chain[0].properties
    merged: dict[str, PropertySpec] = {}
    for definition in chain:
        for key, spec in definition.properties.items():
            merged.setdefault(key, spec)
    return FrozenDict(merged)


def _chain(schema: Schema, definition: TypeDef) -> list[TypeDef]:
    """The definitions of ``prec(definition)``, nearest first."""
    if not definition.supertypes:
        return [definition]
    try:
        return [schema[type_id] for type_id in schema.precedence(definition.id)]
    except KeyError as e:
        raise SchemaError(f"{definition.id!r}: unknown supertype id {e.args[0]!r}") from None


def _endpoint(chain: Sequence[TypeDef], end: str) -> Optional[str]:
    for definition in chain:
        endpoint = getattr(definition, end)
        if endpoint is not None:
            return endpoint
    return None


def inherited_endpoint(schema: Schema, edge: EdgeTypeDef, end: str) -> Optional[str]:
    """The ``source`` or ``target`` of an edge definition, possibly inherited; ``None`` if undetermined."""
    return _endpoint(_chain(schema, edge), end)


def _descendants(schema: Schema) -> dict[str, list[str]]:
    """For each node definition, itself followed by all its descendants."""
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
    """Compute S+ = Flat(S).

    Raises :class:`SchemaError` if an edge type's endpoint is undetermined or
    does not refer to a node definition (a schema read with
    :func:`c2score.load_schema_json` never does this).
    """
    flat_nodes: dict[str, FlatNodeType] = {}
    for node in schema.node_types:
        chain = _chain(schema, node)
        flat_nodes[node.id] = FlatNodeType(node.id, _merged_labels(chain), _merged_properties(chain))

    descendants = _descendants(schema)
    # Eq. 21: E(S+) is a set; equal elements are merged and remember all origins.
    # A key compares the properties as a mapping, which is the same as
    # comparing their sets of items.
    origins: dict[tuple[object, ...], list[str]] = {}
    for edge in schema.edge_types:
        chain = _chain(schema, edge)
        labels = _merged_labels(chain)
        properties = _merged_properties(chain)
        cardinality = next((d.cardinality for d in chain if d.cardinality is not None), None)
        ends = []
        for end in ("source", "target"):
            endpoint = _endpoint(chain, end)
            if endpoint is None:
                raise SchemaError(f"{edge.id!r}: {end} is not determined")
            if endpoint not in flat_nodes:
                raise SchemaError(f"{edge.id!r}.{end}: {endpoint!r} is not a node type")
            ends.append(endpoint)
        source, target = ends
        # Eqs. 18-19: the Cartesian product of the endpoints and their descendants.
        target_ids = descendants[target]
        for source_id in descendants[source]:
            for target_id in target_ids:
                key = (labels, properties, cardinality, source_id, target_id)
                found = origins.get(key)
                if found is None:
                    origins[key] = [edge.id]
                elif found[-1] != edge.id:
                    found.append(edge.id)

    edge_types = [
        FlatEdgeType(labels, properties, cardinality, flat_nodes[source_id], flat_nodes[target_id], frozenset(ids))
        for (labels, properties, cardinality, source_id, target_id), ids in origins.items()
    ]
    edge_types.sort(key=lambda e: (e.schema_type, e.source.id, e.target.id))
    node_types = sorted(flat_nodes.values(), key=lambda n: n.id)
    return FlatSchema(tuple(node_types), tuple(edge_types))
