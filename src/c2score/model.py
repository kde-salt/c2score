"""Data model: schema definitions and the abstracted instance Abs(G).

Label sets are always ``frozenset`` objects; they are never joined into
strings.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

from .cardinality import Cardinality, Interval, round_observed
from .datatypes import DataType, from_python_value, unify
from .errors import SchemaError

_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(Mapping[_K, _V]):
    """An immutable, hashable, picklable mapping.

    It compares equal to any mapping with the same items, including ``dict``.
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, items: Mapping[_K, _V] | Iterable[tuple[_K, _V]] = ()) -> None:
        self._data: dict[_K, _V] = dict(items)
        self._hash: Optional[int] = None

    def __getitem__(self, key: _K) -> _V:
        return self._data[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def __eq__(self, other: object) -> bool:
        # Fast path; Mapping.__eq__ copies both sides into new dicts.
        if isinstance(other, FrozenDict):
            if self is other:
                return True
            # Compare hashes only when both are already known: values need not be hashable.
            if self._hash is not None and other._hash is not None and self._hash != other._hash:
                return False
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return super().__eq__(other)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenDict, (self._data,))


@dataclass(frozen=True)
class PropertySpec:
    """The description of one property: its data type and whether it is required."""

    type: DataType = DataType.ANY
    required: bool = True


def _freeze_properties(properties: Mapping[str, PropertySpec]) -> Mapping[str, PropertySpec]:
    return FrozenDict(properties)


@dataclass(frozen=True)
class NodeTypeDef:
    """A node type as written in a schema (one element with an ``id``).

    ``labels`` and ``properties`` are the definition's own; inherited ones are
    added when the schema is flattened. ``supertypes`` keeps the written order.
    """

    id: str
    labels: frozenset[str] = frozenset()
    properties: Mapping[str, PropertySpec] = field(default_factory=dict)
    supertypes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", frozenset(self.labels))
        object.__setattr__(self, "properties", _freeze_properties(self.properties))
        object.__setattr__(self, "supertypes", tuple(self.supertypes))


@dataclass(frozen=True)
class EdgeTypeDef:
    """An edge type as written in a schema (one element with an ``id``).

    ``source``, ``target`` and ``cardinality`` are ``None`` when the definition
    does not write them itself; they may then be inherited from a supertype.
    """

    id: str
    labels: frozenset[str] = frozenset()
    properties: Mapping[str, PropertySpec] = field(default_factory=dict)
    supertypes: tuple[str, ...] = ()
    source: Optional[str] = None
    target: Optional[str] = None
    cardinality: Optional[Cardinality] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", frozenset(self.labels))
        object.__setattr__(self, "properties", _freeze_properties(self.properties))
        object.__setattr__(self, "supertypes", tuple(self.supertypes))


TypeDef = Union[NodeTypeDef, EdgeTypeDef]


def precedence(type_id: str, supertypes_of: Callable[[str], Sequence[str]]) -> tuple[str, ...]:
    """Return the precedence list ``prec(type_id)``.

    The list starts with ``type_id`` itself, followed by its ancestors in the
    order a breadth-first search first visits them, following ``supertypes``
    in their written order. Each ancestor appears once.

    The inheritance graph must already be known to be acyclic.
    """
    order = [type_id]
    seen = {type_id}
    queue = deque([type_id])
    while queue:
        for parent in supertypes_of(queue.popleft()):
            if parent not in seen:
                seen.add(parent)
                order.append(parent)
                queue.append(parent)
    return tuple(order)


@dataclass(frozen=True)
class Schema:
    """A candidate schema: the node and edge definitions in their written order.

    Use :func:`c2score.load_schema_json` to build a validated schema.
    """

    node_types: tuple[NodeTypeDef, ...] = ()
    edge_types: tuple[EdgeTypeDef, ...] = ()
    _by_id: Mapping[str, TypeDef] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_types", tuple(self.node_types))
        object.__setattr__(self, "edge_types", tuple(self.edge_types))
        by_id: dict[str, TypeDef] = {}
        for definition in (*self.node_types, *self.edge_types):
            if definition.id in by_id:
                raise SchemaError(f"duplicate id {definition.id!r}")
            by_id[definition.id] = definition
        object.__setattr__(self, "_by_id", FrozenDict(by_id))

    def __getitem__(self, type_id: str) -> TypeDef:
        return self._by_id[type_id]

    def __contains__(self, type_id: object) -> bool:
        return type_id in self._by_id

    def precedence(self, type_id: str) -> tuple[str, ...]:
        """Return ``prec(type_id)`` (see :func:`precedence`)."""
        return precedence(type_id, lambda t: self._by_id[t].supertypes)

    @property
    def has_cardinality(self) -> bool:
        """Whether some edge definition writes ``cardinality`` itself."""
        return any(edge.cardinality is not None for edge in self.edge_types)


def _sorted_labels(labels: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(labels))


@dataclass(frozen=True)
class InstanceNodeType:
    """A node type of Abs(G): the nodes that share one label set."""

    labels: frozenset[str]
    properties: Mapping[str, PropertySpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", frozenset(self.labels))
        object.__setattr__(self, "properties", _freeze_properties(self.properties))

    @property
    def key(self) -> tuple[str, ...]:
        """The sort key: the sorted labels."""
        return _sorted_labels(self.labels)


@dataclass(frozen=True)
class InstanceEdgeType:
    """An edge type of Abs(G): the edges that share (edge labels, source labels, target labels).

    ``cardinality`` is the observed degree interval, rounded to one of the
    four intervals.
    """

    labels: frozenset[str]
    source_labels: frozenset[str]
    target_labels: frozenset[str]
    cardinality: Cardinality
    properties: Mapping[str, PropertySpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", frozenset(self.labels))
        object.__setattr__(self, "source_labels", frozenset(self.source_labels))
        object.__setattr__(self, "target_labels", frozenset(self.target_labels))
        object.__setattr__(self, "properties", _freeze_properties(self.properties))

    @property
    def key(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """The sort key: the sorted edge, source and target labels."""
        return (
            _sorted_labels(self.labels),
            _sorted_labels(self.source_labels),
            _sorted_labels(self.target_labels),
        )


@dataclass(frozen=True)
class Instance:
    """The summary Abs(G) of a property graph instance.

    The types are kept sorted by their keys, so that the order does not depend
    on how the instance was read.
    """

    node_types: tuple[InstanceNodeType, ...] = ()
    edge_types: tuple[InstanceEdgeType, ...] = ()
    _by_labels: Mapping[frozenset[str], InstanceNodeType] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        nodes = tuple(sorted(self.node_types, key=lambda t: t.key))
        edges = tuple(sorted(self.edge_types, key=lambda t: t.key))
        for kind, types in (("node", nodes), ("edge", edges)):
            for previous, current in zip(types, types[1:]):
                if previous.key == current.key:
                    raise ValueError(f"duplicate instance {kind} type {current.key!r}")
        by_labels = FrozenDict((node.labels, node) for node in nodes)
        for edge in edges:
            for end, labels in (("source", edge.source_labels), ("target", edge.target_labels)):
                if labels not in by_labels:
                    raise ValueError(
                        f"instance edge type {edge.key!r}: no node type with the {end} labels "
                        f"{_sorted_labels(labels)!r}"
                    )
        object.__setattr__(self, "node_types", nodes)
        object.__setattr__(self, "edge_types", edges)
        object.__setattr__(self, "_by_labels", by_labels)

    def node_type(self, labels: Iterable[str]) -> InstanceNodeType:
        """Return the node type with exactly the label set ``labels``.

        Raises ``KeyError`` if there is none.
        """
        return self._by_labels[frozenset(labels)]


# ---------------------------------------------------------------------------
# Abs(G) from an in-memory graph
# ---------------------------------------------------------------------------


class _Group:
    """The members of one Abs(G) type and the property values they carry."""

    def __init__(self) -> None:
        self.size = 0
        self.types: dict[str, set[DataType]] = {}
        self.counts: dict[str, int] = {}

    def add(self, properties: Mapping[str, Any]) -> None:
        self.size += 1
        for key, value in properties.items():
            # A null value means that the property is absent.
            if value is None:
                continue
            self.counts[key] = self.counts.get(key, 0) + 1
            self.types.setdefault(key, set()).add(from_python_value(value))

    def properties(self) -> dict[str, PropertySpec]:
        """Required iff observed on every member (Eq. 7); the type is unified (Section 4.1)."""
        return {
            key: PropertySpec(unify(types), self.counts[key] == self.size)
            for key, types in self.types.items()
        }


def _degree_interval(degrees: Mapping[str, int], members: Iterable[str]) -> Interval:
    """Round [min, max] of the degrees of ``members``; nodes without an edge count as 0."""
    observed = [degrees.get(member, 0) for member in members]
    return round_observed(min(observed), max(observed))


def _abs_from_graph(nodes: Iterable[Mapping[str, Any]], edges: Iterable[Mapping[str, Any]]) -> Instance:
    """Build Abs(G) from nodes and edges in the JSON instance structure.

    Each node is ``{"id", "labels", "properties"}``; each edge is
    ``{"id", "labels", "source", "target", "properties"}`` where ``source``
    and ``target`` are node ids. A ``None`` value means an absent property.
    """
    node_labels: dict[Any, frozenset[str]] = {}
    members_of: dict[frozenset[str], list[Any]] = {}
    node_groups: dict[frozenset[str], _Group] = {}
    for node in nodes:
        node_id = node["id"]
        if node_id in node_labels:
            raise ValueError(f"duplicate node id {node_id!r}")
        labels = frozenset(node.get("labels", ()))
        node_labels[node_id] = labels
        members_of.setdefault(labels, []).append(node_id)
        node_groups.setdefault(labels, _Group()).add(node.get("properties", {}))

    EdgeKey = tuple[frozenset[str], frozenset[str], frozenset[str]]
    edge_ids: set[Any] = set()
    edge_groups: dict[EdgeKey, _Group] = {}
    out_degrees: dict[EdgeKey, dict[Any, int]] = {}
    in_degrees: dict[EdgeKey, dict[Any, int]] = {}
    for edge in edges:
        edge_id = edge["id"]
        if edge_id in edge_ids:
            raise ValueError(f"duplicate edge id {edge_id!r}")
        edge_ids.add(edge_id)
        source, target = edge["source"], edge["target"]
        for end, node_id in (("source", source), ("target", target)):
            if node_id not in node_labels:
                raise ValueError(f"edge {edge_id!r}: unknown {end} node id {node_id!r}")
        key = (frozenset(edge.get("labels", ())), node_labels[source], node_labels[target])
        edge_groups.setdefault(key, _Group()).add(edge.get("properties", {}))
        out_degree = out_degrees.setdefault(key, {})
        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree = in_degrees.setdefault(key, {})
        in_degree[target] = in_degree.get(target, 0) + 1

    node_types = [
        InstanceNodeType(labels, group.properties()) for labels, group in node_groups.items()
    ]
    edge_types = []
    for key, group in edge_groups.items():
        labels, source_labels, target_labels = key
        # Eqs. 10 and 11: every node with the source (target) labels takes part.
        cardinality = Cardinality(
            _degree_interval(out_degrees[key], members_of[source_labels]),
            _degree_interval(in_degrees[key], members_of[target_labels]),
        )
        edge_types.append(
            InstanceEdgeType(labels, source_labels, target_labels, cardinality, group.properties())
        )
    return Instance(tuple(node_types), tuple(edge_types))
