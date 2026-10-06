"""Data model: schema definitions and the abstracted instance Abs(G).

Label sets are always ``frozenset`` objects; they are never joined into
strings.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

from .cardinality import Cardinality
from .datatypes import DataType
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

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenDict, (self._data,))


@dataclass(frozen=True)
class PropertySpec:
    """The description of one property: its data type and whether it is required.

    ``type_given`` and ``required_given`` record whether the schema wrote the
    value or it was filled in with the default. They do not take part in
    equality or hashing.
    """

    type: DataType = DataType.ANY
    required: bool = True
    type_given: bool = field(default=True, compare=False)
    required_given: bool = field(default=True, compare=False)


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

    def __post_init__(self) -> None:
        nodes = tuple(sorted(self.node_types, key=lambda t: t.key))
        edges = tuple(sorted(self.edge_types, key=lambda t: t.key))
        for kind, types in (("node", nodes), ("edge", edges)):
            for previous, current in zip(types, types[1:]):
                if previous.key == current.key:
                    raise ValueError(f"duplicate instance {kind} type {current.key!r}")
        object.__setattr__(self, "node_types", nodes)
        object.__setattr__(self, "edge_types", edges)
