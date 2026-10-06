"""Property data types and how values are mapped to them."""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from enum import Enum
from typing import Any


class DataType(str, Enum):
    """The data type of a property. ``ANY`` matches every type by default."""

    STRING = "STRING"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    DATETIME = "DATETIME"
    ANY = "ANY"

    def __str__(self) -> str:
        return self.value


_NEO4J_VALUE_TYPES = {
    "STRING": DataType.STRING,
    "INTEGER": DataType.INTEGER,
    "FLOAT": DataType.FLOAT,
    "BOOLEAN": DataType.BOOLEAN,
    "DATE": DataType.DATETIME,
    "LOCAL DATETIME": DataType.DATETIME,
    "ZONED DATETIME": DataType.DATETIME,
}

_NOT_NULL_SUFFIX = " NOT NULL"


def from_neo4j_value_type(value_type: str) -> DataType:
    """Map a result of Neo4j's ``valueType()`` to a data type.

    A trailing ``" NOT NULL"`` is ignored. Types without a counterpart
    (times, durations, points, lists, and anything unknown) map to ``ANY``.
    """
    name = value_type.strip()
    if name.endswith(_NOT_NULL_SUFFIX):
        name = name[: -len(_NOT_NULL_SUFFIX)].rstrip()
    return _NEO4J_VALUE_TYPES.get(name, DataType.ANY)


def is_datetime_marker(value: Any) -> bool:
    """Return whether ``value`` is the JSON datetime marker ``{"$datetime": "..."}``."""
    return (
        isinstance(value, dict)
        and len(value) == 1
        and isinstance(value.get("$datetime"), str)
    )


def from_python_value(value: Any) -> DataType:
    """Map a Python property value to a data type.

    ``None`` means that the property is absent, so it must be dropped by the
    caller before calling this function.
    """
    if value is None:
        raise ValueError("None has no data type; an absent property must be skipped")
    # bool is a subclass of int, so it must be tested first.
    if isinstance(value, bool):
        return DataType.BOOLEAN
    if isinstance(value, int):
        return DataType.INTEGER
    if isinstance(value, float):
        return DataType.FLOAT
    if isinstance(value, str):
        return DataType.STRING
    # datetime.datetime is a subclass of datetime.date.
    if isinstance(value, datetime.date) or is_datetime_marker(value):
        return DataType.DATETIME
    return DataType.ANY


def unify(types: Iterable[DataType]) -> DataType:
    """Decide the single type of a property from the types observed in a group.

    One type is kept as is, ``{INTEGER, FLOAT}`` becomes ``FLOAT``, and every
    other mixture (including any mixture with ``ANY``) becomes ``ANY``.
    """
    observed = frozenset(types)
    if not observed:
        raise ValueError("cannot unify an empty set of types")
    if len(observed) == 1:
        return next(iter(observed))
    if observed == {DataType.INTEGER, DataType.FLOAT}:
        return DataType.FLOAT
    return DataType.ANY


def parse_schema_type(name: str) -> DataType:
    """Read a type name written in a schema, ignoring case."""
    try:
        return DataType(name.upper())
    except ValueError:
        allowed = ", ".join(t.value for t in DataType)
        raise ValueError(f"unknown type {name!r}; expected one of {allowed} (case-insensitive)") from None
