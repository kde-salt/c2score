"""Reading and validating JSON schema files."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib.resources import files
from typing import Any, Union

from jsonschema import Draft202012Validator

from .cardinality import Cardinality, parse_interval
from .datatypes import DataType, parse_schema_type
from .errors import SchemaError
from .model import EdgeTypeDef, NodeTypeDef, PropertySpec, Schema

_NODE_TYPES = "node_types"
_EDGE_TYPES = "edge_types"
_NOUN = {_NODE_TYPES: "a node type", _EDGE_TYPES: "an edge type"}

PathItem = Union[str, int]


@lru_cache(maxsize=None)
def _validator() -> Draft202012Validator:
    text = files("c2score").joinpath("resources/schema.json").read_text(encoding="utf-8")
    return Draft202012Validator(json.loads(text))


# ---------------------------------------------------------------------------
# Locations in messages
# ---------------------------------------------------------------------------


_MAX_VALUE_LENGTH = 80
_MAX_LISTED_KEYS = 5


def _short(value: Any) -> str:
    """``repr(value)``, shortened so that a huge value cannot flood a message."""
    text = repr(value)
    if len(text) > _MAX_VALUE_LENGTH:
        text = text[: _MAX_VALUE_LENGTH - 3] + "..."
    return text


def _format_path(path: tuple[PathItem, ...], data: Any) -> str:
    """Format a JSON location such as ``edge_types[2] (id='managesType').source``."""
    if not path:
        return "<root>"
    text = ""
    for depth, item in enumerate(path):
        if isinstance(item, int):
            text += f"[{item}]"
        else:
            name = item if len(item) <= _MAX_VALUE_LENGTH else item[: _MAX_VALUE_LENGTH - 3] + "..."
            text += f".{name}" if text else name
        if depth == 1 and path[0] in (_NODE_TYPES, _EDGE_TYPES) and isinstance(item, int):
            type_id = _id_at(data, path[0], item)
            if type_id is not None:
                text += f" (id={_short(type_id)})"
    return text


def _id_at(data: Any, kind: PathItem, index: int) -> str | None:
    try:
        element = data[kind][index]
    except (KeyError, IndexError, TypeError):
        return None
    type_id = element.get("id") if isinstance(element, dict) else None
    return type_id if isinstance(type_id, str) and type_id else None


def _raise(problems: list[str], origin: str) -> None:
    if problems:
        lines = "\n".join(f"  {problem}" for problem in problems)
        raise SchemaError(f"invalid schema ({origin}):\n{lines}")


# ---------------------------------------------------------------------------
# R7: JSON parsing with duplicate key detection
# ---------------------------------------------------------------------------


class _Pairs(list):
    """The key-value pairs of a JSON object, before duplicate keys are checked."""


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _parse_json(text: str, origin: str) -> Any:
    try:
        raw = json.loads(text, object_pairs_hook=_Pairs, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise SchemaError(f"invalid schema ({origin}): not valid JSON: {e}") from e
    except ValueError as e:
        raise SchemaError(f"invalid schema ({origin}): {e}") from e

    duplicates: list[tuple[tuple[PathItem, ...], str]] = []

    def build(value: Any, path: tuple[PathItem, ...]) -> Any:
        if isinstance(value, _Pairs):
            result: dict[str, Any] = {}
            for key, item in value:
                if key in result:
                    duplicates.append((path, key))
                result[key] = build(item, (*path, key))
            return result
        if isinstance(value, list):
            return [build(item, (*path, i)) for i, item in enumerate(value)]
        return value

    data = build(raw, ())
    _raise(
        [f"{_format_path(path, data)}: duplicate key {_short(key)}" for path, key in duplicates],
        origin,
    )
    return data


# ---------------------------------------------------------------------------
# R8-R13: semantic checks
# ---------------------------------------------------------------------------


def _check_keys(value: Any, path: tuple[PathItem, ...], data: Any) -> list[str]:
    """Object keys are strings. Only a ``dict`` source can break this rule."""
    problems: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                problems += _check_keys(item, (*path, key), data)
            else:
                where = _format_path(path, data)
                problems.append(f"{where}: key {_short(key)} is not a string")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            problems += _check_keys(item, (*path, index), data)
    return problems


def _format_error(error: Any) -> str:
    """The message of a JSON Schema violation, with the offending value shortened."""
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        # jsonschema lists every unexpected key in full; list a few, shortened.
        known = error.schema.get("properties", {})
        extra = [key for key in error.instance if key not in known]
        shown = ", ".join(_short(key) for key in extra[:_MAX_LISTED_KEYS])
        if len(extra) > _MAX_LISTED_KEYS:
            shown += f" and {len(extra) - _MAX_LISTED_KEYS} more"
        verb = "was" if len(extra) == 1 else "were"
        return f"Additional properties are not allowed ({shown} {verb} unexpected)"
    message = error.message
    value = repr(error.instance)
    if len(value) > _MAX_VALUE_LENGTH:
        message = message.replace(value, _short(error.instance))
    return message


def _elements(data: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    return [
        (kind, index, element)
        for kind in (_NODE_TYPES, _EDGE_TYPES)
        for index, element in enumerate(data.get(kind, []))
    ]


def _schema_type(name: str) -> DataType:
    """R13: read a type name, ignoring case.

    Only ASCII names are accepted: ``str.upper`` maps some other letters to
    ASCII ones (``"\u017ftring".upper() == "STRING"``).
    """
    if name.isascii():
        try:
            return parse_schema_type(name)
        except ValueError:
            pass
    allowed = ", ".join(t.value for t in DataType)
    raise ValueError(f"unknown type {_short(name)}; expected one of {allowed} (case-insensitive)")


def _check_references(data: dict[str, Any]) -> list[str]:
    """R8 (unique ids), R9 (supertypes), R11 (endpoints), and R13 (type names)."""
    problems: list[str] = []
    elements = _elements(data)

    # R8: ids are unique across node and edge types.
    kind_of: dict[str, str] = {}
    first_at: dict[str, str] = {}
    duplicated: set[str] = set()
    for kind, index, element in elements:
        type_id = element["id"]
        where = _format_path((kind, index), data)
        if type_id in kind_of:
            duplicated.add(type_id)
            problems.append(f"{where}.id: duplicate id {_short(type_id)} (already used by {first_at[type_id]})")
        else:
            kind_of[type_id] = kind
            first_at[type_id] = where

    for kind, index, element in elements:
        type_id = element["id"]
        noun = _NOUN[kind]

        # R9: supertypes exist, keep the kind, and are not the type itself.
        # A duplicated id is already reported by R8 and is ambiguous, so
        # references to it are not checked further.
        for position, parent in enumerate(element.get("supertypes", [])):
            if parent in duplicated:
                continue
            where = _format_path((kind, index, "supertypes", position), data)
            if parent == type_id:
                problems.append(f"{where}: a type cannot be its own supertype")
            elif parent not in kind_of:
                problems.append(f"{where}: unknown id {_short(parent)}")
            elif kind_of[parent] != kind:
                problems.append(
                    f"{where}: {_short(parent)} is {_NOUN[kind_of[parent]]}; "
                    f"{noun} can only inherit from {kind.replace('_', ' ')}"
                )

        # R11: source and target refer to node types.
        if kind == _EDGE_TYPES:
            for end in ("source", "target"):
                if end not in element or element[end] in duplicated:
                    continue
                where = _format_path((kind, index, end), data)
                endpoint = element[end]
                if endpoint not in kind_of:
                    problems.append(f"{where}: unknown id {_short(endpoint)}")
                elif kind_of[endpoint] != _NODE_TYPES:
                    problems.append(f"{where}: {_short(endpoint)} is not a node type")

        # R13: type names are known.
        for key, spec in element.get("properties", {}).items():
            if "type" in spec:
                try:
                    _schema_type(spec["type"])
                except ValueError as e:
                    where = _format_path((kind, index, "properties", key, "type"), data)
                    problems.append(f"{where}: {e}")
    return problems


def _check_cycles(data: dict[str, Any]) -> list[str]:
    """R10: the inheritance graph has no cycle."""
    parents = {element["id"]: element.get("supertypes", []) for _, _, element in _elements(data)}
    position = {element["id"]: (kind, index) for kind, index, element in _elements(data)}
    done: set[str] = set()
    problems: list[str] = []

    # Iterative depth-first search, so that deep hierarchies cannot exhaust the call stack.
    # A cycle is reported at the supertypes entry that closes it.
    for root in parents:
        if root in done:
            continue
        path = [root]
        on_path = {root}
        pending = [enumerate(parents[root])]
        while pending:
            position_in_list, parent = next(pending[-1], (None, None))
            if parent is None:
                pending.pop()
                finished = path.pop()
                on_path.discard(finished)
                done.add(finished)
            elif parent in on_path:
                cycle = path[path.index(parent):] + [parent]
                where = _format_path((*position[path[-1]], "supertypes", position_in_list), data)
                problems.append(f"{where}: inheritance cycle {' -> '.join(cycle)}")
            elif parent not in done:
                path.append(parent)
                on_path.add(parent)
                pending.append(enumerate(parents[parent]))
    return problems


def _check_endpoints(schema: Schema, data: dict[str, Any]) -> list[str]:
    """R12: every edge type has a source and a target, possibly inherited."""
    problems: list[str] = []
    for index, edge in enumerate(schema.edge_types):
        chain = schema.precedence(edge.id)
        for end in ("source", "target"):
            if all(getattr(schema[t], end) is None for t in chain):
                where = _format_path((_EDGE_TYPES, index, end), data)
                problems.append(
                    f"{where}: {end} is not determined; neither the edge type nor its "
                    f"supertypes ({', '.join(chain[1:]) or 'none'}) specify {end!r}"
                )
    return problems


# ---------------------------------------------------------------------------
# Building the model with defaults
# ---------------------------------------------------------------------------


def _properties(element: dict[str, Any]) -> dict[str, PropertySpec]:
    return {
        key: PropertySpec(
            type=_schema_type(spec["type"]) if "type" in spec else DataType.ANY,
            required=spec.get("required", True),
        )
        for key, spec in element.get("properties", {}).items()
    }


def _build(data: dict[str, Any]) -> Schema:
    node_types = [
        NodeTypeDef(
            id=element["id"],
            labels=frozenset(element.get("labels", [])),
            properties=_properties(element),
            supertypes=tuple(element.get("supertypes", [])),
        )
        for element in data.get(_NODE_TYPES, [])
    ]
    edge_types = []
    for element in data.get(_EDGE_TYPES, []):
        cardinality = element.get("cardinality")
        edge_types.append(
            EdgeTypeDef(
                id=element["id"],
                labels=frozenset(element.get("labels", [])),
                properties=_properties(element),
                supertypes=tuple(element.get("supertypes", [])),
                source=element.get("source"),
                target=element.get("target"),
                cardinality=(
                    Cardinality(parse_interval(cardinality["out"]), parse_interval(cardinality["in"]))
                    if cardinality is not None
                    else None
                ),
            )
        )
    return Schema(node_types=tuple(node_types), edge_types=tuple(edge_types))


_TOP_LEVEL_ORDER = {"format": 0, "version": 1, _NODE_TYPES: 2, _EDGE_TYPES: 3}


def _error_order(error: Any) -> list[tuple[int, Any]]:
    """Order errors as they appear in a document: node types before edge types, by index."""
    path = list(error.absolute_path)
    order = [(0, _TOP_LEVEL_ORDER.get(path[0], len(_TOP_LEVEL_ORDER)))] if path else []
    return order + [(0, p) if isinstance(p, int) else (1, str(p)) for p in path[1:]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_schema_json(source: str | os.PathLike[str] | dict[str, Any]) -> Schema:
    """Read and validate a JSON schema.

    ``source`` is either a path to a UTF-8 JSON file or an already parsed
    ``dict``. Every violation of the schema format raises :class:`SchemaError`
    with the offending location in the message.
    """
    if isinstance(source, dict):
        data: Any = source
        origin = "<dict>"
    elif isinstance(source, (str, os.PathLike)):
        origin = os.fspath(source)
        with open(source, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SchemaError(f"invalid schema ({origin}): not valid UTF-8: {e}") from e
        data = _parse_json(text, origin)
    else:
        raise TypeError(f"source must be a path or a dict, not {type(source).__name__}")

    # R1-R6: the format, checked against the bundled JSON Schema. Keys are
    # checked first, as the JSON Schema cannot reject a non-string key.
    _raise(_check_keys(data, (), data), origin)
    errors = sorted(_validator().iter_errors(data), key=_error_order)
    _raise([f"{_format_path(tuple(e.absolute_path), data)}: {_format_error(e)}" for e in errors], origin)

    # R8, R9, R11, R13, then R10, then R12 (which needs an acyclic schema).
    _raise(_check_references(data), origin)
    _raise(_check_cycles(data), origin)
    schema = _build(data)
    _raise(_check_endpoints(schema, data), origin)
    return schema
