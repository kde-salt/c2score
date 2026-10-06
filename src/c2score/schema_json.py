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


def _format_path(path: tuple[PathItem, ...], data: Any) -> str:
    """Format a JSON location such as ``edge_types[2] (id='managesType').source``."""
    if not path:
        return "<root>"
    text = ""
    for depth, item in enumerate(path):
        if isinstance(item, int):
            text += f"[{item}]"
        else:
            text += f".{item}" if text else item
        if depth == 1 and path[0] in (_NODE_TYPES, _EDGE_TYPES) and isinstance(item, int):
            type_id = _id_at(data, path[0], item)
            if type_id is not None:
                text += f" (id={type_id!r})"
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
        [f"{_format_path(path, data)}: duplicate key {key!r}" for path, key in duplicates],
        origin,
    )
    return data


# ---------------------------------------------------------------------------
# R8-R13: semantic checks
# ---------------------------------------------------------------------------


def _elements(data: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    return [
        (kind, index, element)
        for kind in (_NODE_TYPES, _EDGE_TYPES)
        for index, element in enumerate(data.get(kind, []))
    ]


def _check_references(data: dict[str, Any]) -> list[str]:
    """R8 (unique ids), R9 (supertypes), R11 (endpoints), and R13 (type names)."""
    problems: list[str] = []
    elements = _elements(data)

    # R8: ids are unique across node and edge types.
    kind_of: dict[str, str] = {}
    first_at: dict[str, str] = {}
    for kind, index, element in elements:
        type_id = element["id"]
        where = _format_path((kind, index), data)
        if type_id in kind_of:
            problems.append(f"{where}.id: duplicate id {type_id!r} (already used by {first_at[type_id]})")
        else:
            kind_of[type_id] = kind
            first_at[type_id] = where

    for kind, index, element in elements:
        type_id = element["id"]
        noun = _NOUN[kind]

        # R9: supertypes exist, keep the kind, and are not the type itself.
        for position, parent in enumerate(element.get("supertypes", [])):
            where = _format_path((kind, index, "supertypes", position), data)
            if parent == type_id:
                problems.append(f"{where}: a type cannot be its own supertype")
            elif parent not in kind_of:
                problems.append(f"{where}: unknown id {parent!r}")
            elif kind_of[parent] != kind:
                problems.append(
                    f"{where}: {parent!r} is {_NOUN[kind_of[parent]]}; "
                    f"{noun} can only inherit from {kind.replace('_', ' ')}"
                )

        # R11: source and target refer to node types.
        if kind == _EDGE_TYPES:
            for end in ("source", "target"):
                if end not in element:
                    continue
                where = _format_path((kind, index, end), data)
                endpoint = element[end]
                if endpoint not in kind_of:
                    problems.append(f"{where}: unknown id {endpoint!r}")
                elif kind_of[endpoint] != _NODE_TYPES:
                    problems.append(f"{where}: {endpoint!r} is not a node type")

        # R13: type names are known.
        for key, spec in element.get("properties", {}).items():
            if "type" in spec:
                try:
                    parse_schema_type(spec["type"])
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
    for root in parents:
        if root in done:
            continue
        path = [root]
        on_path = {root}
        pending = [iter(parents[root])]
        while pending:
            parent = next(pending[-1], None)
            if parent is None:
                pending.pop()
                finished = path.pop()
                on_path.discard(finished)
                done.add(finished)
            elif parent in on_path:
                cycle = path[path.index(parent):] + [parent]
                where = _format_path((*position[parent], "supertypes"), data)
                problems.append(f"{where}: inheritance cycle {' -> '.join(cycle)}")
            elif parent not in done:
                path.append(parent)
                on_path.add(parent)
                pending.append(iter(parents[parent]))
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
            type=parse_schema_type(spec["type"]) if "type" in spec else DataType.ANY,
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

    # R1-R6: the format, checked against the bundled JSON Schema.
    errors = sorted(_validator().iter_errors(data), key=_error_order)
    _raise([f"{_format_path(tuple(e.absolute_path), data)}: {e.message}" for e in errors], origin)

    # R8, R9, R11, R13, then R10, then R12 (which needs an acyclic schema).
    _raise(_check_references(data), origin)
    _raise(_check_cycles(data), origin)
    schema = _build(data)
    _raise(_check_endpoints(schema, data), origin)
    return schema
