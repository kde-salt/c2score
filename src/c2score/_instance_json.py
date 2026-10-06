"""Saving and restoring Abs(G) as JSON (``Instance.to_json`` / ``Instance.from_json``).

The file holds the summary, not the graph::

    {
      "format": "c2score-instance",
      "version": 1,
      "node_types": [{"labels": [...], "properties": {key: {"type": ..., "required": ...}}}],
      "edge_types": [{"labels": [...], "source_labels": [...], "target_labels": [...],
                      "cardinality": {"out": "0..1", "in": "1..*"}, "properties": {...}}]
    }
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Union

from .cardinality import Cardinality, format_interval, parse_interval
from .datatypes import DataType
from .errors import InstanceLoadError
from .model import Instance, InstanceEdgeType, InstanceNodeType, PropertySpec

FORMAT = "c2score-instance"
VERSION = 1

PathLike = Union[str, "os.PathLike[str]"]


def _dump_properties(properties: Mapping[str, PropertySpec]) -> dict[str, Any]:
    return {
        key: {"type": spec.type.value, "required": spec.required}
        for key, spec in sorted(properties.items())
    }


def instance_to_dict(instance: Instance) -> dict[str, Any]:
    """Return the JSON-serializable form of ``instance``."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "node_types": [
            {"labels": sorted(node.labels), "properties": _dump_properties(node.properties)}
            for node in instance.node_types
        ],
        "edge_types": [
            {
                "labels": sorted(edge.labels),
                "source_labels": sorted(edge.source_labels),
                "target_labels": sorted(edge.target_labels),
                "cardinality": {
                    "out": format_interval(edge.cardinality.out),
                    "in": format_interval(edge.cardinality.in_),
                },
                "properties": _dump_properties(edge.properties),
            }
            for edge in instance.edge_types
        ],
    }


def save(instance: Instance, path: PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instance_to_dict(instance), f, ensure_ascii=False, indent=2)
        f.write("\n")


def _labels(value: Any, where: str) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(label, str) for label in value):
        raise ValueError(f"{where}: expected a list of strings, got {value!r}")
    return frozenset(value)


def _load_properties(value: Any, where: str) -> dict[str, PropertySpec]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object, got {value!r}")
    properties = {}
    for key, spec in value.items():
        if not isinstance(spec, dict) or set(spec) != {"type", "required"}:
            raise ValueError(f"{where}.{key}: expected {{\"type\", \"required\"}}, got {spec!r}")
        if not isinstance(spec["required"], bool):
            raise ValueError(f"{where}.{key}.required: expected true or false, got {spec['required']!r}")
        try:
            data_type = DataType(spec["type"])
        except ValueError:
            raise ValueError(f"{where}.{key}.type: unknown type {spec['type']!r}") from None
        properties[key] = PropertySpec(data_type, spec["required"])
    return properties


def _list(data: Mapping[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{key}: expected a list of objects")
    return value


def instance_from_dict(data: Any) -> Instance:
    """Rebuild an :class:`Instance` from :func:`instance_to_dict` output.

    Raises ``ValueError`` (or ``KeyError``) on malformed input.
    """
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"not a c2score instance file (expected \"format\": {FORMAT!r})")
    if data.get("version") != VERSION:
        raise ValueError(f"unsupported version {data.get('version')!r}; this c2score reads version {VERSION}")
    node_types = [
        InstanceNodeType(
            _labels(node["labels"], f"node_types[{i}].labels"),
            _load_properties(node["properties"], f"node_types[{i}].properties"),
        )
        for i, node in enumerate(_list(data, "node_types"))
    ]
    edge_types = []
    for i, edge in enumerate(_list(data, "edge_types")):
        where = f"edge_types[{i}]"
        cardinality = edge["cardinality"]
        if not isinstance(cardinality, dict) or set(cardinality) != {"out", "in"}:
            raise ValueError(f"{where}.cardinality: expected {{\"out\", \"in\"}}, got {cardinality!r}")
        edge_types.append(
            InstanceEdgeType(
                _labels(edge["labels"], f"{where}.labels"),
                _labels(edge["source_labels"], f"{where}.source_labels"),
                _labels(edge["target_labels"], f"{where}.target_labels"),
                Cardinality(parse_interval(cardinality["out"]), parse_interval(cardinality["in"])),
                _load_properties(edge["properties"], f"{where}.properties"),
            )
        )
    return Instance(tuple(node_types), tuple(edge_types))


def load(path: PathLike) -> Instance:
    origin = os.fspath(path)
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return instance_from_dict(json.loads(raw.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InstanceLoadError(f"invalid instance file ({origin}): not valid UTF-8 JSON: {e}") from e
    except KeyError as e:
        raise InstanceLoadError(f"invalid instance file ({origin}): missing key {e}") from e
    except ValueError as e:
        raise InstanceLoadError(f"invalid instance file ({origin}): {e}") from e
