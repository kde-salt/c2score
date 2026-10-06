import json
from pathlib import Path

import pytest

from c2score import Instance, InstanceLoadError
from c2score.cardinality import ONE_MANY, ZERO_ONE, Cardinality
from c2score.datatypes import DataType
from c2score.model import InstanceEdgeType, InstanceNodeType, PropertySpec, _abs_from_graph

DATA = Path(__file__).parent / "data"


def fig1a():
    graph = json.loads((DATA / "fig1a.json").read_text(encoding="utf-8"))
    return _abs_from_graph(graph["nodes"], graph["edges"])


def varied():
    """Unlabelled nodes, ANY, optional properties, and labels outside ASCII."""
    return Instance(
        node_types=[
            InstanceNodeType(set(), {"x": PropertySpec(DataType.ANY, False)}),
            InstanceNodeType({"人", "a:b"}, {"t": PropertySpec(DataType.DATETIME)}),
        ],
        edge_types=[
            InstanceEdgeType({"R"}, set(), {"人", "a:b"}, Cardinality(ZERO_ONE, ONE_MANY), {"w": PropertySpec(DataType.FLOAT)}),
        ],
    )


@pytest.mark.parametrize("instance", [fig1a(), varied(), Instance()], ids=["fig1a", "varied", "empty"])
def test_round_trip(tmp_path, instance):
    path = tmp_path / "instance.json"
    instance.to_json(path)
    restored = Instance.from_json(path)
    assert restored == instance
    assert restored.node_types == instance.node_types
    assert restored.edge_types == instance.edge_types
    # str paths work too.
    instance.to_json(str(path))
    assert Instance.from_json(str(path)) == instance


def test_file_format(tmp_path):
    path = tmp_path / "instance.json"
    fig1a().to_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "c2score-instance"
    assert data["version"] == 1
    assert data["node_types"][0] == {
        "labels": ["OFFICIAL", "USER"],
        "properties": {
            "bio": {"type": "STRING", "required": False},
            "id": {"type": "INTEGER", "required": True},
            "verifiedAt": {"type": "DATETIME", "required": True},
        },
    }
    assert data["edge_types"][0]["cardinality"] == {"out": "0..1", "in": "0..1"}  # CREATES from v4 only


def write(tmp_path, data):
    path = tmp_path / "instance.json"
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


def valid():
    return {
        "format": "c2score-instance",
        "version": 1,
        "node_types": [{"labels": ["A"], "properties": {}}],
        "edge_types": [
            {
                "labels": ["R"],
                "source_labels": ["A"],
                "target_labels": ["A"],
                "cardinality": {"out": "0..1", "in": "0..1"},
                "properties": {},
            }
        ],
    }


def broken(change):
    data = valid()
    change(data)
    return data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("{", "not valid UTF-8 JSON"),
        ([], "not a c2score instance file"),
        (broken(lambda d: d.pop("format")), "not a c2score instance file"),
        (broken(lambda d: d.update(version=2)), "unsupported version 2"),
        (broken(lambda d: d.update(version="1")), "unsupported version '1'"),
        (broken(lambda d: d.update(version=True)), "unsupported version True"),
        (broken(lambda d: d.update(version=1.0)), "unsupported version 1.0"),
        (broken(lambda d: d.pop("version")), "unsupported version None"),
        (broken(lambda d: d.pop("node_types")), "missing key 'node_types'"),
        (broken(lambda d: d.update(node_types={})), "node_types: expected a list"),
        (broken(lambda d: d.update(comment="x")), "unknown key 'comment'"),
        (broken(lambda d: d["node_types"][0].pop("labels")), r"node_types\[0\]: missing key 'labels'"),
        (broken(lambda d: d["node_types"][0].update(label=["A"])), r"node_types\[0\]: unknown key 'label'"),
        (broken(lambda d: d["edge_types"][0].pop("cardinality")), r"edge_types\[0\]: missing key 'cardinality'"),
        (
            broken(lambda d: d["edge_types"][0].update(sourceLabels=["A"], weight=1)),
            r"edge_types\[0\]: unknown keys 'sourceLabels', 'weight'",
        ),
        (broken(lambda d: d["node_types"][0].update(labels="A")), r"node_types\[0\].labels"),
        (broken(lambda d: d["node_types"][0].update(properties={"x": {"type": "TEXT", "required": True}})), "unknown type 'TEXT'"),
        (broken(lambda d: d["node_types"][0].update(properties={"x": {"type": "STRING", "required": 1}})), "required"),
        (broken(lambda d: d["edge_types"][0].update(cardinality={"out": "2..3", "in": "0..1"})), "invalid interval"),
        (broken(lambda d: d["edge_types"][0].update(source_labels=["B"])), "no node type"),
        (broken(lambda d: d["node_types"].append({"labels": ["A"], "properties": {}})), "duplicate"),
    ],
)
def test_malformed_file(tmp_path, data, message):
    with pytest.raises(InstanceLoadError, match=message):
        Instance.from_json(write(tmp_path, data))


def test_valid_example_loads(tmp_path):
    assert len(Instance.from_json(write(tmp_path, valid())).edge_types) == 1


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Instance.from_json(tmp_path / "missing.json")
