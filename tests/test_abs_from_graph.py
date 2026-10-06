import json
from pathlib import Path

import pytest

from c2score.cardinality import ONE_MANY, ONE_ONE, ZERO_MANY, ZERO_ONE, Cardinality
from c2score.datatypes import DataType
from c2score.model import InstanceEdgeType, InstanceNodeType, PropertySpec, _abs_from_graph

DATA = Path(__file__).parent / "data"
D = DataType


def load_fig1a():
    graph = json.loads((DATA / "fig1a.json").read_text(encoding="utf-8"))
    return _abs_from_graph(graph["nodes"], graph["edges"])


def node(node_id, labels, **properties):
    return {"id": node_id, "labels": list(labels), "properties": properties}


def edge(edge_id, labels, source, target, **properties):
    return {"id": edge_id, "labels": list(labels), "source": source, "target": target, "properties": properties}


def test_fig1a_node_types():
    instance = load_fig1a()
    assert instance.node_types == (
        InstanceNodeType(
            {"OFFICIAL", "USER"},
            {
                "id": PropertySpec(D.INTEGER, True),
                "verifiedAt": PropertySpec(D.DATETIME, True),
                "bio": PropertySpec(D.STRING, False),
            },
        ),
        InstanceNodeType({"POST"}, {"text": PropertySpec(D.STRING, True)}),
        InstanceNodeType({"USER"}, {"id": PropertySpec(D.INTEGER, True), "bio": PropertySpec(D.STRING, False)}),
    )


def test_fig1a_edge_types():
    instance = load_fig1a()
    rounded = Cardinality(ZERO_ONE, ZERO_ONE)
    assert instance.edge_types == (
        InstanceEdgeType({"CREATES"}, {"OFFICIAL", "USER"}, {"POST"}, rounded),
        InstanceEdgeType({"CREATES"}, {"USER"}, {"POST"}, rounded),
        InstanceEdgeType({"LIKES"}, {"OFFICIAL", "USER"}, {"POST"}, rounded),
        InstanceEdgeType({"LIKES"}, {"USER"}, {"POST"}, rounded),
    )


def test_node_type_lookup():
    instance = load_fig1a()
    assert instance.node_type(["USER", "OFFICIAL"]).properties["verifiedAt"].type is D.DATETIME
    with pytest.raises(KeyError):
        instance.node_type({"OFFICIAL"})


def test_null_means_absent():
    instance = _abs_from_graph([node("a", ["A"], x=1, y=None), node("b", ["A"], x=None)], [])
    assert instance.node_types[0].properties == {"x": PropertySpec(D.INTEGER, False)}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2.5], D.FLOAT),
        ([1, "x"], D.ANY),
        ([1, 2.5, "x"], D.ANY),
        ([True, 1], D.ANY),
        ([{"$datetime": "2025-12-31"}, {"$datetime": "2025-12-31T10:00:00Z"}], D.DATETIME),
        ([[1, 2], [3]], D.ANY),
    ],
)
def test_property_type_is_unified_in_a_group(values, expected):
    instance = _abs_from_graph([node(f"n{i}", ["A"], p=v) for i, v in enumerate(values)], [])
    assert instance.node_types[0].properties["p"] == PropertySpec(expected, True)


def test_unlabelled_nodes_and_labels_with_colons():
    instance = _abs_from_graph(
        [node("a", []), node("b", ["x:y"]), node("c", ["x", "y"])],
        [edge("e", ["R::S"], "a", "b")],
    )
    assert [t.labels for t in instance.node_types] == [frozenset(), frozenset({"x", "y"}), frozenset({"x:y"})]
    (only,) = instance.edge_types
    assert (only.labels, only.source_labels, only.target_labels) == (
        frozenset({"R::S"}),
        frozenset(),
        frozenset({"x:y"}),
    )


def test_degree_intervals_count_nodes_without_edges():
    # Three A nodes: a1 has two R edges, a2 has one, a3 has none.
    nodes = [node("a1", ["A"]), node("a2", ["A"]), node("a3", ["A"]), node("b1", ["B"]), node("b2", ["B"])]
    edges = [edge("e1", ["R"], "a1", "b1"), edge("e2", ["R"], "a1", "b2"), edge("e3", ["R"], "a2", "b1")]
    (r,) = _abs_from_graph(nodes, edges).edge_types
    # Out: [0, 2] -> 0..*. In: b1 has 2, b2 has 1 -> [1, 2] -> 1..*.
    assert r.cardinality == Cardinality(ZERO_MANY, ONE_MANY)


def test_degree_intervals_are_per_edge_group():
    nodes = [node("a", ["A"]), node("b", ["B"])]
    edges = [edge("e1", ["R"], "a", "b"), edge("e2", ["S"], "a", "b"), edge("e3", ["S"], "a", "b")]
    r, s = _abs_from_graph(nodes, edges).edge_types
    assert r.cardinality == Cardinality(ONE_ONE, ONE_ONE)
    assert s.cardinality == Cardinality(ONE_MANY, ONE_MANY)


def test_edge_properties():
    nodes = [node("a", ["A"]), node("b", ["B"])]
    edges = [edge("e1", ["R"], "a", "b", w=1), edge("e2", ["R"], "b", "a"), edge("e3", ["R"], "a", "b", w=0.5)]
    instance = _abs_from_graph(nodes, edges)
    a_to_b = next(t for t in instance.edge_types if t.source_labels == {"A"})
    assert a_to_b.properties == {"w": PropertySpec(D.FLOAT, True)}


def test_empty_graph():
    instance = _abs_from_graph([], [])
    assert instance.node_types == () and instance.edge_types == ()


@pytest.mark.parametrize(
    ("nodes", "edges", "message"),
    [
        ([node("a", ["A"]), node("a", ["B"])], [], "duplicate node id 'a'"),
        ([node("a", ["A"])], [edge("e", ["R"], "a", "x")], "unknown target node id 'x'"),
        ([node("a", ["A"])], [edge("e", ["R"], "a", "a"), edge("e", ["R"], "a", "a")], "duplicate edge id 'e'"),
    ],
)
def test_invalid_graphs(nodes, edges, message):
    with pytest.raises(ValueError, match=message):
        _abs_from_graph(nodes, edges)
