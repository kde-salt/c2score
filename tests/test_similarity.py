import json
from fractions import Fraction
from pathlib import Path

import pytest

from c2score import load_schema_json
from c2score import similarity as sim
from c2score.cardinality import ZERO_MANY, ZERO_ONE, Cardinality, d_card
from c2score.datatypes import DataType
from c2score.flatten import flatten
from c2score.model import Instance, InstanceEdgeType, InstanceNodeType, PropertySpec, _abs_from_graph
from c2score.similarity import SimilarityParams

DATA = Path(__file__).parent / "data"
D = DataType
P = PropertySpec
WITH_CARD = SimilarityParams(edge_weights=(1 / 3, 1 / 3, 1 / 3))


def approx(value):
    return pytest.approx(float(value), abs=1e-12)


def fig1a():
    graph = json.loads((DATA / "fig1a.json").read_text(encoding="utf-8"))
    return _abs_from_graph(graph["nodes"], graph["edges"])


def example1_flat():
    return flatten(load_schema_json(DATA / "example1.json"))


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (set(), set(), 1),
        ({"A"}, set(), 0),
        ({"A"}, {"A"}, 1),
        ({"A", "B"}, {"A"}, Fraction(2, 3)),
        ({"A"}, {"B"}, 0),
    ],
)
def test_dice(a, b, expected):
    assert sim.dice(frozenset(a), frozenset(b)) == approx(expected)


# D_type: match, mismatch, no common key, no key at all.
@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        ({"x": P(D.INTEGER)}, {"x": P(D.INTEGER)}, 1),
        ({"x": P(D.INTEGER), "y": P(D.STRING)}, {"x": P(D.INTEGER), "y": P(D.FLOAT)}, Fraction(1, 2)),
        ({"x": P(D.INTEGER)}, {"y": P(D.INTEGER)}, 0),
        ({}, {}, 1),
        ({"x": P(D.INTEGER)}, {}, 0),
    ],
)
def test_d_type(p, q, expected):
    assert sim.d_type(p, q, allow_any=True) == approx(expected)


@pytest.mark.parametrize(
    ("instance_type", "schema_type", "allowed", "strict"),
    [
        (D.STRING, D.STRING, 1, 1),
        (D.STRING, D.INTEGER, 0, 0),
        (D.STRING, D.ANY, 1, 0),
        (D.ANY, D.STRING, 1, 0),
        (D.ANY, D.ANY, 1, 0),
    ],
)
def test_delta(instance_type, schema_type, allowed, strict):
    assert sim.delta(instance_type, schema_type, allow_any=True) == allowed
    assert sim.delta(instance_type, schema_type, allow_any=False) == strict


def test_flags():
    p = {"a": P(required=True), "b": P(required=True), "c": P(required=False)}
    q = {"a": P(required=True), "c": P(required=True)}
    assert sim.d_req(p, q) == approx(Fraction(2, 4))
    assert sim.d_opt(p, q) == approx(0)
    assert sim.d_flag(p, q) == approx(Fraction(1, 4))


def test_d_prop_without_types_is_d_flag():
    p = {"x": P(D.INTEGER), "y": P(D.STRING, False)}
    q = {"x": P(D.STRING)}
    params = SimilarityParams(use_types=False)
    assert sim.d_prop(p, q, params) == approx(sim.d_flag(p, q))
    assert sim.d_prop(p, q, SimilarityParams()) == approx((sim.d_flag(p, q) + 0) / 2)


def test_sim_v_user_against_user_type():
    user = fig1a().node_type({"USER"})
    user_type = next(n for n in example1_flat().node_types if n.id == "userType")
    assert sim.dice(user.labels, user_type.labels) == approx(1)
    assert sim.d_flag(user.properties, user_type.properties) == approx(Fraction(1, 2))
    assert sim.d_type(user.properties, user_type.properties, True) == approx(1)
    assert sim.d_prop(user.properties, user_type.properties, WITH_CARD) == approx(Fraction(3, 4))
    assert sim.sim_v(user, user_type, WITH_CARD) == approx(Fraction(7, 8))
    assert sim.sim_v(user, user_type, SimilarityParams(use_types=False)) == approx(Fraction(3, 4))


def test_sim_e_likes_against_likes_type():
    instance = fig1a()
    likes = next(e for e in instance.edge_types if e.labels == {"LIKES"} and e.source_labels == {"USER"})
    element = next(e for e in example1_flat().edge_types if e.schema_type == "likesType" and e.source.id == "userType")
    assert d_card(likes.cardinality, element.cardinality) == approx(Fraction(1, 2))
    # D_attr = 1, D_end = (7/8 + 1) / 2 = 15/16, D_card = 1/2.
    assert sim.sim_e(likes, element, instance, WITH_CARD) == approx(Fraction(13, 16))


def test_sim_v_zero_when_labels_do_not_overlap():
    params = SimilarityParams(label_weight=0.0)
    node = InstanceNodeType({"A"})
    schema = load_schema_json({"node_types": [{"id": "b", "labels": ["B"]}]})
    assert sim.sim_v(node, flatten(schema).node_types[0], params) == 0
    # With overlapping labels and alpha = 0, only D_prop counts.
    schema = load_schema_json({"node_types": [{"id": "a", "labels": ["A", "B"]}]})
    assert sim.sim_v(node, flatten(schema).node_types[0], params) == approx(1)


def two_node_instance(source_labels, target_labels, edge_labels=("R",)):
    return Instance(
        node_types=[InstanceNodeType(source_labels), InstanceNodeType(target_labels)]
        if set(source_labels) != set(target_labels)
        else [InstanceNodeType(source_labels)],
        edge_types=[InstanceEdgeType(edge_labels, source_labels, target_labels, Cardinality(ZERO_ONE, ZERO_ONE))],
    )


def edge_schema(source_labels, target_labels):
    return flatten(
        load_schema_json(
            {
                "node_types": [
                    {"id": "s", "labels": list(source_labels)},
                    {"id": "t", "labels": list(target_labels)},
                ],
                "edge_types": [{"id": "r", "labels": ["R"], "source": "s", "target": "t"}],
            }
        )
    )


def test_endpoint_filter():
    instance = two_node_instance({"A"}, {"B"})
    (edge,) = instance.edge_types
    assert sim.sim_e(edge, edge_schema({"A"}, {"B"}).edge_types[0], instance, SimilarityParams()) == approx(1)
    assert sim.sim_e(edge, edge_schema({"A"}, {"C"}).edge_types[0], instance, SimilarityParams()) == 0
    assert sim.sim_e(edge, edge_schema({"C"}, {"B"}).edge_types[0], instance, SimilarityParams()) == 0


def test_unlabelled_endpoints_pass_the_filter():
    instance = two_node_instance(set(), {"B"})
    (edge,) = instance.edge_types
    assert sim.sim_e(edge, edge_schema(set(), {"B"}).edge_types[0], instance, SimilarityParams()) == approx(1)
    # Only one side unlabelled: dice is 0.
    assert sim.sim_e(edge, edge_schema({"A"}, {"B"}).edge_types[0], instance, SimilarityParams()) == 0


def test_label_with_colon():
    instance = two_node_instance({"a:b"}, {"c"}, edge_labels=("R::S",))
    (edge,) = instance.edge_types
    flat = flatten(
        load_schema_json(
            {
                "node_types": [{"id": "s", "labels": ["a:b"]}, {"id": "t", "labels": ["c"]}],
                "edge_types": [{"id": "r", "labels": ["R::S"], "source": "s", "target": "t"}],
            }
        )
    )
    assert sim.sim_e(edge, flat.edge_types[0], instance, SimilarityParams()) == approx(1)
    split = edge_schema({"a", "b"}, {"c"})
    assert sim.sim_v(instance.node_type({"a:b"}), split.node_types[0], SimilarityParams()) == 0


def test_missing_cardinality_counts_as_zero_many():
    instance = two_node_instance({"A"}, {"B"})
    (edge,) = instance.edge_types
    element = edge_schema({"A"}, {"B"}).edge_types[0]
    assert element.cardinality is None
    expected = (1 + 1 + d_card(Cardinality(ZERO_ONE, ZERO_ONE), Cardinality(ZERO_MANY, ZERO_MANY))) / 3
    assert sim.sim_e(edge, element, instance, WITH_CARD) == approx(expected)


def test_d_card_is_skipped_when_its_weight_is_zero(monkeypatch):
    def fail(*args):
        raise AssertionError("d_card must not be called")

    monkeypatch.setattr(sim, "d_card", fail)
    instance = two_node_instance({"A"}, {"B"})
    (edge,) = instance.edge_types
    assert sim.sim_e(edge, edge_schema({"A"}, {"B"}).edge_types[0], instance, SimilarityParams()) == approx(1)


def test_instance_any_scores_zero_without_allow_any():
    node = InstanceNodeType({"A"}, {"x": P(D.ANY)})
    flat = flatten(load_schema_json({"node_types": [{"id": "a", "labels": ["A"], "properties": {"x": {"type": "STRING"}}}]}))
    assert sim.sim_v(node, flat.node_types[0], SimilarityParams()) == approx(1)
    # D_flag 1, D_type 0 -> D_prop 1/2 -> 1/2 + 1/4.
    assert sim.sim_v(node, flat.node_types[0], SimilarityParams(allow_any=False)) == approx(Fraction(3, 4))


def test_similarity_cache():
    instance = fig1a()
    flat = example1_flat()
    cached = sim.Similarity(instance, WITH_CARD)
    for t in instance.edge_types:
        for s in flat.edge_types:
            assert cached.edge(t, s) == sim.sim_e(t, s, instance, WITH_CARD)
            assert cached.edge(t, s) == sim.sim_e(t, s, instance, WITH_CARD)
