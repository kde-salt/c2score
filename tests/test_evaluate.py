import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from c2score import EvaluationResult, ParameterError, SchemaError, evaluate, load_schema_json
from c2score.conciseness import conciseness, reduce_schema
from c2score.coverage import cov_e, cov_v
from c2score.flatten import flatten
from c2score.model import Instance, Schema, _abs_from_graph
from c2score.similarity import Similarity, SimilarityParams

DATA = Path(__file__).parent / "data"
THIRDS = (1 / 3, 1 / 3, 1 / 3)


def approx(value):
    return pytest.approx(float(value), abs=1e-12)


def fig1a():
    graph = json.loads((DATA / "fig1a.json").read_text(encoding="utf-8"))
    return _abs_from_graph(graph["nodes"], graph["edges"])


def example(n):
    return load_schema_json(DATA / f"example{n}.json")


def graph(nodes=(), edges=()):
    """Nodes are (id, labels, properties); edges are (id, labels, source, target, properties)."""
    return _abs_from_graph(
        [{"id": i, "labels": list(labels), "properties": dict(props)} for i, labels, props in nodes],
        [
            {"id": i, "labels": list(labels), "source": s, "target": t, "properties": dict(props)}
            for i, labels, s, t, props in edges
        ],
    )


def schema(node_types=(), edge_types=()):
    return load_schema_json({"node_types": list(node_types), "edge_types": list(edge_types)})


# ---------------------------------------------------------------------------
# Section 4.9: Fig. 1(a) against example 1
# ---------------------------------------------------------------------------


def fig1a_parts():
    instance, s = fig1a(), example(1)
    similarity = Similarity(instance, SimilarityParams(edge_weights=THIRDS))
    flat = flatten(s)
    return instance, s, similarity, flat


def test_fig1a_coverage():
    _, _, similarity, flat = fig1a_parts()
    value, matches = cov_v(flat, similarity)
    assert value == approx(Fraction(11, 12))
    # Instance order: {OFFICIAL, USER}, {POST}, {USER}.
    assert [m.similarity for m in matches] == [approx(Fraction(7, 8)), approx(1), approx(Fraction(7, 8))]
    value, matches = cov_e(flat, similarity)
    assert value == approx(Fraction(5, 6))
    # Instance order: CREATES {OFFICIAL,USER}, CREATES {USER}, LIKES {OFFICIAL,USER}, LIKES {USER}.
    expected = [Fraction(41, 48), Fraction(41, 48), Fraction(13, 16), Fraction(13, 16)]
    assert [m.similarity for m in matches] == [approx(x) for x in expected]


def test_fig1a_conciseness():
    _, s, similarity, flat = fig1a_parts()
    base_v, base_e = cov_v(flat, similarity)[0], cov_e(flat, similarity)[0]
    result = conciseness(s, flat, base_v, base_e, similarity, 0.15)
    assert result.theta_v == approx(Fraction(11, 320))  # 0.034375
    assert result.theta_e == approx(Fraction(1, 48))
    nodes = {r.type_id: (r.delta_v, r.delta_e, r.redundant) for r in result.node_removals}
    assert nodes["userType"] == (approx(Fraction(13, 36)), approx(Fraction(5, 6)), False)
    assert nodes["officialType"] == (approx(Fraction(5, 72)), approx(Fraction(5, 288)), False)
    assert nodes["postType1"] == (approx(0), approx(Fraction(13, 32)), False)
    assert nodes["postType2"] == (approx(0), approx(0), True)
    edges = {r.type_id: (r.delta_v, r.delta_e, r.redundant) for r in result.edge_removals}
    assert edges["likesType"] == (approx(0), approx(Fraction(13, 32)), False)
    assert edges["createsType1"] == (approx(0), approx(0), True)
    assert edges["createsType2"] == (approx(0), approx(0), True)
    assert result.con_v == approx(Fraction(3, 4))
    assert result.con_e == approx(Fraction(1, 3))


def test_fig1a_evaluate():
    result = evaluate(fig1a(), example(1))
    assert isinstance(result, EvaluationResult)
    assert result.cov_v == approx(Fraction(11, 12))
    assert result.cov_e == approx(Fraction(5, 6))
    assert result.con_v == approx(Fraction(3, 4))
    assert result.con_e == approx(Fraction(1, 3))
    assert result.coverage == approx(Fraction(7, 8))
    assert result.conciseness == approx(Fraction(13, 24))
    assert result.c2 == approx(Fraction(91, 136))
    assert result.redundant_types == {"node_types": ["postType2"], "edge_types": ["createsType1", "createsType2"]}


def test_fig1a_best_matches():
    result = evaluate(fig1a(), example(1))
    nodes = result.best_matches["node_types"]
    assert [(m["labels"], m["schema_type"]) for m in nodes] == [
        (["OFFICIAL", "USER"], "officialType"),
        (["POST"], "postType1"),  # tie with postType2: the first in id order wins
        (["USER"], "userType"),
    ]
    edges = result.best_matches["edge_types"]
    assert edges[0] == {
        "labels": ["CREATES"],
        "source_labels": ["OFFICIAL", "USER"],
        "target_labels": ["POST"],
        "schema_type": "createsType1",
        "source": "officialType",
        "target": "postType1",
        "similarity": approx(Fraction(41, 48)),
    }
    assert [(m["schema_type"], m["source"], m["target"]) for m in edges[1:]] == [
        ("createsType1", "userType", "postType1"),
        ("likesType", "officialType", "postType1"),
        ("likesType", "userType", "postType1"),
    ]


def test_tie_between_nodes_follows_id_order_not_written_order():
    instance = graph([("x", ["A"], {})])
    s = schema([{"id": "zType", "labels": ["A"]}, {"id": "aType", "labels": ["A"]}])
    (match,) = evaluate(instance, s).best_matches["node_types"]
    assert (match["schema_type"], match["similarity"]) == ("aType", approx(1))


def test_tie_between_edge_elements_follows_smallest_origin_not_written_order():
    # The two elements differ only in cardinality, which alpha_C = 0 ignores.
    nodes = [{"id": "s", "labels": ["S"]}, {"id": "t", "labels": ["T"]}]
    s = schema(
        nodes,
        [
            {"id": "zEdge", "labels": ["R"], "source": "s", "target": "t", "cardinality": {"out": "0..*", "in": "0..*"}},
            {"id": "aEdge", "labels": ["R"], "source": "s", "target": "t"},
        ],
    )
    instance = graph([("x", ["S"], {}), ("y", ["T"], {})], [("e", ["R"], "x", "y", {})])
    (match,) = evaluate(instance, s, edge_weights=(0.5, 0.5, 0)).best_matches["edge_types"]
    assert (match["schema_type"], match["similarity"]) == ("aEdge", approx(1))


def test_tie_between_edge_elements_follows_endpoint_id_order():
    # One edge definition expands to (s, p), (s, zChild), (s, aChild); the two children tie.
    s = schema(
        [
            {"id": "s", "labels": ["S"]},
            {"id": "p", "labels": ["P"]},
            {"id": "zChild", "labels": ["C"], "supertypes": ["p"]},
            {"id": "aChild", "labels": ["C"], "supertypes": ["p"]},
        ],
        [{"id": "r", "labels": ["R"], "source": "s", "target": "p"}],
    )
    instance = graph([("x", ["S"], {}), ("y", ["C", "P"], {})], [("e", ["R"], "x", "y", {})])
    (match,) = evaluate(instance, s).best_matches["edge_types"]
    assert (match["schema_type"], match["source"], match["target"]) == ("r", "s", "aChild")
    assert match["similarity"] == approx(1)


def test_fig1a_without_types():
    result = evaluate(fig1a(), example(1), use_types=False)
    user = result.best_matches["node_types"][2]
    assert user["similarity"] == approx(Fraction(3, 4))


def test_fig1a_params_and_to_dict():
    result = evaluate(fig1a(), example(1))
    assert result.params == {
        "label_weight": 0.5,
        "edge_weights": THIRDS,
        "edge_weights_mode": "auto",
        "redundancy_ratio": 0.15,
        "use_types": True,
        "allow_any": True,
    }
    data = json.loads(json.dumps(result.to_dict()))
    assert data["c2"] == result.c2
    assert data["params"]["edge_weights"] == list(THIRDS)
    assert data["best_matches"] == result.best_matches
    with pytest.raises(AttributeError):
        result.c2 = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Removing definitions (Section 4.7, step 3)
# ---------------------------------------------------------------------------


def test_removing_user_type_removes_its_edges():
    reduced = reduce_schema(example(1), "userType")
    assert [n.id for n in reduced.node_types] == ["officialType", "postType1", "postType2"]
    assert reduced.edge_types == ()
    official = next(n for n in flatten(reduced).node_types if n.id == "officialType")
    assert official.labels == {"OFFICIAL"}
    assert set(official.properties) == {"verifiedAt"}


def test_removing_employee_type_cascades():
    reduced = reduce_schema(example(3), "employeeType")
    assert reduced.edge_types == ()
    assert reduced["managerType"].supertypes == ("shareholderType",)
    flat = flatten(reduced)
    manager = next(n for n in flat.node_types if n.id == "managerType")
    assert manager.labels == {"Manager", "Shareholder", "Person"}


def test_removing_a_parent_edge_type():
    reduced = reduce_schema(example(3), "worksAtType")
    # managesType loses its inherited target, so it goes too.
    assert reduced.edge_types == ()
    reduced = reduce_schema(example(3), "managesType")
    assert [e.id for e in reduced.edge_types] == ["worksAtType"]


def test_identical_definitions_are_both_redundant():
    instance = graph([("a", ["A"], {})])
    s = schema([{"id": "a1", "labels": ["A"]}, {"id": "a2", "labels": ["A"]}])
    result = evaluate(instance, s)
    assert result.redundant_types["node_types"] == ["a1", "a2"]
    assert result.con_v == 0


# ---------------------------------------------------------------------------
# Edge weights
# ---------------------------------------------------------------------------


def test_weights_without_cardinality():
    result = evaluate(fig1a(), example(2))
    assert result.params["edge_weights"] == (0.5, 0.5, 0.0)
    assert result.params["edge_weights_mode"] == "auto"


def test_weights_given_by_the_user():
    result = evaluate(fig1a(), example(1), edge_weights=(0.5, 0.5, 0))
    assert result.params["edge_weights"] == (0.5, 0.5, 0.0)
    assert result.params["edge_weights_mode"] == "user"
    # Without D_card: D_attr 1 and D_end 15/16 for LIKES {USER} against likesType.
    assert result.best_matches["edge_types"][3]["similarity"] == approx(Fraction(31, 32))


def test_auto_weights_are_kept_when_the_only_cardinality_is_removed():
    nodes = [{"id": "a", "labels": ["A"]}, {"id": "b", "labels": ["B"]}]
    s = schema(
        nodes,
        [
            {"id": "r1", "labels": ["R"], "source": "a", "target": "b", "cardinality": {"out": "0..1", "in": "0..1"}},
            {"id": "r2", "labels": ["R"], "source": "a", "target": "b"},
        ],
    )
    instance = graph([("x", ["A"], {}), ("y", ["B"], {})], [("e", ["R"], "x", "y", {})])
    result = evaluate(instance, s)
    # Observed 1..1 / 1..1: sim(r1) = (1 + 1 + 3/4) / 3, sim(r2) = (1 + 1 + 1/4) / 3.
    assert result.cov_e == approx(Fraction(11, 12))
    # Removing r1 leaves r2 with the same weights: Delta_E = 1/6 > theta_E = 0.06875.
    # Re-deciding the weights would give (0.5, 0.5, 0), Delta_E < 0 and r1 redundant.
    assert result.redundant_types["edge_types"] == ["r2"]


# ---------------------------------------------------------------------------
# ANY
# ---------------------------------------------------------------------------


def test_schema_any_is_rejected_without_allow_any():
    s = schema(
        [
            {"id": "a", "labels": ["A"], "properties": {"x": {}, "y": {"type": "STRING"}, "z": {"type": "any"}}},
            {"id": "b", "labels": ["B"]},
        ],
        [{"id": "r", "labels": ["R"], "source": "a", "target": "b", "properties": {"w": {"required": False}}}],
    )
    with pytest.raises(SchemaError) as info:
        evaluate(fig1a(), s, allow_any=False)
    message = str(info.value)
    assert "a.x" in message and "a.z" in message and "r.w" in message
    assert "a.y" not in message
    # Ignored without types.
    evaluate(fig1a(), s, allow_any=False, use_types=False)
    evaluate(fig1a(), s)


def test_allow_any_false_with_concrete_types():
    result = evaluate(fig1a(), example(1), allow_any=False)
    assert result.c2 == approx(Fraction(91, 136))


def test_instance_any_without_allow_any():
    instance = graph([("a", ["A"], {"x": [1]})])
    s = schema([{"id": "a", "labels": ["A"], "properties": {"x": {"type": "STRING"}}}])
    assert evaluate(instance, s).cov_v == approx(1)
    assert evaluate(instance, s, allow_any=False).cov_v == approx(Fraction(3, 4))


# ---------------------------------------------------------------------------
# Boundary cases E1-E4 and B3 (Section 7)
# ---------------------------------------------------------------------------


def test_e1_instance_without_edges():
    instance = graph([("x", ["A"], {})])
    s = schema([{"id": "a", "labels": ["A"]}], [{"id": "r", "labels": ["R"], "source": "a", "target": "a"}])
    result = evaluate(instance, s)
    assert (result.cov_v, result.cov_e, result.con_v, result.con_e) == (approx(1), approx(1), approx(1), approx(0))
    assert result.c2 == approx(Fraction(2, 3))


def test_e2_schema_without_edge_definitions():
    instance = graph([("x", ["A"], {})], [("e", ["R"], "x", "x", {})])
    s = schema([{"id": "a", "labels": ["A"]}])
    flat = flatten(s)
    similarity = Similarity(instance, SimilarityParams())
    base_v, base_e = cov_v(flat, similarity)[0], cov_e(flat, similarity)[0]
    result = conciseness(s, flat, base_v, base_e, similarity, 0.15)
    assert base_e == 0
    assert result.theta_e == 0
    (removal,) = result.node_removals
    assert removal.delta_e == 0
    # Decided by Delta_V alone: removing the only node definition loses everything.
    assert removal.delta_v == approx(1) and not removal.redundant
    evaluated = evaluate(instance, s)
    assert (evaluated.cov_v, evaluated.cov_e, evaluated.con_v, evaluated.con_e) == (
        approx(1),
        approx(0),
        approx(1),
        approx(1),
    )
    assert evaluated.c2 == approx(Fraction(2, 3))
    assert evaluated.best_matches["edge_types"][0]["schema_type"] is None
    assert evaluated.best_matches["edge_types"][0]["source"] is None
    assert evaluated.best_matches["edge_types"][0]["target"] is None


def test_e3_empty_schema():
    instance = graph([("x", ["A"], {})], [("e", ["R"], "x", "x", {})])
    result = evaluate(instance, Schema())
    assert (result.cov_v, result.cov_e, result.con_v, result.con_e) == (0, 0, 1, 1)
    assert result.c2 == 0
    assert result.best_matches["node_types"][0]["schema_type"] is None


def test_e4_empty_instance():
    s = schema([{"id": "a", "labels": ["A"]}], [{"id": "r", "labels": ["R"], "source": "a", "target": "a"}])
    result = evaluate(Instance(), s)
    assert (result.cov_v, result.cov_e, result.con_v, result.con_e) == (1, 1, 0, 0)
    assert result.c2 == 0
    assert result.redundant_types == {"node_types": ["a"], "edge_types": ["r"]}


def test_empty_instance_and_schema():
    result = evaluate(Instance(), Schema())
    assert (result.cov_v, result.cov_e, result.con_v, result.con_e, result.c2) == (1, 1, 1, 1, 1)


def b3():
    s = schema(
        [{"id": "aType", "labels": ["A"]}, {"id": "bType", "labels": ["B"]}, {"id": "cType", "labels": ["C"]}],
        [
            {"id": "r1", "labels": ["R"], "source": "aType", "target": "bType", "properties": {"x": {}}},
            {"id": "r2", "labels": ["R"], "source": "aType", "target": "bType", "properties": {"y": {}}},
            {"id": "s1", "labels": ["S"], "source": "cType", "target": "cType"},
        ],
    )
    instance = graph(
        [("a", ["A"], {}), ("b", ["B"], {}), ("c", ["C"], {})],
        [("e1", ["R"], "a", "b", {"x": 1}), ("e2", ["S"], "c", "c", {})],
    )
    return instance, s


def test_b3_similarities():
    instance, s = b3()
    similarity = Similarity(instance, SimilarityParams())
    flat = flatten(s)
    r = next(t for t in instance.edge_types if t.labels == {"R"})
    by_origin = {e.schema_type: e for e in flat.edge_types}
    assert similarity.edge(r, by_origin["r1"]) == approx(1)
    assert similarity.edge(r, by_origin["r2"]) == approx(Fraction(13, 16))


def test_b3_conciseness():
    instance, s = b3()
    similarity = Similarity(instance, SimilarityParams())
    flat = flatten(s)
    base_v, base_e = cov_v(flat, similarity)[0], cov_e(flat, similarity)[0]
    assert base_e == approx(1)
    result = conciseness(s, flat, base_v, base_e, similarity, 0.15)
    assert result.theta_e == approx(Fraction(1, 20))
    deltas = {r.type_id: r.delta_e for r in result.edge_removals}
    assert deltas == {"r1": approx(Fraction(3, 32)), "r2": approx(0), "s1": approx(Fraction(1, 2))}
    assert result.con_e == approx(Fraction(2, 3))
    assert result.con_v == approx(1)
    assert evaluate(instance, s).redundant_types["edge_types"] == ["r2"]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"label_weight": -0.1}, "label_weight"),
        ({"label_weight": 1.1}, "label_weight"),
        ({"label_weight": math.nan}, "label_weight"),
        ({"label_weight": "0.5"}, "label_weight"),
        ({"label_weight": True}, "label_weight"),
        ({"edge_weights": (0.5, 0.5)}, "edge_weights"),
        ({"edge_weights": (0.5, 0.6, -0.1)}, r"edge_weights\[2\]"),
        ({"edge_weights": (0.5, 0.5, 0.5)}, "edge_weights must sum to 1"),
        ({"edge_weights": "abc"}, "edge_weights"),
        ({"redundancy_ratio": 0}, "redundancy_ratio"),
        ({"redundancy_ratio": -1}, "redundancy_ratio"),
        ({"redundancy_ratio": math.inf}, "redundancy_ratio"),
        ({"use_types": 1}, "use_types"),
        ({"allow_any": None}, "allow_any"),
    ],
)
def test_invalid_parameters(kwargs, name):
    with pytest.raises(ParameterError, match=name) as info:
        evaluate(fig1a(), example(1), **kwargs)
    assert repr(next(iter(kwargs.values()))) in str(info.value)


def test_numpy_parameters():
    # Numpy values are accepted, and params holds plain Python values.
    plain = evaluate(fig1a(), example(1), edge_weights=(0.5, 0.25, 0.25), use_types=True, allow_any=False)
    result = evaluate(
        fig1a(),
        example(1),
        label_weight=np.float64(0.5),
        edge_weights=np.array([0.5, 0.25, 0.25]),
        redundancy_ratio=np.float32(0.15),
        use_types=np.True_,
        allow_any=np.False_,
    )
    assert result.c2 == plain.c2
    assert result.params == {**plain.params, "redundancy_ratio": float(np.float32(0.15))}
    assert type(result.params["edge_weights"][0]) is float
    assert type(result.params["use_types"]) is bool
    assert type(result.params["allow_any"]) is bool
    json.dumps(result.to_dict())


@pytest.mark.parametrize(
    ("edge_weights", "fragment"),
    [
        (np.array([[0.5, 0.25, 0.25]]), "sequence of three numbers"),
        (np.array([0.5, 0.5]), "sequence of three numbers"),
        (np.array([0.5, 0.6, -0.1]), r"edge_weights\[2\]"),
        (np.array([True, False, False]), r"edge_weights\[0\]"),
    ],
)
def test_invalid_numpy_edge_weights(edge_weights, fragment):
    with pytest.raises(ParameterError, match=fragment):
        evaluate(fig1a(), example(1), edge_weights=edge_weights)


def test_valid_boundary_parameters():
    evaluate(fig1a(), example(1), label_weight=0, edge_weights=[1, 0, 0])
    evaluate(fig1a(), example(1), label_weight=1, edge_weights=(0.1, 0.2, 0.7), redundancy_ratio=2)


def test_wrong_argument_types():
    with pytest.raises(TypeError):
        evaluate(example(1), fig1a())  # type: ignore[arg-type]
