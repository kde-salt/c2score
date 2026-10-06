"""Endpoint expansion of an edge type whose source and target are the same node type."""

from c2score import load_schema_json
from c2score.cardinality import ONE_ONE, ZERO_MANY, Cardinality
from c2score.flatten import flatten


def ends(flat):
    return [(e.schema_type, e.source.id, e.target.id) for e in flat.edge_types]


def test_self_loop_expands_to_every_pair_of_descendants():
    # A is the parent of B. The edge A -> A yields (A, A), (A, B), (B, A), and (B, B).
    schema = load_schema_json(
        {
            "node_types": [
                {"id": "aType", "labels": ["A"]},
                {"id": "bType", "labels": ["B"], "supertypes": ["aType"]},
            ],
            "edge_types": [
                {
                    "id": "rType",
                    "labels": ["R"],
                    "source": "aType",
                    "target": "aType",
                    "properties": {"w": {"type": "INTEGER"}},
                    "cardinality": {"out": "1..1", "in": "0..*"},
                }
            ],
        }
    )
    flat = flatten(schema)
    assert ends(flat) == [
        ("rType", "aType", "aType"),
        ("rType", "aType", "bType"),
        ("rType", "bType", "aType"),
        ("rType", "bType", "bType"),
    ]
    for edge in flat.edge_types:
        assert edge.labels == {"R"}
        assert set(edge.properties) == {"w"}
        assert edge.cardinality == Cardinality(ONE_ONE, ZERO_MANY)
    # The endpoints are the flattened node types, so B carries A's label.
    b_to_b = flat.edge_types[3]
    assert b_to_b.source.labels == {"A", "B"}
    assert b_to_b.target.labels == {"A", "B"}


def test_self_loop_on_a_grandparent_expands_to_all_nine_pairs():
    schema = load_schema_json(
        {
            "node_types": [
                {"id": "aType", "labels": ["A"]},
                {"id": "bType", "labels": ["B"], "supertypes": ["aType"]},
                {"id": "cType", "labels": ["C"], "supertypes": ["bType"]},
            ],
            "edge_types": [{"id": "rType", "labels": ["R"], "source": "aType", "target": "aType"}],
        }
    )
    ids = ["aType", "bType", "cType"]
    assert ends(flatten(schema)) == [("rType", s, t) for s in ids for t in ids]


def test_self_loop_on_a_leaf_is_not_expanded():
    schema = load_schema_json(
        {
            "node_types": [
                {"id": "aType", "labels": ["A"]},
                {"id": "bType", "labels": ["B"], "supertypes": ["aType"]},
            ],
            "edge_types": [{"id": "rType", "labels": ["R"], "source": "bType", "target": "bType"}],
        }
    )
    assert ends(flatten(schema)) == [("rType", "bType", "bType")]
