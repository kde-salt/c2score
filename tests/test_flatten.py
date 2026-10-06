from pathlib import Path

import pytest

from c2score import load_schema_json
from c2score.cardinality import ONE_ONE, ZERO_MANY, Cardinality
from c2score.datatypes import DataType
from c2score.errors import SchemaError
from c2score.flatten import flatten
from c2score.model import EdgeTypeDef, NodeTypeDef, PropertySpec, Schema

DATA = Path(__file__).parent / "data"
D = DataType


def schema_of(node_types, edge_types=()):
    return load_schema_json({"node_types": list(node_types), "edge_types": list(edge_types)})


def ends(flat):
    return [(e.schema_type, e.source.id, e.target.id) for e in flat.edge_types]


def test_example1_sizes():
    flat = flatten(load_schema_json(DATA / "example1.json"))
    assert len(flat.node_types) == 4
    assert len(flat.edge_types) == 6
    assert ends(flat) == [
        ("createsType1", "officialType", "postType1"),
        ("createsType1", "userType", "postType1"),
        ("createsType2", "officialType", "postType2"),
        ("createsType2", "userType", "postType2"),
        ("likesType", "officialType", "postType1"),
        ("likesType", "userType", "postType1"),
    ]
    official = next(n for n in flat.node_types if n.id == "officialType")
    assert official.labels == {"USER", "OFFICIAL"}
    assert official.properties == {"verifiedAt": PropertySpec(D.DATETIME), "id": PropertySpec(D.INTEGER)}


def test_example3_nodes():
    flat = flatten(load_schema_json(DATA / "example3.json"))
    # Node types are in id order, not in written order.
    assert [n.id for n in flat.node_types] == [
        "companyType",
        "employeeType",
        "managerType",
        "personType",
        "shareholderType",
    ]
    manager = next(n for n in flat.node_types if n.id == "managerType")
    assert manager.labels == {"Manager", "Employee", "Shareholder", "Person"}
    assert manager.properties == {
        # employeeType comes before shareholderType in managerType.supertypes.
        "email": PropertySpec(D.STRING, True),
        "name": PropertySpec(D.STRING, True),
        "salary": PropertySpec(D.FLOAT, True),
        "shares": PropertySpec(D.INTEGER, True),
    }


def test_example3_edges():
    flat = flatten(load_schema_json(DATA / "example3.json"))
    assert ends(flat) == [
        ("managesType", "managerType", "companyType"),
        ("worksAtType", "employeeType", "companyType"),
        ("worksAtType", "managerType", "companyType"),
    ]
    manages = flat.edge_types[0]
    assert manages.labels == {"MANAGES", "WORKS_AT"}
    assert manages.properties == {"since": PropertySpec(D.DATETIME, False)}
    assert manages.cardinality == Cardinality(ONE_ONE, ZERO_MANY)
    assert flat.edge_types[1].properties == {"since": PropertySpec(D.DATETIME, True)}


def test_nearer_definition_wins_over_written_order():
    # prec(c) = [c, a, b, d]: b (one step) wins over d (two steps), although a is written first.
    schema = schema_of(
        [
            {"id": "d", "properties": {"k": {"type": "BOOLEAN"}}},
            {"id": "a", "supertypes": ["d"]},
            {"id": "b", "properties": {"k": {"type": "STRING", "required": False}}},
            {"id": "c", "supertypes": ["a", "b"]},
        ]
    )
    flat = {n.id: n for n in flatten(schema).node_types}
    assert flat["c"].properties["k"] == PropertySpec(D.STRING, False)
    assert flat["a"].properties["k"] == PropertySpec(D.BOOLEAN, True)


def test_written_order_breaks_ties_at_the_same_distance():
    schema = schema_of(
        [
            {"id": "a", "properties": {"k": {"type": "INTEGER"}}},
            {"id": "b", "properties": {"k": {"type": "STRING", "required": False}}},
            {"id": "ab", "supertypes": ["a", "b"]},
            {"id": "ba", "supertypes": ["b", "a"]},
            {"id": "own", "supertypes": ["a"], "properties": {"k": {"type": "FLOAT"}}},
        ]
    )
    flat = {n.id: n for n in flatten(schema).node_types}
    assert flat["ab"].properties["k"] == PropertySpec(D.INTEGER, True)
    assert flat["ba"].properties["k"] == PropertySpec(D.STRING, False)
    assert flat["own"].properties["k"] == PropertySpec(D.FLOAT, True)


def test_edge_inherits_endpoints_and_cardinality_separately():
    schema = schema_of(
        [{"id": "a", "labels": ["A"]}, {"id": "b", "labels": ["B"]}, {"id": "c", "labels": ["C"]}],
        [
            {"id": "p", "labels": ["P"], "source": "a", "target": "b", "cardinality": {"out": "1..1", "in": "0..*"}},
            {"id": "q", "labels": ["Q"], "source": "a", "target": "b"},
            {"id": "r", "labels": ["R"], "supertypes": ["q", "p"], "target": "c"},
        ],
    )
    flat = flatten(schema)
    r = next(e for e in flat.edge_types if e.schema_type == "r")
    assert (r.source.id, r.target.id) == ("a", "c")
    assert r.labels == {"P", "Q", "R"}
    assert r.cardinality == Cardinality(ONE_ONE, ZERO_MANY)


def test_missing_cardinality_stays_none():
    flat = flatten(load_schema_json(DATA / "example2.json"))
    assert all(e.cardinality is None for e in flat.edge_types)


def test_cartesian_product_includes_child_to_child():
    schema = schema_of(
        [
            {"id": "a", "labels": ["A"]},
            {"id": "a2", "labels": ["A2"], "supertypes": ["a"]},
            {"id": "a3", "labels": ["A3"], "supertypes": ["a2"]},
            {"id": "b", "labels": ["B"]},
            {"id": "b2", "labels": ["B2"], "supertypes": ["b"]},
        ],
        [{"id": "r", "labels": ["R"], "source": "a", "target": "b"}],
    )
    assert ends(flatten(schema)) == [
        ("r", "a", "b"),
        ("r", "a", "b2"),
        ("r", "a2", "b"),
        ("r", "a2", "b2"),
        ("r", "a3", "b"),
        ("r", "a3", "b2"),
    ]


def test_equal_elements_are_merged_with_all_origins():
    nodes = [{"id": "a", "labels": ["A"]}, {"id": "b", "labels": ["B"]}]
    schema = schema_of(
        nodes,
        [
            {"id": "r2", "labels": ["R"], "source": "a", "target": "b"},
            {"id": "r1", "labels": ["R"], "source": "a", "target": "b"},
            {"id": "r3", "labels": ["R"], "source": "a", "target": "b", "cardinality": {"out": "0..*", "in": "0..*"}},
        ],
    )
    flat = flatten(schema)
    # A written 0..* / 0..* is a different key from no cardinality.
    assert [(e.schema_type, sorted(e.origins)) for e in flat.edge_types] == [("r1", ["r1", "r2"]), ("r3", ["r3"])]


def test_parent_edge_type_is_expanded_too():
    flat = flatten(load_schema_json(DATA / "example3.json"))
    assert any(e.schema_type == "worksAtType" and e.source.id == "employeeType" for e in flat.edge_types)


def test_unvalidated_schema_with_undetermined_endpoint():
    schema = Schema(node_types=[NodeTypeDef("a")], edge_types=[EdgeTypeDef("r", source="a")])
    with pytest.raises(SchemaError, match="target is not determined"):
        flatten(schema)


def test_unvalidated_schema_with_unknown_supertype():
    schema = Schema(node_types=[NodeTypeDef("a", supertypes=["zzz"])])
    with pytest.raises(SchemaError, match="unknown supertype id 'zzz'"):
        flatten(schema)


def test_empty_schema():
    flat = flatten(Schema())
    assert flat.node_types == () and flat.edge_types == ()
