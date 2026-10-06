import pickle

import pytest

from c2score.cardinality import ONE_ONE, ZERO_MANY, ZERO_ONE, Cardinality
from c2score.datatypes import DataType
from c2score.errors import SchemaError
from c2score.model import (
    EdgeTypeDef,
    FrozenDict,
    Instance,
    InstanceEdgeType,
    InstanceNodeType,
    NodeTypeDef,
    PropertySpec,
    Schema,
    precedence,
)


def test_precedence_bfs_in_written_order():
    parents = {
        "manager": ["employee", "shareholder"],
        "employee": ["person"],
        "shareholder": ["person"],
        "person": [],
    }
    assert precedence("manager", parents.__getitem__) == ("manager", "employee", "shareholder", "person")
    assert precedence("person", parents.__getitem__) == ("person",)


def test_precedence_is_breadth_first():
    parents = {"a": ["b", "c"], "b": ["d"], "c": ["e"], "d": [], "e": []}
    assert precedence("a", parents.__getitem__) == ("a", "b", "c", "d", "e")


def test_property_spec_defaults():
    assert PropertySpec() == PropertySpec(DataType.ANY, True)
    assert hash(PropertySpec()) == hash(PropertySpec(DataType.ANY, True))


def test_definitions_are_immutable():
    node = NodeTypeDef("a", labels=["A"], properties={"x": PropertySpec()}, supertypes=["b"])
    assert node.labels == frozenset({"A"})
    assert node.supertypes == ("b",)
    with pytest.raises(TypeError):
        node.properties["y"] = PropertySpec()


def test_schema_lookup_and_duplicate_ids():
    schema = Schema(
        node_types=[NodeTypeDef("a")],
        edge_types=[EdgeTypeDef("r", source="a", target="a")],
    )
    assert schema["a"].id == "a"
    assert "r" in schema and "x" not in schema
    assert not schema.has_cardinality
    with pytest.raises(SchemaError, match="duplicate id 'a'"):
        Schema(node_types=[NodeTypeDef("a")], edge_types=[EdgeTypeDef("a")])


def test_instance_sorts_types_by_label_sets():
    instance = Instance(
        node_types=[
            InstanceNodeType(frozenset({"USER", "OFFICIAL"})),
            InstanceNodeType(frozenset({"POST"})),
            InstanceNodeType(frozenset()),
            InstanceNodeType(frozenset({"USER"})),
        ],
        edge_types=[
            InstanceEdgeType({"LIKES"}, {"USER"}, {"POST"}, Cardinality(ZERO_ONE, ZERO_ONE)),
            InstanceEdgeType({"CREATES"}, {"USER"}, {"POST"}, Cardinality(ONE_ONE, ZERO_MANY)),
        ],
    )
    assert [t.key for t in instance.node_types] == [(), ("OFFICIAL", "USER"), ("POST",), ("USER",)]
    assert [t.key[0] for t in instance.edge_types] == [("CREATES",), ("LIKES",)]


def test_instance_label_with_colon_is_kept():
    node = InstanceNodeType(frozenset({"a:b", "c"}))
    assert node.labels == frozenset({"a:b", "c"})


def test_instance_rejects_duplicate_types():
    with pytest.raises(ValueError, match="duplicate"):
        Instance(node_types=[InstanceNodeType({"A"}), InstanceNodeType({"A"})])


def test_frozen_dict():
    frozen = FrozenDict({"a": 1, "b": 2})
    assert frozen == {"b": 2, "a": 1}
    assert {"a": 1, "b": 2} == frozen
    assert frozen != {"a": 1}
    assert hash(frozen) == hash(FrozenDict([("b", 2), ("a", 1)]))
    assert list(frozen) == ["a", "b"]
    with pytest.raises(TypeError):
        frozen["c"] = 3
    with pytest.raises(AttributeError):
        frozen.x = 1


def test_frozen_dict_equality_with_unhashable_values():
    assert FrozenDict({"a": [1]}) == FrozenDict({"a": [1]})
    assert FrozenDict({"a": [1]}) != FrozenDict({"a": [2]})
    assert FrozenDict({"a": [1]}) == {"a": [1]}
    # Equal contents with cached hashes, and different contents with cached hashes.
    x, y, z = FrozenDict({"a": 1}), FrozenDict({"a": 1}), FrozenDict({"a": 2})
    hash(x), hash(y), hash(z)
    assert x == y and x != z


def test_definitions_are_hashable():
    a = NodeTypeDef("a", labels={"A"}, properties={"x": PropertySpec(DataType.STRING)})
    b = NodeTypeDef("a", labels={"A"}, properties={"x": PropertySpec(DataType.STRING)})
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    edge = InstanceEdgeType({"R"}, {"A"}, {"B"}, Cardinality(ZERO_ONE, ZERO_ONE), {"x": PropertySpec()})
    assert {edge: 1}[edge] == 1


def test_schema_and_instance_pickle():
    schema = Schema(
        node_types=[NodeTypeDef("a", {"A"}, {"x": PropertySpec(DataType.STRING)}), NodeTypeDef("b", supertypes=["a"])],
        edge_types=[EdgeTypeDef("r", {"R"}, source="b", target="a", cardinality=Cardinality(ONE_ONE, ZERO_MANY))],
    )
    restored = pickle.loads(pickle.dumps(schema))
    assert restored == schema
    assert restored.precedence("b") == ("b", "a")

    instance = Instance(
        node_types=[InstanceNodeType({"A"}, {"x": PropertySpec(DataType.INTEGER, False)})],
        edge_types=[InstanceEdgeType({"R"}, {"A"}, {"A"}, Cardinality(ZERO_ONE, ZERO_MANY))],
    )
    assert pickle.loads(pickle.dumps(instance)) == instance


def test_instance_requires_endpoint_node_types():
    with pytest.raises(ValueError, match="no node type with the target labels"):
        Instance(
            node_types=[InstanceNodeType({"A"})],
            edge_types=[InstanceEdgeType({"R"}, {"A"}, {"B"}, Cardinality(ZERO_ONE, ZERO_ONE))],
        )


def test_instance_node_type_lookup():
    instance = Instance(node_types=[InstanceNodeType(frozenset()), InstanceNodeType({"A", "B"})])
    assert instance.node_type(["B", "A"]).labels == frozenset({"A", "B"})
    assert instance.node_type([]).labels == frozenset()
    with pytest.raises(KeyError):
        instance.node_type({"A"})
