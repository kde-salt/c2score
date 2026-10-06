"""The fast evaluation gives exactly the results of the straightforward reference.

Random schemas and instances (fixed seeds) cover node and edge inheritance
(multiple, deep, inherited endpoints), cardinalities, data types with ANY,
unlabelled types, duplicated definitions, an instance without edges, and the
parameter combinations. Small label pools make ties common.
"""

from __future__ import annotations

import random

import pytest

from c2score import _reference, evaluate
from c2score.cardinality import ONE_MANY, ONE_ONE, ZERO_MANY, ZERO_ONE, Cardinality
from c2score.conciseness import _Shortcuts, conciseness, reduce_schema
from c2score.coverage import CoverageTables
from c2score.datatypes import DataType
from c2score.errors import SchemaError
from c2score.evaluate import _check_params
from c2score.flatten import flatten
from c2score.model import (
    EdgeTypeDef,
    Instance,
    InstanceEdgeType,
    InstanceNodeType,
    NodeTypeDef,
    PropertySpec,
    Schema,
)
from c2score.similarity import Similarity

NODE_LABELS = ["A", "B", "C"]
EDGE_LABELS = ["R", "S"]
KEYS = ["k1", "k2", "k3", "k4"]
TYPES = [DataType.STRING, DataType.INTEGER, DataType.FLOAT, DataType.ANY]
INTERVALS = [ZERO_ONE, ZERO_MANY, ONE_ONE, ONE_MANY]
SEEDS = range(500)


def _labels(rng: random.Random, pool: list[str], most: int) -> frozenset[str]:
    # Mostly one label, so that types often overlap; sometimes none.
    size = rng.choices(range(most + 1), weights=[1, 6, 2][: most + 1])[0]
    return frozenset(rng.sample(pool, size))


def _properties(rng: random.Random, any_type: bool) -> dict[str, PropertySpec]:
    types = TYPES if any_type else TYPES[:-1]
    return {
        key: PropertySpec(rng.choice(types), rng.random() < 0.6)
        for key in rng.sample(KEYS, rng.randint(0, 3))
    }


def _cardinality(rng: random.Random) -> Cardinality:
    return Cardinality(rng.choice(INTERVALS), rng.choice(INTERVALS))


def random_schema(rng: random.Random, any_type: bool = True) -> Schema:
    nodes: list[NodeTypeDef] = []
    for i in range(rng.randint(0, 7)):
        if nodes and rng.random() < 0.15:
            # A definition with the same content as an earlier one.
            twin = rng.choice(nodes)
            nodes.append(NodeTypeDef(f"n{i}", twin.labels, twin.properties, twin.supertypes))
            continue
        parents = [n.id for n in nodes]
        supertypes = tuple(rng.sample(parents, min(len(parents), rng.choice([0, 0, 1, 1, 2, 3]))))
        nodes.append(NodeTypeDef(f"n{i}", _labels(rng, NODE_LABELS, 2), _properties(rng, any_type), supertypes))
    edges: list[EdgeTypeDef] = []
    if nodes:
        node_ids = [n.id for n in nodes]
        for i in range(rng.randint(0, 8)):
            if edges and rng.random() < 0.15:
                twin = rng.choice(edges)
                edges.append(
                    EdgeTypeDef(
                        f"e{i}", twin.labels, twin.properties, twin.supertypes, twin.source, twin.target,
                        twin.cardinality,
                    )
                )
                continue
            parents = [e.id for e in edges]
            supertypes = tuple(rng.sample(parents, min(len(parents), rng.choice([0, 0, 0, 1, 1, 2]))))
            # A definition without parents writes both endpoints; a child may inherit them.
            source = rng.choice(node_ids) if not supertypes or rng.random() < 0.4 else None
            target = rng.choice(node_ids) if not supertypes or rng.random() < 0.4 else None
            edges.append(
                EdgeTypeDef(
                    f"e{i}",
                    _labels(rng, EDGE_LABELS, 2) if rng.random() < 0.9 else frozenset(),
                    _properties(rng, any_type),
                    supertypes,
                    source,
                    target,
                    _cardinality(rng) if rng.random() < 0.3 else None,
                )
            )
    return Schema(tuple(nodes), tuple(edges))


def random_instance(rng: random.Random, schema: Schema) -> Instance:
    label_sets = {_labels(rng, NODE_LABELS, 2) for _ in range(rng.randint(1, 6))}
    nodes = [InstanceNodeType(labels, _properties(rng, True)) for labels in label_sets]
    sets = sorted(label_sets, key=sorted)
    elements = flatten(schema).edge_types
    edges = {}
    if rng.random() < 0.9:
        for _ in range(rng.randint(0, 10)):
            if elements and rng.random() < 0.6:
                # Close to some element of S+, so that the similarity is often positive.
                e = rng.choice(elements)
                labels = e.labels if rng.random() < 0.7 else _labels(rng, EDGE_LABELS, 2)
                ends = []
                for end in (e.source, e.target):
                    near = [l for l in sets if l & end.labels or l == end.labels]
                    ends.append(rng.choice(near or sets))
                key = (labels, *ends)
            else:
                labels = _labels(rng, EDGE_LABELS, 2) if rng.random() < 0.9 else frozenset()
                key = (labels, rng.choice(sets), rng.choice(sets))
            edges[key] = InstanceEdgeType(*key, _cardinality(rng), _properties(rng, True))
    return Instance(tuple(nodes), tuple(edges.values()))


def random_params(rng: random.Random) -> dict[str, object]:
    params: dict[str, object] = {}
    if rng.random() < 0.5:
        params["label_weight"] = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
    if rng.random() < 0.4:
        params["edge_weights"] = rng.choice([(0.5, 0.5, 0.0), (1 / 3, 1 / 3, 1 / 3), (0.2, 0.3, 0.5), (1.0, 0.0, 0.0)])
    if rng.random() < 0.4:
        params["redundancy_ratio"] = rng.choice([1e-9, 0.15, 0.5, 2.0])
    if rng.random() < 0.4:
        params["use_types"] = False
    if rng.random() < 0.3:
        params["allow_any"] = False
    return params


def example(seed: int) -> tuple[Instance, Schema, dict[str, object]]:
    rng = random.Random(seed)
    params = random_params(rng)
    schema = random_schema(rng, any_type=params.get("allow_any", True) or rng.random() < 0.3)
    return random_instance(rng, schema), schema, params


def fast_conciseness(instance, schema, params):
    checked = _check_params(instance, schema, **params)
    similarity = Similarity(instance, checked.similarity)
    flat = flatten(schema)
    tables = CoverageTables(flat, similarity)
    return conciseness(
        schema, flat, tables.cov_v_value, tables.cov_e_value, similarity, checked.redundancy_ratio, tables=tables
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_fast_equals_reference(seed):
    instance, schema, params = example(seed)
    try:
        expected = _reference.run(instance, schema, **params)
    except SchemaError as e:
        with pytest.raises(SchemaError, match="allow_any=False"):
            evaluate(instance, schema, **params)
        assert "allow_any=False" in str(e)
        return
    # The fast path does the same arithmetic in the same order: equal to the last bit.
    assert evaluate(instance, schema, **params).to_dict() == expected.result.to_dict()
    assert flatten(schema) == expected.flat
    assert [e.origins for e in flatten(schema).edge_types] == [e.origins for e in expected.flat.edge_types]
    fast = fast_conciseness(instance, schema, params)
    assert fast.node_removals == expected.conciseness.node_removals
    assert fast.edge_removals == expected.conciseness.edge_removals


def test_both_removal_paths_are_exercised():
    incremental = reflattened = 0
    for seed in SEEDS:
        instance, schema, params = example(seed)
        try:
            con = fast_conciseness(instance, schema, params)
        except SchemaError:
            continue
        reflattened += len(con.reflattened)
        incremental += len(schema.node_types) + len(schema.edge_types) - len(con.reflattened)
    assert incremental > 1000
    assert reflattened > 300


@pytest.mark.parametrize("seed", SEEDS)
def test_incremental_removal_leaves_the_rest_of_s_plus_unchanged(seed):
    # Where the shortcut applies, S+ of the reduced schema is S+ without the
    # columns that the shortcut takes away, with every other element unchanged.
    _, schema, _ = example(seed)
    flat = flatten(schema)
    shortcuts = _Shortcuts(schema, flat)
    for node in schema.node_types:
        taken = shortcuts.node(node.id)
        if taken is None:
            continue
        column, edges = taken
        reduced = flatten(reduce_schema(schema, node.id))
        assert reduced.node_types == flat.node_types[:column] + flat.node_types[column + 1 :]
        assert set(reduced.edge_types) == {e for j, e in enumerate(flat.edge_types) if j not in edges}
        assert len(reduced.edge_types) == len(flat.edge_types) - len(edges)
    for edge in schema.edge_types:
        edges = shortcuts.edge(edge.id)
        if edges is None:
            continue
        reduced = flatten(reduce_schema(schema, edge.id))
        assert reduced.node_types == flat.node_types
        assert set(reduced.edge_types) == {e for j, e in enumerate(flat.edge_types) if j not in edges}
        assert len(reduced.edge_types) == len(flat.edge_types) - len(edges)
