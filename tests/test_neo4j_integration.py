"""Integration test #3: small graphs stored in Neo4j.

Fig. 1(a) checks the values of the spec; ``mixed.json`` covers what Fig. 1(a)
lacks: edge properties, nodes without labels, values that map to ANY (lists),
DATE / LOCAL DATETIME / ZONED DATETIME, INTEGER and FLOAT in one property,
labels and an edge type containing ``:``, and self-loops.

It runs only when all four variables below are set, and writes only to an
empty database, which it empties again at the end::

    C2SCORE_TEST_NEO4J_URI, C2SCORE_TEST_NEO4J_USER,
    C2SCORE_TEST_NEO4J_PASSWORD, C2SCORE_TEST_NEO4J_DATABASE
"""

import json
import os
import re
from fractions import Fraction
from pathlib import Path

import pytest

from c2score import Instance, evaluate, load_neo4j, load_schema_json
from c2score.datatypes import DataType
from c2score.model import PropertySpec, _abs_from_graph

DATA = Path(__file__).parent / "data"
VARIABLES = ("URI", "USER", "PASSWORD", "DATABASE")
SETTINGS = {name: os.environ.get(f"C2SCORE_TEST_NEO4J_{name}") for name in VARIABLES}

needs_neo4j = pytest.mark.skipif(
    not all(SETTINGS.values()),
    reason="set C2SCORE_TEST_NEO4J_URI, _USER, _PASSWORD and _DATABASE to run",
)


def is_read_only(database):
    """A database whose name ends with '-gt' is treated as read-only, to protect reference data.

    Neo4j database names ignore case, and so does this check.
    """
    return database.lower().endswith("-gt")


def approx(value):
    return pytest.approx(float(value), abs=1e-12)


def quote(name):
    return "`" + name.replace("`", "``") + "`"


def temporal_function(text):
    """date() for a date, datetime() with a time zone, localdatetime() without one."""
    if "T" not in text:
        return "date"
    return "datetime" if re.search(r"(Z|[+-]\d\d:\d\d)$", text) else "localdatetime"


def cypher_properties(properties, prefix, parameters):
    """A Cypher map for ``properties``; ``{"$datetime": ...}`` values become Neo4j temporal values."""
    entries = []
    for i, (key, value) in enumerate(properties.items()):
        name = f"{prefix}_{i}"
        if isinstance(value, dict):
            text = value["$datetime"]
            parameters[name] = text
            entries.append(f"{quote(key)}: {temporal_function(text)}(${name})")
        else:
            parameters[name] = value
            entries.append(f"{quote(key)}: ${name}")
    return "{" + ", ".join(entries) + "}"


def create_statement(graph):
    """One CREATE statement that stores the graph in the JSON instance structure."""
    parameters = {}
    variable = {node["id"]: f"n{i}" for i, node in enumerate(graph["nodes"])}
    patterns = []
    for i, node in enumerate(graph["nodes"]):
        labels = "".join(":" + quote(label) for label in node["labels"])
        props = cypher_properties(node.get("properties", {}), f"n{i}", parameters)
        patterns.append(f"({variable[node['id']]}{labels} {props})")
    for i, edge in enumerate(graph["edges"]):
        (label,) = edge["labels"]
        props = cypher_properties(edge.get("properties", {}), f"e{i}", parameters)
        patterns.append(
            f"({variable[edge['source']]})-[:{quote(label)} {props}]->({variable[edge['target']]})"
        )
    return "CREATE " + ",\n       ".join(patterns), parameters


def read_graph(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def store_and_load(graph):
    """Store ``graph`` in the empty test database, load it, and empty the database again."""
    neo4j = pytest.importorskip("neo4j")
    database = SETTINGS["DATABASE"]
    if is_read_only(database):
        pytest.fail(f"refusing to write to {database!r}: names ending with '-gt' are treated as read-only")
    driver = neo4j.GraphDatabase.driver(SETTINGS["URI"], auth=(SETTINGS["USER"], SETTINGS["PASSWORD"]))
    try:
        with driver.session(database=database) as session:
            count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            if count:
                pytest.fail(f"database {database!r} is not empty ({count} nodes); nothing was written")
            statement, parameters = create_statement(graph)
            session.run(statement, parameters).consume()
            try:
                return load_neo4j(
                    SETTINGS["URI"], auth=(SETTINGS["USER"], SETTINGS["PASSWORD"]), database=database
                )
            finally:
                session.run("MATCH (n) DETACH DELETE n").consume()
    finally:
        driver.close()


@pytest.fixture(scope="module")
def fig1a_graph():
    return read_graph("fig1a.json")


@pytest.fixture(scope="module")
def loaded(fig1a_graph):
    return store_and_load(fig1a_graph)


@pytest.fixture(scope="module")
def mixed_graph():
    return read_graph("mixed.json")


@needs_neo4j
def test_load_matches_abs_from_graph(loaded, fig1a_graph):
    assert loaded == _abs_from_graph(fig1a_graph["nodes"], fig1a_graph["edges"])


@needs_neo4j
def test_mixed_graph_matches_abs_from_graph(mixed_graph):
    assert store_and_load(mixed_graph) == _abs_from_graph(mixed_graph["nodes"], mixed_graph["edges"])


@needs_neo4j
def test_evaluate_section_4_9(loaded):
    result = evaluate(loaded, load_schema_json(DATA / "example1.json"))
    assert result.cov_v == approx(Fraction(11, 12))
    assert result.cov_e == approx(Fraction(5, 6))
    assert result.con_v == approx(Fraction(3, 4))
    assert result.con_e == approx(Fraction(1, 3))
    assert result.c2 == approx(Fraction(91, 136))
    assert result.redundant_types == {"node_types": ["postType2"], "edge_types": ["createsType1", "createsType2"]}


@needs_neo4j
def test_round_trip_through_json(loaded, tmp_path):
    path = tmp_path / "fig1a-instance.json"
    loaded.to_json(path)
    assert Instance.from_json(path) == loaded


@pytest.mark.parametrize(("database", "read_only"), [("example-gt", True), ("EXAMPLE-GT", True), ("x-Gt", True), ("c2score-test", False), ("gt", False)])
def test_gt_databases_are_read_only(database, read_only):
    assert is_read_only(database) is read_only


def test_mixed_graph_covers_the_cases():
    """The summary of ``mixed.json`` has what the Neo4j test is meant to exercise."""
    graph = read_graph("mixed.json")
    instance = _abs_from_graph(graph["nodes"], graph["edges"])
    person = instance.node_type(["Person"])
    assert person.properties["score"] == PropertySpec(DataType.FLOAT, True)  # INTEGER and FLOAT
    assert person.properties["born"] == PropertySpec(DataType.DATETIME, True)  # DATE and ZONED DATETIME
    assert person.properties["tags"] == PropertySpec(DataType.ANY, False)  # a list
    assert instance.node_type(["Person", "ns:Admin"]).properties["code"].type is DataType.ANY  # STRING and INTEGER
    assert instance.node_type([]).properties == {"note": PropertySpec(DataType.STRING, False)}
    assert instance.node_type(["Doc"]).properties["at"].type is DataType.DATETIME  # LOCAL DATETIME and DATE
    edges = {(tuple(sorted(e.labels)), tuple(sorted(e.source_labels)), tuple(sorted(e.target_labels))): e
             for e in instance.edge_types}
    wrote = edges[(("WROTE",), ("Person",), ("Doc",))]
    assert wrote.properties["weight"] == PropertySpec(DataType.FLOAT, False)
    assert wrote.properties["parts"] == PropertySpec(DataType.ANY, False)
    assert (("has:part",), ("Doc",), ("Doc",)) in edges
    assert ((("LINKS",), (), ())) in edges
    assert (("MANAGES",), ("Person", "ns:Admin"), ()) in edges
