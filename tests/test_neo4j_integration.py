"""Integration test #3: Fig. 1(a) stored in Neo4j.

It runs only when all four variables below are set, and writes only to an
empty database, which it empties again at the end::

    C2SCORE_TEST_NEO4J_URI, C2SCORE_TEST_NEO4J_USER,
    C2SCORE_TEST_NEO4J_PASSWORD, C2SCORE_TEST_NEO4J_DATABASE
"""

import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from c2score import Instance, evaluate, load_neo4j, load_schema_json
from c2score.model import _abs_from_graph

DATA = Path(__file__).parent / "data"
VARIABLES = ("URI", "USER", "PASSWORD", "DATABASE")
SETTINGS = {name: os.environ.get(f"C2SCORE_TEST_NEO4J_{name}") for name in VARIABLES}

needs_neo4j = pytest.mark.skipif(
    not all(SETTINGS.values()),
    reason="set C2SCORE_TEST_NEO4J_URI, _USER, _PASSWORD and _DATABASE to run",
)


def is_read_only(database):
    """'-gt' databases must never be written; Neo4j database names ignore case."""
    return database.lower().endswith("-gt")


def approx(value):
    return pytest.approx(float(value), abs=1e-12)


def quote(name):
    return "`" + name.replace("`", "``") + "`"


def cypher_properties(properties, prefix, parameters):
    """A Cypher map for ``properties``; ``{"$datetime": ...}`` values go through date() or datetime()."""
    entries = []
    for i, (key, value) in enumerate(properties.items()):
        name = f"{prefix}_{i}"
        if isinstance(value, dict):
            text = value["$datetime"]
            parameters[name] = text
            entries.append(f"{quote(key)}: {'datetime' if 'T' in text else 'date'}(${name})")
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


@pytest.fixture(scope="module")
def fig1a_graph():
    return json.loads((DATA / "fig1a.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stored(fig1a_graph):
    """Store Fig. 1(a) in the empty test database, and empty it afterwards."""
    neo4j = pytest.importorskip("neo4j")
    database = SETTINGS["DATABASE"]
    if is_read_only(database):
        pytest.fail(f"refusing to write to {database!r}: '-gt' databases are read-only")
    driver = neo4j.GraphDatabase.driver(SETTINGS["URI"], auth=(SETTINGS["USER"], SETTINGS["PASSWORD"]))
    try:
        with driver.session(database=database) as session:
            count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            if count:
                pytest.fail(f"database {database!r} is not empty ({count} nodes); nothing was written")
            statement, parameters = create_statement(fig1a_graph)
            session.run(statement, parameters).consume()
            try:
                yield
            finally:
                session.run("MATCH (n) DETACH DELETE n").consume()
    finally:
        driver.close()


@pytest.fixture(scope="module")
def loaded(stored):
    return load_neo4j(
        SETTINGS["URI"], auth=(SETTINGS["USER"], SETTINGS["PASSWORD"]), database=SETTINGS["DATABASE"]
    )


@needs_neo4j
def test_load_matches_abs_from_graph(loaded, fig1a_graph):
    assert loaded == _abs_from_graph(fig1a_graph["nodes"], fig1a_graph["edges"])


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


@pytest.mark.parametrize(("database", "read_only"), [("ldbc-gt", True), ("LDBC-GT", True), ("x-Gt", True), ("c2score-test", False), ("gt", False)])
def test_gt_databases_are_read_only(database, read_only):
    assert is_read_only(database) is read_only
