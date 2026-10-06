"""Unit tests of load_neo4j that run without a Neo4j server.

A fake driver answers the loader's queries from a small in-memory graph,
following the semantics of each Cypher query.
"""

import json
from collections import defaultdict
from pathlib import Path

import neo4j
import pytest

from c2score import InstanceLoadError, ParameterError, load_neo4j
from c2score.cardinality import ONE_MANY, ONE_ONE, ZERO_ONE, Cardinality
from c2score.datatypes import DataType
from c2score.loaders import neo4j as loader
from c2score.model import _abs_from_graph

DATA = Path(__file__).parent / "data"


def neo4j_type(value):
    """The valueType() string Neo4j reports for a stored property value."""
    if isinstance(value, dict):
        return "DATE NOT NULL"
    if isinstance(value, bool):
        return "BOOLEAN NOT NULL"
    if isinstance(value, int):
        return "INTEGER NOT NULL"
    if isinstance(value, float):
        return "FLOAT NOT NULL"
    if isinstance(value, str):
        return "STRING NOT NULL"
    return "LIST<INTEGER NOT NULL> NOT NULL"


def answer(nodes, edges):
    """Rows of every loader query over a graph in the JSON instance structure.

    The label lists keep each node's own order, as Neo4j may return the same
    set in different orders.
    """
    labels = {n["id"]: tuple(n["labels"]) for n in nodes}
    rows = defaultdict(list)

    def grouped(keyed):
        counts = defaultdict(int)
        for key in keyed:
            counts[key] += 1
        return counts

    for ls, c in grouped(labels[n["id"]] for n in nodes).items():
        rows[loader._NODE_COUNTS].append({"ls": list(ls), "c": c})
    for (ls, k, t), c in grouped(
        (labels[n["id"]], k, neo4j_type(v)) for n in nodes for k, v in n["properties"].items()
    ).items():
        rows[loader._NODE_PROPERTIES].append({"ls": list(ls), "k": k, "t": t, "c": c})
    for (sl, rt, tl), c in grouped((labels[e["source"]], e["labels"][0], labels[e["target"]]) for e in edges).items():
        rows[loader._EDGE_COUNTS].append({"sl": list(sl), "rt": rt, "tl": list(tl), "c": c})
    for (sl, rt, tl, k, t), c in grouped(
        (labels[e["source"]], e["labels"][0], labels[e["target"]], k, neo4j_type(v))
        for e in edges
        for k, v in e["properties"].items()
    ).items():
        rows[loader._EDGE_PROPERTIES].append({"sl": list(sl), "rt": rt, "tl": list(tl), "k": k, "t": t, "c": c})
    for query, near, far in ((loader._OUT_DEGREES, "source", "target"), (loader._IN_DEGREES, "target", "source")):
        degree = grouped((e[near], e["labels"][0], tuple(sorted(labels[e[far]]))) for e in edges)
        bounds = {}
        for (v, rt, other), d in degree.items():
            key = (tuple(sorted(labels[v])), rt, other)
            low, high, n = bounds.get(key, (d, d, 0))
            bounds[key] = (min(low, d), max(high, d), n + 1)
        for (ls, rt, other), (low, high, n) in bounds.items():
            rows[query].append({"ls": list(ls), "rt": rt, "other": list(other), "dmin": low, "dmax": high, "n": n})
    return rows


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return dict(self._data)


class FakeTransaction:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query):
        self.driver.queries.append(query)
        if self.driver.error is not None:
            raise self.driver.error
        return [FakeRecord(row) for row in self.driver.rows.get(query, [])]


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def begin_transaction(self):
        return FakeTransaction(self.driver)


class FakeDriver:
    def __init__(self, nodes=(), edges=(), version="5.26.0", error=None):
        self.rows = answer(list(nodes), list(edges))
        self.rows[loader._VERSION_QUERY] = [{"name": "Neo4j Kernel", "versions": [version]}]
        self.error = error
        self.closed = False
        self.session_kwargs = []
        self.queries = []

    def session(self, **kwargs):
        self.session_kwargs.append(kwargs)
        return FakeSession(self)

    def close(self):
        self.closed = True


def fig1a_graph():
    return json.loads((DATA / "fig1a.json").read_text(encoding="utf-8"))


def node(node_id, labels, **properties):
    return {"id": node_id, "labels": list(labels), "properties": properties}


def edge(edge_id, label, source, target, **properties):
    return {"id": edge_id, "labels": [label], "source": source, "target": target, "properties": properties}


# ---------------------------------------------------------------------------
# Arguments and connection
# ---------------------------------------------------------------------------


def test_needs_exactly_one_of_uri_and_driver():
    with pytest.raises(ParameterError, match="exactly one of uri and driver"):
        load_neo4j()
    with pytest.raises(ParameterError, match="exactly one of uri and driver"):
        load_neo4j("bolt://localhost:7687", driver=FakeDriver())


def test_auth_only_with_uri():
    with pytest.raises(ParameterError, match="auth"):
        load_neo4j(driver=FakeDriver(), auth=("neo4j", "password"))


def test_session_is_read_only_and_uses_the_database():
    driver = FakeDriver()
    load_neo4j(driver=driver, database="graphs")
    assert driver.session_kwargs == [{"database": "graphs", "default_access_mode": neo4j.READ_ACCESS}]
    load_neo4j(driver=driver)
    assert driver.session_kwargs[1]["database"] is None


def test_given_driver_is_left_open():
    driver = FakeDriver()
    load_neo4j(driver=driver)
    assert not driver.closed


@pytest.mark.parametrize("version", ["5.25.0", "4.4.30"])
def test_driver_from_uri_is_closed(monkeypatch, version):
    created = []

    def fake_driver(uri, auth=None):
        created.append(FakeDriver(version=version))
        return created[-1]

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", fake_driver)
    try:
        load_neo4j("bolt://example:7687", auth=("u", "p"))
    except InstanceLoadError:
        pass  # 4.4 is too old; the driver must be closed all the same.
    assert created[0].closed


@pytest.mark.parametrize("version", ["5.12.0", "4.4.30", "5.9.0-aura"])
def test_old_version(version):
    with pytest.raises(InstanceLoadError, match=f"Neo4j {version} is not supported"):
        load_neo4j(driver=FakeDriver(version=version))


@pytest.mark.parametrize("version", ["5.13.0", "5.26.0", "2025.01.0", "2026.08.1"])
def test_supported_version(version):
    assert load_neo4j(driver=FakeDriver(version=version)) == _abs_from_graph([], [])


def test_unreadable_version():
    with pytest.raises(InstanceLoadError, match="cannot read the Neo4j version"):
        load_neo4j(driver=FakeDriver(version="dev"))
    driver = FakeDriver()
    driver.rows[loader._VERSION_QUERY] = [{"name": "Some Plugin", "versions": ["1.0"]}]
    with pytest.raises(InstanceLoadError, match="cannot find the Neo4j version"):
        load_neo4j(driver=driver)


def test_version_is_checked_before_reading():
    driver = FakeDriver(version="5.12.0")
    with pytest.raises(InstanceLoadError):
        load_neo4j(driver=driver)
    assert driver.queries == [loader._VERSION_QUERY]


def test_query_failure_is_wrapped():
    error = neo4j.exceptions.ClientError("boom")
    with pytest.raises(InstanceLoadError, match="cannot read database 'graphs'") as info:
        load_neo4j(driver=FakeDriver(error=error), database="graphs")
    assert info.value.__cause__ is error


def test_unreachable_server():
    # Port 1 refuses connections; there must be no retry loop.
    with pytest.raises(InstanceLoadError) as info:
        load_neo4j("bolt://127.0.0.1:1", auth=("neo4j", "password"))
    assert isinstance(info.value.__cause__, neo4j.exceptions.ServiceUnavailable)


def test_invalid_uri():
    with pytest.raises(InstanceLoadError, match="cannot create a Neo4j driver") as info:
        load_neo4j("http://localhost:7474", auth=("neo4j", "password"))
    assert isinstance(info.value.__cause__, neo4j.exceptions.ConfigurationError)


# ---------------------------------------------------------------------------
# Building Abs(G) from the aggregated rows
# ---------------------------------------------------------------------------


def test_fig1a_matches_abs_from_graph():
    graph = fig1a_graph()
    expected = _abs_from_graph(graph["nodes"], graph["edges"])
    assert load_neo4j(driver=FakeDriver(graph["nodes"], graph["edges"])) == expected


def test_label_order_does_not_split_a_type():
    # Neo4j may return one label set in two orders: the rows must be added up.
    nodes = [
        node("a", ["USER", "OFFICIAL"], id=1),
        node("b", ["OFFICIAL", "USER"], id=2, bio="x"),
        node("p", ["POST"]),
    ]
    edges = [edge("e1", "LIKES", "a", "p"), edge("e2", "LIKES", "b", "p")]
    driver = FakeDriver(nodes, edges)
    assert len(driver.rows[loader._NODE_COUNTS]) == 3
    instance = load_neo4j(driver=driver)
    assert instance == _abs_from_graph(nodes, edges)
    official = instance.node_type({"USER", "OFFICIAL"})
    assert official.properties["id"].required
    assert not official.properties["bio"].required
    (likes,) = instance.edge_types
    assert likes.cardinality == Cardinality(ONE_ONE, ONE_MANY)


def test_degree_rows_are_merged():
    rows = [
        {"ls": ["A", "B"], "rt": "R", "other": ["C"], "dmin": 2, "dmax": 3, "n": 2},
        {"ls": ["B", "A"], "rt": "R", "other": ["C"], "dmin": 1, "dmax": 2, "n": 1},
    ]
    key = (frozenset({"R"}), frozenset({"A", "B"}), frozenset({"C"}))
    assert loader._degree_bounds(rows, outgoing=True) == {key: (1, 3, 3)}
    reverse = (frozenset({"R"}), frozenset({"C"}), frozenset({"A", "B"}))
    assert loader._degree_bounds(rows, outgoing=False) == {reverse: (1, 3, 3)}


def test_nodes_without_edges_count_as_zero():
    nodes = [node("a1", ["A"]), node("a2", ["A"]), node("b", ["B"])]
    edges = [edge("e", "R", "a1", "b")]
    instance = load_neo4j(driver=FakeDriver(nodes, edges))
    assert instance == _abs_from_graph(nodes, edges)
    assert instance.edge_types[0].cardinality == Cardinality(ZERO_ONE, ONE_ONE)


def test_types_unlabelled_nodes_and_isolated_label_sets():
    nodes = [
        node("x", [], n=1, tags=[1, 2]),
        node("y", [], n=2.5, flag=True),
        node("z", ["Lonely"], when={"$datetime": "2025-01-01"}),
    ]
    edges = [edge("e", "SELF", "x", "y", w=1), edge("f", "SELF", "y", "y", w=0.5)]
    instance = load_neo4j(driver=FakeDriver(nodes, edges))
    assert instance == _abs_from_graph(nodes, edges)
    blank = instance.node_type(set())
    assert blank.properties["n"].type == DataType.FLOAT
    assert blank.properties["tags"].type == DataType.ANY
    assert not blank.properties["flag"].required
    assert instance.node_type({"Lonely"}).properties["when"].type == DataType.DATETIME
    assert instance.edge_types[0].properties["w"].type == DataType.FLOAT


def test_empty_database():
    assert load_neo4j(driver=FakeDriver()) == _abs_from_graph([], [])


def test_inconsistent_rows():
    graph = fig1a_graph()
    driver = FakeDriver(graph["nodes"], graph["edges"])
    driver.rows[loader._OUT_DEGREES] = []
    with pytest.raises(InstanceLoadError, match="inconsistent results"):
        load_neo4j(driver=driver)
