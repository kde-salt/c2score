"""Build Abs(G) from a Neo4j database (Section 3.3).

Every aggregation runs on the server; only per-group counts are read into
Python, never the graph itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Optional

import neo4j
from neo4j.exceptions import DriverError, Neo4jError

from ..cardinality import Cardinality, Interval, round_observed
from ..datatypes import DataType, from_neo4j_value_type, unify
from ..errors import InstanceLoadError, ParameterError
from ..model import Instance, InstanceEdgeType, InstanceNodeType, PropertySpec

#: The oldest supported Neo4j version: ``valueType()`` appeared in 5.13.
MIN_VERSION = (5, 13)

# A label set is sorted inside Cypher so that the same set always yields the
# same list. Python still merges rows by frozenset, adding counts.
def _sorted_labels(expression: str) -> str:
    return "COLLECT { UNWIND " + expression + " AS l RETURN l ORDER BY l }"


_VERSION_QUERY = "CALL dbms.components() YIELD name, versions RETURN name, versions"

# (1) Nodes per label set.
_NODE_COUNTS = "MATCH (n) RETURN labels(n) AS ls, count(*) AS c"

# (2) Label set x property key x value type.
_NODE_PROPERTIES = (
    "MATCH (n) UNWIND keys(n) AS k "
    "RETURN labels(n) AS ls, k, valueType(n[k]) AS t, count(*) AS c"
)

# (3) Edges per (source labels, edge type, target labels).
_EDGE_COUNTS = (
    "MATCH (s)-[r]->(t) "
    "RETURN labels(s) AS sl, type(r) AS rt, labels(t) AS tl, count(*) AS c"
)

# (4) The groups of (3) x property key x value type.
_EDGE_PROPERTIES = (
    "MATCH (s)-[r]->(t) UNWIND keys(r) AS k "
    "RETURN labels(s) AS sl, type(r) AS rt, labels(t) AS tl, k, valueType(r[k]) AS t, count(*) AS c"
)

# (5) Degree bounds per group, over the nodes with at least one such edge.
# The degree of a node v is counted per (edge type, label set of the other
# end); the other end's labels are sorted once per group, not per edge.
def _degree_query(pattern: str) -> str:
    return (
        f"MATCH {pattern} "
        "WITH v, type(r) AS rt, labels(w) AS raw, count(r) AS c "
        f"WITH v, rt, {_sorted_labels('raw')} AS other, sum(c) AS d "
        f"RETURN {_sorted_labels('labels(v)')} AS ls, rt, other, "
        "min(d) AS dmin, max(d) AS dmax, count(v) AS n"
    )


_OUT_DEGREES = _degree_query("(v)-[r]->(w)")
_IN_DEGREES = _degree_query("(v)<-[r]-(w)")

Labels = frozenset[str]
EdgeKey = tuple[Labels, Labels, Labels]  # (edge labels, source labels, target labels)


def load_neo4j(
    uri: Optional[str] = None,
    *,
    auth: Optional[tuple[str, str]] = None,
    database: Optional[str] = None,
    driver: Optional[neo4j.Driver] = None,
) -> Instance:
    """Read Abs(G) from a Neo4j database (5.13 or later).

    Give either ``uri`` (with ``auth``) or an existing ``driver``, not both.
    A driver created from ``uri`` is closed before returning; a given
    ``driver`` is left open. ``database`` selects the database; the server's
    default is used if it is ``None``. The session is read-only.

    Connection, authentication and query failures, and a server older than
    5.13, raise :class:`~c2score.InstanceLoadError`.
    """
    if (uri is None) == (driver is None):
        raise ParameterError("give exactly one of uri and driver")
    if driver is not None and auth is not None:
        raise ParameterError("auth is used only with uri; a given driver carries its own")
    if driver is None:
        try:
            driver = neo4j.GraphDatabase.driver(uri, auth=auth)
        except (DriverError, Neo4jError, ValueError) as e:
            raise InstanceLoadError(f"cannot create a Neo4j driver for {uri!r}: {e}") from e
        try:
            return _load(driver, database)
        finally:
            driver.close()
    return _load(driver, database)


def _load(driver: neo4j.Driver, database: Optional[str]) -> Instance:
    where = f"database {database!r}" if database is not None else "the default database"
    try:
        with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
            # One explicit transaction: no automatic retries, so an unreachable
            # server fails at once. Neo4j isolates transactions at read committed,
            # so the queries do not see one snapshot: if the graph is written
            # meanwhile, the counts may disagree (InstanceLoadError below) or the
            # summary may mix the states before and after the write.
            with session.begin_transaction() as tx:
                _check_version(_rows(tx, _VERSION_QUERY))
                rows = {
                    name: _rows(tx, query)
                    for name, query in (
                        ("node_counts", _NODE_COUNTS),
                        ("node_properties", _NODE_PROPERTIES),
                        ("edge_counts", _EDGE_COUNTS),
                        ("edge_properties", _EDGE_PROPERTIES),
                        ("out_degrees", _OUT_DEGREES),
                        ("in_degrees", _IN_DEGREES),
                    )
                }
    except (DriverError, Neo4jError) as e:
        raise InstanceLoadError(f"cannot read {where} from Neo4j: {e}") from e
    try:
        return _build(**rows)
    except ValueError as e:
        # Inconsistent counts: the graph changed while it was being read.
        raise InstanceLoadError(f"inconsistent results from {where}: {e}") from e


def _rows(tx: Any, query: str) -> list[dict[str, Any]]:
    return [record.data() for record in tx.run(query)]


def _parse_version(text: str) -> Optional[tuple[int, int]]:
    match = re.match(r"\s*(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _check_version(components: Iterable[Mapping[str, Any]]) -> None:
    """Raise unless the Neo4j kernel is at least :data:`MIN_VERSION`."""
    for component in components:
        if component.get("name") != "Neo4j Kernel":
            continue
        versions = component.get("versions") or []
        version = _parse_version(str(versions[0])) if versions else None
        if version is None:
            raise InstanceLoadError(f"cannot read the Neo4j version from {versions!r}")
        if version < MIN_VERSION:
            raise InstanceLoadError(
                f"Neo4j {versions[0]} is not supported; c2score needs {MIN_VERSION[0]}.{MIN_VERSION[1]} "
                "or later (for valueType())"
            )
        return
    raise InstanceLoadError("cannot find the Neo4j version in dbms.components()")


# ---------------------------------------------------------------------------
# Building Abs(G) from the aggregated rows
# ---------------------------------------------------------------------------


def _edge_key(row: Mapping[str, Any]) -> EdgeKey:
    return frozenset([row["rt"]]), frozenset(row["sl"]), frozenset(row["tl"])


def _properties(
    size: int, observed: Mapping[str, Mapping[DataType, int]]
) -> dict[str, PropertySpec]:
    """Required iff observed on every member (Eqs. 7 and 9); the type is unified (Section 4.1)."""
    return {
        key: PropertySpec(unify(types), sum(types.values()) == size)
        for key, types in observed.items()
    }


def _add_property(
    table: dict[Any, dict[str, dict[DataType, int]]], group: Any, row: Mapping[str, Any]
) -> None:
    by_type = table.setdefault(group, {}).setdefault(row["k"], {})
    data_type = from_neo4j_value_type(row["t"])
    by_type[data_type] = by_type.get(data_type, 0) + row["c"]


def _degree_bounds(
    rows: Iterable[Mapping[str, Any]], outgoing: bool
) -> dict[EdgeKey, tuple[int, int, int]]:
    """Merge degree rows into (min, max, n) per edge group."""
    bounds: dict[EdgeKey, tuple[int, int, int]] = {}
    for row in rows:
        node, other = frozenset(row["ls"]), frozenset(row["other"])
        key = (frozenset([row["rt"]]), node, other) if outgoing else (frozenset([row["rt"]]), other, node)
        low, high, n = row["dmin"], row["dmax"], row["n"]
        if key in bounds:
            old_low, old_high, old_n = bounds[key]
            low, high, n = min(low, old_low), max(high, old_high), n + old_n
        bounds[key] = (low, high, n)
    return bounds


def _interval(bounds: tuple[int, int, int], group_size: int) -> Interval:
    """Eqs. 10 and 11: nodes of the group without such an edge have degree 0."""
    low, high, n = bounds
    if n > group_size:
        raise ValueError(f"{n} nodes with an edge, but only {group_size} nodes in the group")
    return round_observed(0 if n < group_size else low, high)


def _build(
    node_counts: list[dict[str, Any]],
    node_properties: list[dict[str, Any]],
    edge_counts: list[dict[str, Any]],
    edge_properties: list[dict[str, Any]],
    out_degrees: list[dict[str, Any]],
    in_degrees: list[dict[str, Any]],
) -> Instance:
    node_sizes: dict[Labels, int] = {}
    for row in node_counts:
        labels = frozenset(row["ls"])
        node_sizes[labels] = node_sizes.get(labels, 0) + row["c"]
    node_props: dict[Labels, dict[str, dict[DataType, int]]] = {}
    for row in node_properties:
        _add_property(node_props, frozenset(row["ls"]), row)

    edge_sizes: dict[EdgeKey, int] = {}
    for row in edge_counts:
        key = _edge_key(row)
        edge_sizes[key] = edge_sizes.get(key, 0) + row["c"]
    edge_props: dict[EdgeKey, dict[str, dict[DataType, int]]] = {}
    for row in edge_properties:
        _add_property(edge_props, _edge_key(row), row)

    out_bounds = _degree_bounds(out_degrees, outgoing=True)
    in_bounds = _degree_bounds(in_degrees, outgoing=False)

    # Every label set of query (1) becomes a node type, so that each edge
    # type finds its endpoints.
    node_types = [
        InstanceNodeType(labels, _properties(size, node_props.get(labels, {})))
        for labels, size in node_sizes.items()
    ]
    edge_types = []
    for key, size in edge_sizes.items():
        labels, source, target = key
        if key not in out_bounds or key not in in_bounds:
            raise ValueError(f"no degree counts for the edge group {sorted(labels)}")
        if source not in node_sizes or target not in node_sizes:
            raise ValueError(f"no nodes for an endpoint of the edge group {sorted(labels)}")
        cardinality = Cardinality(
            _interval(out_bounds[key], node_sizes[source]),
            _interval(in_bounds[key], node_sizes[target]),
        )
        edge_types.append(
            InstanceEdgeType(labels, source, target, cardinality, _properties(size, edge_props.get(key, {})))
        )
    return Instance(tuple(node_types), tuple(edge_types))
