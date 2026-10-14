# Changelog

## 0.1.1 (2026-10-14)

### Changed

- The README and the package description refer to the final title of the paper,
  *C2-score: Evaluating Schema Fit to Property Graphs* (EDBT 2027).

No changes to the code or to the scores.

## 0.1.0 (2026-10-13)

First release.

### Added

- `load_schema_json`: read a schema from a JSON file or a `dict` and validate it. Every violation
  raises `SchemaError` with its location. The format is also shipped as a JSON Schema
  (`c2score/resources/schema.json`).
- `load_neo4j`: summarize a Neo4j 5.13+ database into its node and edge types (labels,
  properties with data types, and cardinalities). The aggregation runs on the server in a
  read-only session; APOC is not needed.
- `Instance.to_json` and `Instance.from_json`: save and restore such a summary.
- `evaluate`: compute C2-score with Coverage and Conciseness for node and edge types, the
  redundant definitions, and the best-matching schema type for every instance type.
  Parameters: `label_weight`, `edge_weights`, `redundancy_ratio`, `use_types`, `allow_any`.
- Schema features: data types (`STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATETIME`, `ANY`),
  mandatory and optional properties, cardinalities (`0..1`, `0..*`, `1..1`, `1..*`), and multiple
  inheritance of node and edge types, including inherited edge endpoints.
- Exceptions: `C2ScoreError` and its subclasses `SchemaError`, `InstanceLoadError`, and `ParameterError`.
- Scales to large graphs and schemas: edge similarities are computed only for schema edge types
  that can match, and Conciseness reuses the computed similarities when removing a definition
  does not change the rest of the flattened schema. A DBpedia graph with about 20 million nodes
  and 45 million edges loads in about 6 minutes and is evaluated against a schema with about
  180,000 edge definitions in about 8 minutes.

The library refines the definitions of the paper, so its scores can differ from those reported in
the paper. To reproduce the results of the paper, use https://github.com/kde-salt/C2-SCORE.
