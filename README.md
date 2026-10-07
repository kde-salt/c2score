# c2score

C2-score measures the quality of a property graph schema **without a ground-truth schema**.
Given a property graph instance and a candidate schema, it computes two scores in [0, 1]:

- **Coverage**: how well the schema explains the node and edge types observed in the instance
  (labels, properties and their data types, endpoints, and cardinalities).
- **Conciseness**: how free the schema is of redundant definitions, that is, definitions that
  can be removed without a meaningful loss of Coverage.

C2-score is the harmonic mean of the two. Each score is also reported separately for nodes and
edges, together with the redundant definitions and the best-matching schema type for every type
in the instance, so that you can see *why* a schema scores as it does.

The measure is described in the paper *C2-score: Quantitative Quality Evaluation of Property Graph
Schemas without Ground Truth* (EDBT 2027).
This library refines the definitions of the paper, so its scores can differ from those reported in
the paper. To reproduce the results of the paper, use the artifact repository
[kde-salt/C2-SCORE](https://github.com/kde-salt/C2-SCORE).

## Citation

If you use c2score in your research, please cite:

```bibtex
@inproceedings{yukawa2027c2score,
  author    = {Fusuke Yukawa and Hiroaki Shiokawa},
  title     = {{C2-score}: Quantitative Quality Evaluation of Property Graph Schemas without Ground Truth},
  booktitle = {Proceedings of the International Conference on Extending Database Technology (EDBT)},
  year      = {2027}
}
```

(The bibliographic details will be completed when the proceedings are published.)

## Installation

```bash
pip install c2score
```

Python 3.10 or later is required. The dependencies (`numpy`, `jsonschema`, and the `neo4j` driver 5.x)
are installed automatically.

## Quick start

### 1. Create an instance summary

c2score evaluates a schema against a summary of the graph: its node and edge types with their
properties and cardinalities. Read the summary from Neo4j and save it as `instance.json`. The
aggregation runs on the server; the graph is not copied to Python.

```python
from c2score import load_neo4j

instance = load_neo4j("bolt://localhost:7687", auth=("neo4j", "password"), database="neo4j")
instance.to_json("instance.json")
```

Suppose the summary below is created:

```json
{
  "format": "c2score-instance",
  "version": 1,
  "node_types": [
    {"labels": ["Company"], "properties": {"name": {"type": "STRING", "required": true}}},
    {"labels": ["Person"], "properties": {"name": {"type": "STRING", "required": true}}}
  ],
  "edge_types": [
    {"labels": ["WORKS_AT"], "source_labels": ["Person"], "target_labels": ["Company"],
     "cardinality": {"out": "0..1", "in": "0..*"}, "properties": {}}
  ]
}
```

### 2. Write a schema

Write the schema to evaluate as `schema.json` (see "Writing a schema" below), for example:

```json
{
  "node_types": [
    {"id": "personType", "labels": ["Person"], "properties": {"name": {"type": "STRING"}}},
    {"id": "companyType", "labels": ["Company"], "properties": {"name": {"type": "STRING"}}}
  ],
  "edge_types": [
    {"id": "worksAtType", "labels": ["WORKS_AT"], "source": "personType", "target": "companyType",
     "cardinality": {"out": "0..1", "in": "0..*"}}
  ]
}
```

### 3. Evaluate

```python
from c2score import Instance, evaluate, load_schema_json

instance = Instance.from_json("instance.json")
schema = load_schema_json("schema.json")
result = evaluate(instance, schema)

print(result.c2)                            # C2-score
print(result.coverage, result.conciseness)  # overall Coverage and Conciseness
print(result.cov_v, result.cov_e, result.con_v, result.con_e)
print(result.redundant_types)               # {"node_types": [...], "edge_types": [...]}
```

A saved summary can be evaluated against any number of schemas without reading the graph again.

### The result

`evaluate` returns an `EvaluationResult`, a read-only dataclass:

| Attribute | Meaning |
| --- | --- |
| `c2` | C2-score: the harmonic mean of `coverage` and `conciseness` (0 if both are 0) |
| `coverage`, `conciseness` | `(cov_v + cov_e) / 2` and `(con_v + con_e) / 2` |
| `cov_v`, `cov_e`, `con_v`, `con_e` | Coverage and Conciseness of node types and of edge types |
| `redundant_types` | `{"node_types": [id, ...], "edge_types": [id, ...]}`: the redundant definitions, in id order |
| `best_matches` | `{"node_types": [...], "edge_types": [...]}`: one entry per type of the instance (see below) |
| `params` | The parameters actually used, including the edge weights and whether they were chosen automatically (`"edge_weights_mode": "auto"` or `"user"`) |
| `to_dict()` | Everything above as a JSON-serializable `dict` |

The dataclass is frozen, but the dictionaries and lists in `redundant_types`, `best_matches`, and
`params` are ordinary Python objects: modifying them modifies the result. Use `to_dict()` to get an
independent copy that you can change.

An entry of `best_matches["node_types"]` looks like
`{"labels": ["Person"], "schema_type": "personType", "similarity": 1.0}`.
An entry of `best_matches["edge_types"]` also has `source_labels` and `target_labels` (of the
instance type) and `source` and `target` (the ids of the schema node types at the two ends).
Labels are sorted. When the best similarity is 0, `schema_type` (and, for edges, `source` and
`target`) is `None`. When several schema types tie, the first one in a fixed order is reported
(node types by id; edge types by the smallest id of the definitions that produce them, then by
the source and target ids).

## Writing a schema

A schema is a JSON object with two arrays, `node_types` and `edge_types`. Both may be omitted.
`load_schema_json` accepts a path to a UTF-8 JSON file (a byte order mark is allowed) or an
already parsed `dict`.

| Key | In | Required | Meaning and default |
| --- | --- | --- | --- |
| `id` | node and edge types | yes | A non-empty string, unique across node **and** edge types |
| `labels` | node and edge types | no | An array of labels without duplicates. Default `[]`. A type inheriting from others lists only its own labels |
| `properties` | node and edge types | no | An object mapping a property name to `{"type": ..., "required": ...}`. Default `{}` |
| `properties.*.type` | | no | `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATETIME`, or `ANY` (case-insensitive). **Default `ANY`** |
| `properties.*.required` | | no | `true` (mandatory) or `false` (optional). **Default `true`** |
| `supertypes` | node and edge types | no | The ids of the direct parents, without duplicates. **The order matters** (see below). Node types inherit only from node types, edge types only from edge types |
| `source`, `target` | edge types | no | The ids of the node types at the two ends. If omitted, they are inherited from the nearest parent edge type that has them |
| `cardinality` | edge types | no | `{"out": interval, "in": interval}`, both required. An interval is one of `"0..1"`, `"0..*"`, `"1..1"`, `"1..*"`. If omitted, it is inherited; if no ancestor has one, it is treated as `0..*` in both directions |

The top level may also contain `"format": "c2score-schema"` and a `"version"` string; both are optional.

A minimal schema, in which `name` is mandatory and `email` optional, both of type `ANY`:

```json
{
  "node_types": [
    {"id": "personType", "labels": ["Person"],
     "properties": {"name": {}, "email": {"required": false}}}
  ],
  "edge_types": [
    {"id": "knowsType", "labels": ["KNOWS"], "source": "personType", "target": "personType"}
  ]
}
```

### Inheritance

A type inherits the labels, properties, cardinality, and (for edge types) endpoints of its
ancestors.

- **Labels** are the union of the labels of the type and all its ancestors.
- **Properties, cardinality, and endpoints** are taken from the nearest type that defines them.
  The type itself comes first, followed by its ancestors in breadth-first order, visiting the
  `supertypes` of each type in the order written. The description of a property (type and
  `required`) is taken as a whole from one type.
- The `source` and `target` of an edge type are inherited independently. Every edge type must end
  up with both.
- **An edge type also applies to the descendants of its endpoints.** An edge type from `s` to `t`
  stands for every pair (s′, t′) where s′ is `s` or one of its descendants and t′ is `t` or one of
  its descendants.
- **Inheritance cycles are rejected**, and so is a type listing itself in `supertypes`.

Multiple inheritance and an inherited edge type:

```json
{
  "node_types": [
    {"id": "personType", "labels": ["Person"],
     "properties": {"name": {"type": "STRING"}, "email": {"type": "STRING", "required": false}}},
    {"id": "employeeType", "labels": ["Employee"], "supertypes": ["personType"],
     "properties": {"email": {"type": "STRING", "required": true}, "salary": {"type": "FLOAT"}}},
    {"id": "shareholderType", "labels": ["Shareholder"], "supertypes": ["personType"],
     "properties": {"email": {"type": "STRING", "required": false}, "shares": {"type": "INTEGER"}}},
    {"id": "managerType", "labels": ["Manager"], "supertypes": ["employeeType", "shareholderType"]},
    {"id": "companyType", "labels": ["Company"], "properties": {"name": {"type": "STRING"}}}
  ],
  "edge_types": [
    {"id": "worksAtType", "labels": ["WORKS_AT"], "source": "employeeType", "target": "companyType",
     "properties": {"since": {"type": "DATETIME"}},
     "cardinality": {"out": "1..1", "in": "0..*"}},
    {"id": "managesType", "labels": ["MANAGES"], "supertypes": ["worksAtType"], "source": "managerType",
     "properties": {"since": {"type": "DATETIME", "required": false}}}
  ]
}
```

Here `managerType` has the labels Manager, Employee, Shareholder, and Person, and its `email` is
mandatory because `employeeType` is listed before `shareholderType`. `managesType` has the labels
MANAGES and WORKS_AT, an optional `since`, the cardinality of `worksAtType`, and the target
`companyType` inherited from `worksAtType`.

### Validation

`load_schema_json` raises `SchemaError` (a subclass of `ValueError`) for every violation, with the
location in the message, for example `edge_types[1] (id='managesType').source: unknown id 'x'`.
Besides the rules above, it rejects unknown keys (to catch typos) and an object that repeats a key.
The format is also defined as a JSON Schema (Draft 2020-12), shipped with the package as
`c2score/resources/schema.json`
([view](https://github.com/kde-salt/c2score/blob/main/src/c2score/resources/schema.json)).

## Parameters

```python
evaluate(instance, schema, *, label_weight=0.5, edge_weights=None,
         redundancy_ratio=0.15, use_types=True, allow_any=True)
```

| Argument | Paper | Default | Allowed values |
| --- | --- | --- | --- |
| `label_weight` | α | `0.5` | A number in [0, 1]: the weight of labels against properties, for nodes and edges |
| `edge_weights` | (α_A, α_E, α_C) | `None` (automatic) | A tuple, list, or 1-D NumPy array of three numbers in [0, 1] summing to 1: the weights of the edge's own attributes, its endpoints, and its cardinality |
| `redundancy_ratio` | γ (replaces ε) | `0.15` | A finite number greater than 0. The larger it is, the more readily a definition counts as redundant |
| `use_types` | — | `True` | `True` or `False` (a NumPy boolean is accepted). See the table below |
| `allow_any` | — | `True` | `True` or `False`. See the table below |

`use_types` and `allow_any` decide how properties are compared. A property without `type` in the
schema has type `ANY`.

| `use_types` | `allow_any` | What is compared |
| --- | --- | --- |
| `True` | `True` (default) | Property names, mandatory or optional, and data types. `ANY` matches every type |
| `True` | `False` | The same, but `ANY` matches no type. Every property of the schema must have a type other than `ANY`; otherwise `evaluate` raises `SchemaError` |
| `False` | ignored | Property names and mandatory or optional only |

With `edge_weights=None`, the weights are (1/3, 1/3, 1/3) if any edge definition of the schema
writes a `cardinality`, and (0.5, 0.5, 0) otherwise. They are decided once, on the whole schema.

An argument out of range raises `ParameterError` (a subclass of `ValueError`).

The library names correspond to the paper as follows: `Instance` holds Abs(G), the types of the
instance; `Schema` is the candidate schema S; `cov_v`, `cov_e`, `con_v`, `con_e` are Cov_V, Cov_E,
Con_V, Con_E; `coverage` and `conciseness` are Cov and Con; `c2` is C2.

## Neo4j requirements

```python
load_neo4j(uri=None, *, auth=None, database=None, driver=None)
```

- Neo4j 5.13 or later. APOC is not needed.
- Connect with `uri` and `auth`, or pass an existing `neo4j.Driver` as `driver`. Choose the
  database with `database`; if omitted, the server's default database is used.
- Only read access is needed. Load a graph that is not being written to.
- Failures (connection, authentication, queries, an older server) raise `InstanceLoadError`.
- Reading a large graph can take several minutes. Save the summary once, as in step 1 of the
  quick start, and evaluate the saved file. The file also shows what was read from the graph.

## Notes

- A string passed to `load_schema_json` or `Instance.from_json` is always treated as a file
  path, never as JSON text. To pass a schema that is already in memory, pass a `dict`
  (for example the result of `json.loads`).
- A missing file raises `FileNotFoundError` (from `open`), not a c2score exception.
- `evaluate` raises `TypeError` if `instance` is not an `Instance` or `schema` is not a `Schema`.
- All exceptions raised by c2score derive from `C2ScoreError`: `SchemaError`, `InstanceLoadError`,
  and `ParameterError`.
- Not included in this release: reading PG-Schema, loading an instance from a JSON graph, and
  databases other than Neo4j.

## License

MIT. See [LICENSE](https://github.com/kde-salt/c2score/blob/main/LICENSE).
