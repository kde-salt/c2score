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
Schemas without Ground Truth* (EDBT 2027). The library implements the measure with a few
refinements; see [How the library computes C2-score](https://github.com/kde-salt/c2score#how-the-library-computes-c2-score).

## Installation

```bash
pip install c2score
```

Python 3.10 or later is required. The dependencies (`numpy`, `jsonschema`, and the `neo4j` driver 5.x)
are installed automatically.

## Quick start

### With Neo4j

```python
from c2score import load_neo4j, load_schema_json, evaluate

instance = load_neo4j("bolt://localhost:7687", auth=("neo4j", "password"), database="neo4j")
schema = load_schema_json("schema.json")   # see "Writing a schema" below
result = evaluate(instance, schema)

print(result.c2)                            # C2-score
print(result.coverage, result.conciseness)  # overall Coverage and Conciseness
print(result.cov_v, result.cov_e, result.con_v, result.con_e)
print(result.redundant_types)               # {"node_types": [...], "edge_types": [...]}
```

`load_neo4j` reads only a summary of the graph (the node and edge types with their properties
and cardinalities); the aggregation runs on the Neo4j server. The summary can be saved and
evaluated later without a database:

```python
instance.to_json("instance.json")

from c2score import Instance
instance = Instance.from_json("instance.json")
```

### Without Neo4j

The example below evaluates the schema of Fig. 1(b) of the paper against the instance of
Fig. 1(a). Save the instance summary as `fig1a-instance.json`:

```json
{
  "format": "c2score-instance",
  "version": 1,
  "node_types": [
    {"labels": ["OFFICIAL", "USER"], "properties": {
      "bio": {"type": "STRING", "required": false},
      "id": {"type": "INTEGER", "required": true},
      "verifiedAt": {"type": "DATETIME", "required": true}}},
    {"labels": ["POST"], "properties": {"text": {"type": "STRING", "required": true}}},
    {"labels": ["USER"], "properties": {
      "bio": {"type": "STRING", "required": false},
      "id": {"type": "INTEGER", "required": true}}}
  ],
  "edge_types": [
    {"labels": ["CREATES"], "source_labels": ["OFFICIAL", "USER"], "target_labels": ["POST"],
     "cardinality": {"out": "0..1", "in": "0..1"}, "properties": {}},
    {"labels": ["CREATES"], "source_labels": ["USER"], "target_labels": ["POST"],
     "cardinality": {"out": "0..1", "in": "0..1"}, "properties": {}},
    {"labels": ["LIKES"], "source_labels": ["OFFICIAL", "USER"], "target_labels": ["POST"],
     "cardinality": {"out": "0..1", "in": "0..1"}, "properties": {}},
    {"labels": ["LIKES"], "source_labels": ["USER"], "target_labels": ["POST"],
     "cardinality": {"out": "0..1", "in": "0..1"}, "properties": {}}
  ]
}
```

Then run:

```python
from c2score import Instance, evaluate, load_schema_json

schema = load_schema_json({
    "node_types": [
        {"id": "userType", "labels": ["USER"],
         "properties": {"id": {"type": "INTEGER"}}},
        {"id": "officialType", "labels": ["OFFICIAL"], "supertypes": ["userType"],
         "properties": {"verifiedAt": {"type": "DATETIME"}}},
        {"id": "postType1", "labels": ["POST"], "properties": {"text": {"type": "STRING"}}},
        {"id": "postType2", "labels": ["POST"], "properties": {"text": {"type": "STRING"}}},
    ],
    "edge_types": [
        {"id": "likesType", "labels": ["LIKES"], "source": "userType", "target": "postType1",
         "cardinality": {"out": "0..*", "in": "0..*"}},
        {"id": "createsType1", "labels": ["CREATES"], "source": "userType", "target": "postType1",
         "cardinality": {"out": "1..1", "in": "0..*"}},
        {"id": "createsType2", "labels": ["CREATES"], "source": "userType", "target": "postType2",
         "cardinality": {"out": "1..1", "in": "0..*"}},
    ],
})
instance = Instance.from_json("fig1a-instance.json")
result = evaluate(instance, schema)

print(round(result.c2, 4))     # 0.6691
print(result.redundant_types)  # {'node_types': ['postType2'], 'edge_types': ['createsType1', 'createsType2']}
```

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
`{"labels": ["OFFICIAL", "USER"], "schema_type": "officialType", "similarity": 0.875}`.
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
| `redundancy_ratio` | γ (replaces ε) | `0.15` | A finite number greater than 0. See [How the library computes C2-score](https://github.com/kde-salt/c2score#how-the-library-computes-c2-score) |
| `use_types` | — | `True` | `True` or `False` (a NumPy boolean is accepted): whether property data types are compared |
| `allow_any` | — | `True` | `True` or `False`: whether `ANY` matches every type. Ignored when `use_types` is `False` |

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

- **Neo4j 5.13 or later** (including the 2025.x releases), because property types are read with
  `valueType()`. An older server raises `InstanceLoadError`.
- APOC is not needed.
- Pass either `uri` (with `auth`) or an existing `neo4j.Driver` as `driver`, not both. A driver
  passed in is not closed. `database` selects the database; if omitted, the default database is used.
- The session is read-only. All aggregation runs on the server; the graph is not copied to Python.
- Errors from the driver (connection, authentication, queries) are raised as `InstanceLoadError`.
- Load a graph that is not being written to. The queries do not see a single snapshot, so a
  concurrent write can make the counts disagree (`InstanceLoadError`) or mix the states before and
  after the write.

How the graph is summarized:

- A node type is a set of node labels; nodes without labels form the type with the empty label set.
  An edge type is the combination of the edge's type and the label sets of its two end nodes.
- A property is mandatory in a type if every node (or edge) of the type has it, and optional otherwise.
- Data types: `STRING`, `INTEGER`, `FLOAT`, and `BOOLEAN` map to themselves; `DATE`,
  `LOCAL DATETIME`, and `ZONED DATETIME` map to `DATETIME`; everything else (lists, points,
  durations, times) maps to `ANY`. If a property has several types within a type, INTEGER and
  FLOAT together become `FLOAT`, and any other mixture becomes `ANY`.
- Cardinalities are the observed minimum and maximum degrees (a node without such an edge counts
  as 0), rounded to the four intervals: the lower bound becomes 0 or 1, the upper bound 1 or `*`.

## How the library computes C2-score

The library follows the definitions of the paper, with the refinements below. Each item is marked
**Same as the paper**, **Differs from the paper** (with what the paper does), or **Not in the paper**
(the library fills a choice that the paper leaves open, or adds a feature).

**Coverage**

- **Same as the paper.** For each type of the instance, Coverage takes the most similar type of
  the flattened schema (inheritance resolved). A schema type may be the best match of several
  instance types.
- **Same as the paper.** Labels are compared as sets with the Dice coefficient, for node types and
  edge types alike. An edge type of the instance has one label, its relationship type.
- **Not in the paper.** The data types are `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATETIME`,
  and `ANY`, and values are assigned to them as described in [Neo4j requirements](https://github.com/kde-salt/c2score#neo4j-requirements)
  (for example, INTEGER and FLOAT values of one property give `FLOAT`). The paper treats data
  types abstractly; this set of types and the mapping of values are defined by the library.
- **Not in the paper.** `ANY` matches every type by default. With `allow_any=False`, `ANY` matches
  nothing: an `ANY` property in the instance never matches, and an `ANY` in the schema (including
  an omitted `type`) raises `SchemaError`. With `use_types=False`, only the mandatory and optional
  property names are compared.
- **Differs from the paper.** The similarity of two edge types is 0 if their source label sets, or
  their target label sets, share no label. Two empty label sets (nodes without labels) count as
  matching. The paper's edge similarity has no such filter; it only weighs the endpoint similarity.
- **Differs from the paper.** Cardinalities in the schema are limited to the four intervals `0..1`,
  `0..*`, `1..1`, `1..*`, and the cardinalities observed in the instance are rounded to them (the
  lower bound to 0 or 1, the upper bound to 1 or `*`). The paper allows any interval n..m and uses
  the observed minimum and maximum degrees as they are.
- **Same as the paper.** If the instance has no node types (or no edge types), the corresponding
  Coverage is 1. If the instance has some but the schema has none, it is 0.

**Conciseness**

- **Differs from the paper.** Redundancy is decided per **definition as written in the schema**,
  and Con_V and Con_E are 1 minus the fraction of redundant node and edge definitions. The paper
  removes the types of the flattened schema S+ one at a time and divides by their number.
  To test a definition, the library removes it, also removes what can no longer be defined
  (a type loses what it inherited from the removed definition, and an edge definition whose
  endpoint is removed or no longer determined is removed as well), flattens the rest again, and
  recomputes Coverage.
- **Differs from the paper.** A definition is redundant if removing it lowers neither Cov_V nor
  Cov_E by the threshold or more. The thresholds are θ_V = γ · Cov_V / |V(S+)| and
  θ_E = γ · Cov_E / |E(S+)|, where γ is `redundancy_ratio` and |V(S+)|, |E(S+)| are the numbers of
  node and edge types of the flattened schema. The paper uses a constant threshold ε instead.
  If a threshold is 0, a removal that does not lower Coverage counts as redundant.
- **Same as the paper.** With no definitions of a kind, the Conciseness of that kind is 1.
- Consequences of the rules above: two identical definitions are both redundant, since removing
  either one loses nothing; and if the instance has no edges, every edge definition is redundant.

**Schema semantics**

- **Not in the paper.** Inherited properties, cardinalities, and endpoints are resolved by the
  precedence order described in [Inheritance](https://github.com/kde-salt/c2score#inheritance); the paper only requires a
  deterministic rule.
- **Not in the paper.** Edge types may omit their endpoints and inherit them; in the paper every
  edge type has its own endpoints.
- **Same as the paper.** Edge types are expanded to every combination of the descendants of their
  two endpoints. Flattened edge types with identical contents are counted once.
- **Same as the paper.** C2-score is the harmonic mean of the overall Coverage and the overall
  Conciseness, each the average of its node and edge parts.

## Performance

`evaluate` computes edge similarities only for the schema edge types that can match (their edge
labels and the labels of both endpoints overlap); all other similarities are 0 by definition.
For Conciseness, removing a definition that nothing else inherits from or depends on leaves the
rest of the flattened schema unchanged, so Coverage is recomputed from the similarities already
computed; other removals flatten the reduced schema again. The results are the same as those of
the straightforward computation.

Measured on an Apple M1 Ultra (20 cores, 128 GB of memory) with macOS 15.5, Python 3.12, and
Neo4j 5.26 on the same machine. `evaluate` was run with `use_types=False` and the other
parameters at their defaults. Peak memory is the maximum resident set size of the Python process.

| Task | Size | Time | Peak memory |
| --- | --- | --- | --- |
| `load_neo4j` | DBpedia graph, about 20 million nodes and 45 million edges | 340 s | 1.2 GB |
| `evaluate` | The summary of that graph, with a schema of 446 node definitions and 178,786 edge definitions (465,401 edge types after flattening) | 450 s | 1.8 GB |
| `evaluate` | A graph whose summary has 1,743 edge types, with an extracted schema | under 1 s | — |

`load_neo4j` runs its aggregation on the Neo4j server, so its time depends mostly on the size of
the graph and on the server. Most of the time of `evaluate` goes to Conciseness, which tests every
definition of the schema; it grows with the number of definitions and the number of types in the
instance summary. Save a loaded summary with `Instance.to_json` to evaluate several schemas without
reading the graph again.

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

## License

MIT. See [LICENSE](https://github.com/kde-salt/c2score/blob/main/LICENSE).
