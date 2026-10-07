import copy
import json
import pickle
from pathlib import Path

import pytest

from c2score import SchemaError, load_schema_json
from c2score.cardinality import ONE_ONE, ZERO_MANY, Cardinality
from c2score.datatypes import DataType
from c2score.model import PropertySpec

DATA = Path(__file__).parent / "data"
EXAMPLES = ["example1.json", "example2.json", "example3.json"]

BASE = {
    "node_types": [
        {"id": "personType", "labels": ["Person"], "properties": {"name": {"type": "STRING"}}},
        {"id": "companyType", "labels": ["Company"]},
    ],
    "edge_types": [
        {"id": "worksAtType", "labels": ["WORKS_AT"], "source": "personType", "target": "companyType"},
    ],
}


def base():
    return copy.deepcopy(BASE)


def assert_rejected(data, *fragments):
    with pytest.raises(SchemaError) as info:
        load_schema_json(data)
    message = str(info.value)
    for fragment in fragments:
        assert fragment in message
    return message


# ---------------------------------------------------------------------------
# Examples 1-3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLES)
def test_examples_load_from_file_and_dict(name):
    path = DATA / name
    from_file = load_schema_json(path)
    assert load_schema_json(str(path)) == from_file
    with path.open(encoding="utf-8") as f:
        assert load_schema_json(json.load(f)) == from_file


def test_example1_contents():
    schema = load_schema_json(DATA / "example1.json")
    assert [t.id for t in schema.node_types] == ["userType", "officialType", "postType1", "postType2"]
    assert [t.id for t in schema.edge_types] == ["likesType", "createsType1", "createsType2"]
    official = schema["officialType"]
    assert official.labels == frozenset({"OFFICIAL"})  # own labels only
    assert official.supertypes == ("userType",)
    assert official.properties == {"verifiedAt": PropertySpec(DataType.DATETIME, True)}
    creates = schema["createsType1"]
    assert creates.labels == frozenset({"CREATES"})
    assert (creates.source, creates.target) == ("userType", "postType1")
    assert creates.cardinality == Cardinality(ONE_ONE, ZERO_MANY)
    assert schema.has_cardinality


def test_example2_defaults():
    schema = load_schema_json(DATA / "example2.json")
    person = schema["personType"]
    name, email = person.properties["name"], person.properties["email"]
    assert (name.type, name.required) == (DataType.ANY, True)
    assert (email.type, email.required) == (DataType.ANY, False)
    knows = schema["knowsType"]
    assert knows.cardinality is None
    assert knows.supertypes == ()
    assert not schema.has_cardinality


def test_example3_inheritance():
    schema = load_schema_json(DATA / "example3.json")
    assert schema.precedence("managerType") == ("managerType", "employeeType", "shareholderType", "personType")
    manages = schema["managesType"]
    assert manages.source == "managerType"
    assert manages.target is None  # inherited from worksAtType when flattened
    assert manages.cardinality is None
    assert schema.precedence("managesType") == ("managesType", "worksAtType")


def test_loaded_schema_pickles():
    schema = load_schema_json(DATA / "example3.json")
    restored = pickle.loads(pickle.dumps(schema))
    assert restored == schema
    assert restored.precedence("managerType") == schema.precedence("managerType")
    assert not restored["personType"].properties["email"].required


def test_empty_schema():
    schema = load_schema_json({})
    assert schema.node_types == () and schema.edge_types == ()


def test_type_names_are_case_insensitive():
    data = base()
    data["node_types"][0]["properties"] = {"a": {"type": "integer"}, "b": {"type": "DateTime"}, "c": {"type": "any"}}
    props = load_schema_json(data)["personType"].properties
    assert [props[k].type for k in "abc"] == [DataType.INTEGER, DataType.DATETIME, DataType.ANY]


def test_format_and_version_are_optional():
    data = base()
    data["format"] = "c2score-schema"
    data["version"] = "1"
    load_schema_json(data)


def test_rejects_other_source_types():
    with pytest.raises(TypeError):
        load_schema_json(42)


# ---------------------------------------------------------------------------
# (a) JSON Schema: R1-R6
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d.update(nodes=[]), "<root>"),
        (lambda d: d["node_types"][0].update(label=["X"]), "node_types[0] (id='personType')"),
        (
            lambda d: d["node_types"][0]["properties"]["name"].update(optional=True),
            "node_types[0] (id='personType').properties.name",
        ),
        (lambda d: d["edge_types"][0].update(src="personType"), "edge_types[0] (id='worksAtType')"),
    ],
)
def test_r1_unknown_keys(mutate, fragment):
    data = base()
    mutate(data)
    assert_rejected(data, fragment, "Additional properties are not allowed")


@pytest.mark.parametrize(
    "cardinality",
    [{"out": "0..2", "in": "0..*"}, {"out": "1..1"}, {"in": "0..*"}, {"out": "1", "in": "*"}],
)
def test_r2_cardinality(cardinality):
    data = base()
    data["edge_types"][0]["cardinality"] = cardinality
    assert_rejected(data, "edge_types[0] (id='worksAtType').cardinality")


def test_r3_format():
    data = base()
    data["format"] = "pg-schema"
    assert_rejected(data, "format")


@pytest.mark.parametrize("key", ["labels", "supertypes"])
def test_r4_duplicates_in_arrays(key):
    data = base()
    data["node_types"].append({"id": "employeeType", key: ["personType", "personType"]})
    assert_rejected(data, f"node_types[2] (id='employeeType').{key}", "non-unique")


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [({"required": "yes"}, "required"), ({"required": 1}, "required"), ({"type": 1}, "type")],
)
def test_r5_property_value_types(spec, fragment):
    data = base()
    data["node_types"][0]["properties"]["name"] = spec
    assert_rejected(data, f"node_types[0] (id='personType').properties.name.{fragment}")


@pytest.mark.parametrize("bad_id", ["", 3, None])
def test_r6_id(bad_id):
    data = base()
    data["node_types"][1]["id"] = bad_id
    assert_rejected(data, "node_types[1].id")


def test_r6_missing_id():
    data = base()
    del data["edge_types"][0]["id"]
    assert_rejected(data, "edge_types[0]", "'id' is a required property")


# ---------------------------------------------------------------------------
# R7: broken JSON and duplicate keys
# ---------------------------------------------------------------------------


def test_r7_duplicate_key(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        '{"node_types": [{"id": "a", "labels": ["A"], "properties": {"x": {}, "x": {"required": false}}}]}',
        encoding="utf-8",
    )
    assert_rejected(path, "node_types[0] (id='a').properties", "duplicate key 'x'", str(path))


def test_r7_duplicate_top_level_key(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text('{"node_types": [], "node_types": []}', encoding="utf-8")
    assert_rejected(path, "<root>: duplicate key 'node_types'")


def test_r7_broken_json(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text('{"node_types": [', encoding="utf-8")
    assert_rejected(path, "not valid JSON", "line 1")


def test_r7_non_standard_constant(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text('{"version": NaN}', encoding="utf-8")
    assert_rejected(path, "NaN")


def test_r7_not_utf8(tmp_path):
    path = tmp_path / "schema.json"
    path.write_bytes('{"node_types": [{"id": "ä"}]}'.encode("latin-1"))
    assert_rejected(path, "not valid UTF-8")


@pytest.mark.parametrize("name", EXAMPLES)
def test_utf8_with_byte_order_mark(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"\xef\xbb\xbf" + (DATA / name).read_bytes())
    assert load_schema_json(path) == load_schema_json(DATA / name)


def test_byte_order_mark_only_at_the_start(tmp_path):
    # A second byte order mark is not whitespace, so the JSON is broken.
    path = tmp_path / "schema.json"
    path.write_bytes(b"\xef\xbb\xbf\xef\xbb\xbf{}")
    assert_rejected(path, "not valid JSON")


# ---------------------------------------------------------------------------
# (b) Semantic checks: R8-R13
# ---------------------------------------------------------------------------


def test_r8_duplicate_id_within_kind():
    data = base()
    data["node_types"].append({"id": "personType"})
    assert_rejected(data, "node_types[2] (id='personType').id", "duplicate id", "node_types[0]")


def test_r8_duplicate_id_across_kinds():
    data = base()
    data["edge_types"][0]["id"] = "companyType"
    assert_rejected(data, "edge_types[0] (id='companyType').id", "duplicate id 'companyType'", "node_types[1]")


def test_r9_unknown_supertype():
    data = base()
    data["node_types"][1]["supertypes"] = ["organizationType"]
    assert_rejected(data, "node_types[1] (id='companyType').supertypes[0]", "unknown id 'organizationType'")


def test_r9_node_type_inherits_edge_type():
    data = base()
    data["node_types"][1]["supertypes"] = ["worksAtType"]
    assert_rejected(data, "node_types[1] (id='companyType').supertypes[0]", "is an edge type")


def test_r9_edge_type_inherits_node_type():
    data = base()
    data["edge_types"][0]["supertypes"] = ["personType"]
    assert_rejected(data, "edge_types[0] (id='worksAtType').supertypes[0]", "is a node type")


def test_r9_self_reference():
    data = base()
    data["node_types"][0]["supertypes"] = ["personType"]
    assert_rejected(data, "node_types[0] (id='personType').supertypes[0]", "own supertype")


def test_r10_cycle_of_node_types():
    data = base()
    data["node_types"][0]["supertypes"] = ["companyType"]
    data["node_types"][1]["supertypes"] = ["personType"]
    assert_rejected(data, "inheritance cycle", "personType -> companyType -> personType", "supertypes")


def test_r10_cycle_of_edge_types():
    data = base()
    data["edge_types"] += [
        {"id": "r1", "supertypes": ["r3", "worksAtType"]},
        {"id": "r2", "supertypes": ["r1"]},
        {"id": "r3", "supertypes": ["r2"]},
    ]
    assert_rejected(data, "inheritance cycle", "r1 -> r3 -> r2 -> r1")


def test_r10_cycle_is_reported_where_it_closes():
    # The search goes a -> b -> c, and c's second supertype closes the cycle.
    data = {
        "node_types": [
            {"id": "a", "supertypes": ["b"]},
            {"id": "b", "supertypes": ["c"]},
            {"id": "c", "supertypes": ["z", "a"]},
            {"id": "z"},
        ]
    }
    message = assert_rejected(data, "node_types[2] (id='c').supertypes[1]: inheritance cycle a -> b -> c -> a")
    assert message.count("inheritance cycle") == 1


def test_r10_deep_hierarchy_without_cycle():
    depth = 5000
    data = {"node_types": [{"id": f"t{i}", "supertypes": [f"t{i + 1}"]} for i in range(depth)] + [{"id": f"t{depth}"}]}
    schema = load_schema_json(data)
    assert len(schema.precedence("t0")) == depth + 1


def test_r10_deep_cycle():
    depth = 5000
    data = {"node_types": [{"id": f"t{i}", "supertypes": [f"t{(i + 1) % depth}"]} for i in range(depth)]}
    message = assert_rejected(data, "inheritance cycle t0 -> t1 -> t2")
    assert message.count("inheritance cycle") == 1


@pytest.mark.parametrize("end", ["source", "target"])
def test_r11_endpoint_unknown(end):
    data = base()
    data["edge_types"][0][end] = "nobodyType"
    assert_rejected(data, f"edge_types[0] (id='worksAtType').{end}", "unknown id 'nobodyType'")


def test_r11_endpoint_is_edge_type():
    data = base()
    data["edge_types"].append({"id": "r", "source": "worksAtType", "target": "personType"})
    assert_rejected(data, "edge_types[1] (id='r').source", "'worksAtType' is not a node type")


@pytest.mark.parametrize("end", ["source", "target"])
def test_r12_endpoint_missing(end):
    data = base()
    del data["edge_types"][0][end]
    assert_rejected(data, f"edge_types[0] (id='worksAtType').{end}", f"{end} is not determined")


def test_r12_endpoint_missing_through_inheritance():
    data = base()
    data["edge_types"] = [
        {"id": "a", "source": "personType"},
        {"id": "b", "supertypes": ["a"]},
    ]
    message = assert_rejected(data, "edge_types[0] (id='a').target", "edge_types[1] (id='b').target")
    assert "edge_types[1] (id='b').source" not in message  # inherited from a


def test_r12_endpoint_inherited_from_any_ancestor():
    data = base()
    data["edge_types"] += [
        {"id": "a", "supertypes": ["worksAtType"]},
        {"id": "b", "supertypes": ["a"], "source": "companyType"},
    ]
    schema = load_schema_json(data)
    assert schema.precedence("b") == ("b", "a", "worksAtType")


@pytest.mark.parametrize("type_name", ["DATE", "TEXT", "list", "\u017ftring", "str\u0131ng"])
def test_r13_unknown_type(type_name):
    data = base()
    data["edge_types"][0]["properties"] = {"since": {"type": type_name}}
    assert_rejected(data, "edge_types[0] (id='worksAtType').properties.since.type", f"unknown type {type_name!r}")


def test_reports_every_violation_of_a_phase():
    data = base()
    data["node_types"][1]["supertypes"] = ["missingType"]
    data["edge_types"][0]["target"] = "missingType"
    message = assert_rejected(data, "node_types[1] (id='companyType').supertypes[0]")
    assert "edge_types[0] (id='worksAtType').target" in message


def test_schema_error_is_value_error():
    with pytest.raises(ValueError):
        load_schema_json({"format": "x"})


# ---------------------------------------------------------------------------
# References to a duplicated id
# ---------------------------------------------------------------------------


def test_r8_duplicate_id_does_not_cause_r9_errors():
    # The edge type 'companyType' duplicates a node type id; references to the
    # ambiguous id are not checked again, so only R8 is reported.
    data = base()
    data["edge_types"].append({"id": "companyType", "source": "personType", "target": "personType"})
    data["edge_types"].append({"id": "r", "supertypes": ["companyType"]})
    data["node_types"].append({"id": "employeeType", "supertypes": ["companyType"]})
    message = assert_rejected(data, "duplicate id 'companyType'")
    assert message.count("\n  ") == 1
    assert "supertypes" not in message


def test_r8_duplicate_id_does_not_cause_r11_errors():
    data = base()
    data["edge_types"].append({"id": "personType", "source": "companyType", "target": "companyType"})
    data["edge_types"].append({"id": "r", "source": "personType", "target": "personType"})
    message = assert_rejected(data, "duplicate id 'personType'")
    assert message.count("\n  ") == 1


def test_r9_and_r11_still_checked_for_other_ids_when_an_id_is_duplicated():
    data = base()
    data["node_types"].append({"id": "personType"})
    data["edge_types"].append({"id": "r", "supertypes": ["missingType"], "target": "worksAtType"})
    assert_rejected(
        data,
        "duplicate id 'personType'",
        "edge_types[1] (id='r').supertypes[0]: unknown id 'missingType'",
        "edge_types[1] (id='r').target: 'worksAtType' is not a node type",
    )


# ---------------------------------------------------------------------------
# Keys of a dict source
# ---------------------------------------------------------------------------


def test_non_string_property_name_is_rejected():
    data = base()
    data["node_types"][0]["properties"][1] = {"type": "STRING"}
    assert_rejected(data, "node_types[0] (id='personType').properties: key 1 is not a string")


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d.update({1: []}), "<root>: key 1 is not a string"),
        (lambda d: d["node_types"][1].update({None: 1}), "node_types[1] (id='companyType'): key None is not a string"),
        (
            lambda d: d["node_types"][0]["properties"]["name"].update({("type",): "STRING"}),
            "node_types[0] (id='personType').properties.name: key ('type',) is not a string",
        ),
    ],
)
def test_non_string_keys_are_rejected_anywhere(mutate, fragment):
    data = base()
    mutate(data)
    assert_rejected(data, fragment)


# ---------------------------------------------------------------------------
# Long values are shortened in messages
# ---------------------------------------------------------------------------

LONG = "A" * 5000


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d["node_types"][0].update(labels=LONG), "is not of type 'array'"),
        (lambda d: d["node_types"][0].update(labels=[LONG, LONG]), "has non-unique elements"),
        (lambda d: d["node_types"][0].update({LONG: 1}), "was unexpected"),
        (lambda d: d["node_types"][0]["properties"].update(p={"required": LONG}), "is not of type 'boolean'"),
        (lambda d: d["node_types"][0]["properties"].update(p={"type": LONG}), "unknown type"),
        (lambda d: d["edge_types"][0].update(cardinality={"out": LONG, "in": "0..*"}), "is not one of"),
        (lambda d: d["node_types"].append({"id": LONG, "supertypes": [LONG + "x"]}), "unknown id"),
        (lambda d: d["node_types"].extend([{"id": LONG}, {"id": LONG}]), "duplicate id"),
    ],
)
def test_long_values_are_shortened(mutate, fragment):
    data = base()
    mutate(data)
    message = assert_rejected(data, fragment, "...")
    assert len(message) < 500


def test_many_unknown_keys_are_listed_briefly():
    data = base()
    data["node_types"][0].update({f"k{i}": 1 for i in range(8)})
    assert_rejected(data, "('k0', 'k1', 'k2', 'k3', 'k4' and 3 more were unexpected)")
