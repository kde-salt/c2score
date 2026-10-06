import datetime
from fractions import Fraction

import numpy as np
import pytest

from c2score.datatypes import (
    DataType,
    from_neo4j_value_type,
    from_python_value,
    parse_schema_type,
    unify,
)

D = DataType


@pytest.mark.parametrize(
    ("value_type", "expected"),
    [
        ("STRING", D.STRING),
        ("STRING NOT NULL", D.STRING),
        ("INTEGER NOT NULL", D.INTEGER),
        ("FLOAT", D.FLOAT),
        ("BOOLEAN NOT NULL", D.BOOLEAN),
        ("DATE NOT NULL", D.DATETIME),
        ("LOCAL DATETIME NOT NULL", D.DATETIME),
        ("ZONED DATETIME NOT NULL", D.DATETIME),
        ("LOCAL TIME NOT NULL", D.ANY),
        ("ZONED TIME NOT NULL", D.ANY),
        ("DURATION NOT NULL", D.ANY),
        ("POINT NOT NULL", D.ANY),
        ("LIST<INTEGER NOT NULL> NOT NULL", D.ANY),
        ("LIST<STRING>", D.ANY),
        ("SOMETHING NEW", D.ANY),
    ],
)
def test_from_neo4j_value_type(value_type, expected):
    assert from_neo4j_value_type(value_type) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("text", D.STRING),
        ("2025-12-31", D.STRING),  # a plain string stays a string
        (1, D.INTEGER),
        (0, D.INTEGER),
        (1.5, D.FLOAT),
        (True, D.BOOLEAN),
        (False, D.BOOLEAN),
        (datetime.date(2025, 12, 31), D.DATETIME),
        (datetime.datetime(2025, 12, 31, 12, 0), D.DATETIME),
        ({"$datetime": "2025-12-31"}, D.DATETIME),
        ({"$datetime": "2025-12-31T12:00:00Z"}, D.DATETIME),
        ({"$datetime": 1}, D.ANY),
        ({"$datetime": "2025-12-31", "x": 1}, D.ANY),
        ({"a": 1}, D.ANY),
        ([1, 2], D.ANY),
        (datetime.time(12, 0), D.ANY),
        (np.int64(3), D.INTEGER),
        (np.int32(3), D.INTEGER),
        (np.float64(1.5), D.FLOAT),
        (np.float32(1.5), D.FLOAT),
        (np.bool_(True), D.BOOLEAN),
        (Fraction(1, 2), D.FLOAT),
    ],
)
def test_from_python_value(value, expected):
    assert from_python_value(value) is expected


def test_from_python_value_rejects_none():
    with pytest.raises(ValueError):
        from_python_value(None)


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        ([D.STRING], D.STRING),
        ([D.INTEGER, D.INTEGER], D.INTEGER),
        ([D.INTEGER, D.FLOAT], D.FLOAT),
        ([D.FLOAT, D.INTEGER, D.FLOAT], D.FLOAT),
        ([D.INTEGER, D.STRING], D.ANY),
        ([D.INTEGER, D.FLOAT, D.STRING], D.ANY),
        ([D.INTEGER, D.ANY], D.ANY),
        ([D.ANY], D.ANY),
        ([D.BOOLEAN, D.INTEGER], D.ANY),
    ],
)
def test_unify(types, expected):
    assert unify(types) is expected


def test_unify_date_and_datetime_values():
    # DATE and ZONED DATETIME both map to DATETIME, so they do not mix.
    observed = [from_neo4j_value_type("DATE NOT NULL"), from_neo4j_value_type("ZONED DATETIME NOT NULL")]
    assert unify(observed) is D.DATETIME


def test_unify_empty():
    with pytest.raises(ValueError):
        unify([])


@pytest.mark.parametrize(
    ("name", "expected"),
    [("STRING", D.STRING), ("string", D.STRING), ("Integer", D.INTEGER), ("dateTime", D.DATETIME), ("any", D.ANY)],
)
def test_parse_schema_type(name, expected):
    assert parse_schema_type(name) is expected


@pytest.mark.parametrize("name", ["DATE", "TEXT", "", " STRING"])
def test_parse_schema_type_unknown(name):
    with pytest.raises(ValueError, match="unknown type"):
        parse_schema_type(name)
