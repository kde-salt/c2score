from fractions import Fraction

import pytest

from c2score.cardinality import (
    DEFAULT_CARDINALITY,
    ONE_MANY,
    ONE_ONE,
    STAR,
    ZERO_MANY,
    ZERO_ONE,
    Cardinality,
    bd,
    d_card,
    d_int,
    format_interval,
    parse_interval,
    round_observed,
)


@pytest.mark.parametrize(
    ("text", "interval"),
    [("0..1", (0, 1)), ("0..*", (0, STAR)), ("1..1", (1, 1)), ("1..*", (1, STAR))],
)
def test_parse_and_format(text, interval):
    assert parse_interval(text) == interval
    assert format_interval(interval) == text


@pytest.mark.parametrize("text", ["0..2", "1..0", "*", "0 ..1", "", None])
def test_parse_invalid(text):
    with pytest.raises(ValueError):
        parse_interval(text)


def test_format_invalid():
    with pytest.raises(ValueError):
        format_interval((0, 2))


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((0, 1), ZERO_ONE),
        ((0, 2), ZERO_MANY),
        ((0, 100), ZERO_MANY),
        ((1, 1), ONE_ONE),
        ((1, 2), ONE_MANY),
        ((3, 5), ONE_MANY),
    ],
)
def test_round_observed(bounds, expected):
    assert round_observed(*bounds) == expected


@pytest.mark.parametrize("bounds", [(0, 0), (-1, 1), (2, 1)])
def test_round_observed_invalid(bounds):
    with pytest.raises(ValueError):
        round_observed(*bounds)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (STAR, STAR, 1),
        (STAR, 1, 0),
        (1, STAR, 0),
        (0, STAR, 0),
        (0, 0, 1),
        (1, 1, 1),
        (0, 1, Fraction(1, 2)),
        (1, 0, Fraction(1, 2)),
        (2, 5, Fraction(1, 2)),
    ],
)
def test_bd(x, y, expected):
    assert bd(x, y) == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (ZERO_MANY, ZERO_MANY, 1),
        (ZERO_ONE, ZERO_MANY, Fraction(1, 2)),  # (bd(0,0) + bd(1,*)) / 2
        (ONE_ONE, ZERO_MANY, Fraction(1, 4)),  # (bd(1,0) + bd(1,*)) / 2
        (ZERO_ONE, ONE_ONE, Fraction(3, 4)),
        (ONE_MANY, ZERO_MANY, Fraction(3, 4)),
    ],
)
def test_d_int(a, b, expected):
    assert d_int(a, b) == pytest.approx(float(expected), abs=1e-12)
    assert d_int(b, a) == pytest.approx(float(expected), abs=1e-12)


def test_d_card():
    observed = Cardinality(ZERO_ONE, ZERO_ONE)
    # Fig. 1(a) LIKES against likesType (out 0..*, in 0..*): D_card = 0.5.
    assert d_card(observed, Cardinality(ZERO_MANY, ZERO_MANY)) == pytest.approx(0.5, abs=1e-12)
    # Against createsType1 (out 1..1, in 0..*): (3/4 + 1/2) / 2 = 5/8.
    assert d_card(observed, Cardinality(ONE_ONE, ZERO_MANY)) == pytest.approx(5 / 8, abs=1e-12)


def test_default_cardinality():
    assert DEFAULT_CARDINALITY == Cardinality(ZERO_MANY, ZERO_MANY)
    assert str(DEFAULT_CARDINALITY) == "out 0..* / in 0..*"
