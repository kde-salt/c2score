"""The public entry point :func:`evaluate` and its result."""

from __future__ import annotations

import copy
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .conciseness import ConcisenessResult, conciseness
from .coverage import CoverageTables, Match
from .datatypes import DataType
from .errors import ParameterError, SchemaError
from .flatten import flatten
from .model import Instance, Schema
from .similarity import Similarity, SimilarityParams

#: Edge weights (alpha_A, alpha_E, alpha_C) when some edge definition writes a cardinality.
WEIGHTS_WITH_CARDINALITY = (1 / 3, 1 / 3, 1 / 3)
#: Edge weights when no edge definition writes a cardinality.
WEIGHTS_WITHOUT_CARDINALITY = (0.5, 0.5, 0.0)


@dataclass(frozen=True)
class EvaluationResult:
    """The C2-score of a schema for an instance, with its components.

    ``redundant_types`` lists the redundant definition ids (in id order) under
    ``"node_types"`` and ``"edge_types"``. ``best_matches`` gives, for each
    instance type, the schema type with the highest similarity. ``params``
    records the parameters actually used.

    The dataclass is frozen, but these dicts and lists are not: modifying
    them modifies the result. :meth:`to_dict` returns an independent copy.
    """

    c2: float
    cov_v: float
    cov_e: float
    con_v: float
    con_e: float
    coverage: float
    conciseness: float
    redundant_types: dict[str, list[str]]
    best_matches: dict[str, list[dict[str, Any]]]
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return every field as plain JSON-serializable values."""
        params = dict(self.params)
        params["edge_weights"] = list(params["edge_weights"])
        return {
            "c2": self.c2,
            "cov_v": self.cov_v,
            "cov_e": self.cov_e,
            "con_v": self.con_v,
            "con_e": self.con_e,
            "coverage": self.coverage,
            "conciseness": self.conciseness,
            "redundant_types": copy.deepcopy(self.redundant_types),
            "best_matches": copy.deepcopy(self.best_matches),
            "params": params,
        }


# ---------------------------------------------------------------------------
# Parameter checks
# ---------------------------------------------------------------------------


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _check_unit(name: str, value: Any) -> float:
    if not _is_real(value) or not 0 <= value <= 1:
        raise ParameterError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


def _check_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ParameterError(f"{name} must be True or False, got {value!r}")
    return bool(value)


def _check_edge_weights(value: Any) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 1:
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ParameterError(f"edge_weights must be None or a sequence of three numbers, got {value!r}")
    try:
        weights = tuple(_check_unit(f"edge_weights[{i}]", w) for i, w in enumerate(value))
    except ParameterError as e:
        raise ParameterError(f"{e} (edge_weights={value!r})") from None
    if abs(sum(weights) - 1) > 1e-9:
        raise ParameterError(f"edge_weights must sum to 1, got {value!r} (sum {sum(weights)!r})")
    return weights  # type: ignore[return-value]


def _check_redundancy_ratio(value: Any) -> float:
    if not _is_real(value) or not value > 0 or not math.isfinite(value):
        raise ParameterError(f"redundancy_ratio must be a finite number greater than 0, got {value!r}")
    return float(value)


def _check_no_any(schema: Schema) -> None:
    """With allow_any=False, the schema must give every property a concrete type."""
    found = [
        f"{definition.id}.{key}"
        for definition in (*schema.node_types, *schema.edge_types)
        for key, spec in definition.properties.items()
        if spec.type == DataType.ANY
    ]
    if found:
        raise SchemaError(
            "allow_any=False, but these properties have type ANY (or no type): " + ", ".join(found)
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _harmonic(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


@dataclass(frozen=True)
class _Params:
    """The checked parameters of one evaluation."""

    similarity: SimilarityParams
    redundancy_ratio: float
    edge_weights_mode: str


def _check_params(
    instance: Instance,
    schema: Schema,
    *,
    label_weight: float = 0.5,
    edge_weights: Optional[tuple[float, float, float]] = None,
    redundancy_ratio: float = 0.15,
    use_types: bool = True,
    allow_any: bool = True,
) -> _Params:
    """Check the arguments of :func:`evaluate` and decide the edge weights."""
    if not isinstance(instance, Instance):
        raise TypeError(f"instance must be an Instance, not {type(instance).__name__}")
    if not isinstance(schema, Schema):
        raise TypeError(f"schema must be a Schema, not {type(schema).__name__}")
    label_weight = _check_unit("label_weight", label_weight)
    weights = _check_edge_weights(edge_weights)
    redundancy_ratio = _check_redundancy_ratio(redundancy_ratio)
    use_types = _check_bool("use_types", use_types)
    allow_any = _check_bool("allow_any", allow_any)
    if use_types and not allow_any:
        _check_no_any(schema)

    # The automatic weights are decided once, on the original schema.
    mode = "user" if weights is not None else "auto"
    if weights is None:
        weights = WEIGHTS_WITH_CARDINALITY if schema.has_cardinality else WEIGHTS_WITHOUT_CARDINALITY
    return _Params(SimilarityParams(label_weight, weights, use_types, allow_any), redundancy_ratio, mode)


def _result(
    params: _Params,
    cov_v_value: float,
    cov_e_value: float,
    node_matches: Sequence[Match],
    edge_matches: Sequence[Match],
    con: ConcisenessResult,
) -> EvaluationResult:
    """Assemble the public result."""
    coverage = (cov_v_value + cov_e_value) / 2
    concise = (con.con_v + con.con_e) / 2
    return EvaluationResult(
        c2=_harmonic(coverage, concise),
        cov_v=cov_v_value,
        cov_e=cov_e_value,
        con_v=con.con_v,
        con_e=con.con_e,
        coverage=coverage,
        conciseness=concise,
        redundant_types={
            "node_types": sorted(r.type_id for r in con.node_removals if r.redundant),
            "edge_types": sorted(r.type_id for r in con.edge_removals if r.redundant),
        },
        best_matches={
            "node_types": [
                {
                    "labels": sorted(m.instance_type.labels),
                    "schema_type": m.schema_type.id if m.schema_type is not None else None,
                    "similarity": m.similarity,
                }
                for m in node_matches
            ],
            "edge_types": [
                {
                    "labels": sorted(m.instance_type.labels),
                    "source_labels": sorted(m.instance_type.source_labels),
                    "target_labels": sorted(m.instance_type.target_labels),
                    "schema_type": m.schema_type.schema_type if m.schema_type is not None else None,
                    "source": m.schema_type.source.id if m.schema_type is not None else None,
                    "target": m.schema_type.target.id if m.schema_type is not None else None,
                    "similarity": m.similarity,
                }
                for m in edge_matches
            ],
        },
        params={
            "label_weight": params.similarity.label_weight,
            "edge_weights": params.similarity.edge_weights,
            "edge_weights_mode": params.edge_weights_mode,
            "redundancy_ratio": params.redundancy_ratio,
            "use_types": params.similarity.use_types,
            "allow_any": params.similarity.allow_any,
        },
    )


def evaluate(
    instance: Instance,
    schema: Schema,
    *,
    label_weight: float = 0.5,
    edge_weights: Optional[tuple[float, float, float]] = None,
    redundancy_ratio: float = 0.15,
    use_types: bool = True,
    allow_any: bool = True,
) -> EvaluationResult:
    """Evaluate ``schema`` against ``instance`` and return the C2-score.

    ``edge_weights`` defaults to (1/3, 1/3, 1/3) if some edge definition
    writes a cardinality and to (0.5, 0.5, 0) otherwise. ``allow_any`` is
    ignored when ``use_types`` is false.
    """
    params = _check_params(
        instance,
        schema,
        label_weight=label_weight,
        edge_weights=edge_weights,
        redundancy_ratio=redundancy_ratio,
        use_types=use_types,
        allow_any=allow_any,
    )
    similarity = Similarity(instance, params.similarity)
    flat = flatten(schema)
    tables = CoverageTables(flat, similarity)
    cov_v_value, node_matches = tables.cov_v()
    cov_e_value, edge_matches = tables.cov_e()
    con = conciseness(
        schema, flat, cov_v_value, cov_e_value, similarity, params.redundancy_ratio, tables=tables
    )
    return _result(params, cov_v_value, cov_e_value, node_matches, edge_matches, con)
