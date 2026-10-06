"""C2-score: quantitative quality evaluation of property graph schemas."""

from importlib.metadata import version

from .errors import C2ScoreError, InstanceLoadError, ParameterError, SchemaError
from .model import Instance, Schema
from .schema_json import load_schema_json

__version__ = version("c2score")

__all__ = [
    "load_schema_json",
    "Instance",
    "Schema",
    "C2ScoreError",
    "SchemaError",
    "InstanceLoadError",
    "ParameterError",
    "__version__",
]
