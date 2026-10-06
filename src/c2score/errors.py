"""Exception hierarchy of c2score."""


class C2ScoreError(Exception):
    """Base class of all errors raised by c2score."""


class SchemaError(C2ScoreError, ValueError):
    """The schema is malformed or violates a validation rule."""


class InstanceLoadError(C2ScoreError):
    """The instance could not be loaded (connection, authentication, query, or version)."""


class ParameterError(C2ScoreError, ValueError):
    """An argument is out of range or the arguments are inconsistent."""
