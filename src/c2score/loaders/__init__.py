"""Loaders that build Abs(G) from a stored property graph."""

from .neo4j import load_neo4j

__all__ = ["load_neo4j"]
