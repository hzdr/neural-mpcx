"""A module that provides start point generation for warm-starting optimization.

Overview
========

This module provides classes to generate starting points for NLP warm-starting
in a structured or random way:

- :class:`RandomStartPoint` and :class:`RandomStartPoints` allows to generate random
  starting points and group them, respectively
- :class:`StructuredStartPoint` and :class:`StructuredStartPoints` allows to generate
  structured (i.e., deterministic) starting points and group them, respectively.
"""

__all__ = [
    "RandomStartPoint",
    "RandomStartPoints",
    "StructuredStartPoint",
    "StructuredStartPoints",
]

from .startpoints import (
    RandomStartPoint,
    RandomStartPoints,
    StructuredStartPoint,
    StructuredStartPoints,
)
