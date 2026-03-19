r"""A module with utility functions and classes around and for optimization.

Overview
========

It contains the following submodules:

- :mod:`neuralmpcx.util.io`: a collection of utilities for input/output operations. The goals
  of these methods are:

   * compatibility of pickling/deepcopying with CasADi objects and classes that hold
     such objects (since these are often not picklable)
   * saving and loading data to/from files, possibly compressed.

- :mod:`neuralmpcx.util.math`: a collection of stand-alone functions that implement some of
  the basic mathematical operations that are not available in CasADi. The
  implementations are simple and thus not optimized for performance. They are meant to
  be used as a fallback when the CasADi ally does not provide the required
  functionality.

Submodules
==========

.. autosummary::
   :toctree: generated
   :template: module.rst

   io
   math
"""

__all__ = ["control", "estimators", "io", "math"]

from . import control, estimators, io, math
