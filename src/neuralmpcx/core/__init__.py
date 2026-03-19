r"""This module contains the core components, aside :class:`neuralmpcx.Nlp` and its base
classes, that are used to build the package.

Overview
========

It contains the following submodules:

- :mod:`neuralmpcx.core.cache`: a collection of methods to handle caching in the package. In
  particular, it offers a decorator :func:`invalidate_cache` that allows to invalidate
  the cache of a given set of other cached properties or methods when the decorated
  method is  invoked, as well as a function :func:`invalidate_caches_of` that allows to
  invalidate the cache of a given object on the fly.
- :mod:`neuralmpcx.core.data`: a collection of functions for manipulating data in CasADi, in
  particular, on how to convert to and from numpy arrays and CasADi symbolic variables,
  and how to find the indices of a symbolic variable in a vector of symbols.
- :mod:`neuralmpcx.core.debug`: contains classes for storing debug information on the
  parameters, variables and constraints in an instance of the :class:`neuralmpcx.Nlp` class.
- :mod:`neuralmpcx.core.solutions`: contains classes and methods to store the solution of an
  NLP problem after a call to :meth:`neuralmpcx.Nlp.solve`.

Submodules
==========

.. autosummary::
   :toctree: generated
   :template: module.rst

   cache
   data
   debug
   solutions
"""
