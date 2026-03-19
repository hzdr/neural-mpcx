"""A module to provide wrappers to enhance the NLP class's capabilities.

Motivation
==========

Inspired by the approach adopted by the `gymnasium <https://gymnasium.farama.org/>`
package, we provide a way to wrap instances of the basic :class:`neuralmpcx.Nlp` class
with wrapper classes that can add desired features.

Overview
========

The basic idea is to create a base class :class:`neuralmpcx.Wrapper` that can be subclassed
to implement the desired features. The base class provides the same interface as
:class:`neuralmpcx.Mpc`, so that the user can interact with the wrapped instance in the same
way as with the basic NLP instance. We also provide a
:class:`neuralmpcx.NonRetroactiveWrapper`, which is a special wrapper that can only wrap
instances of :class:`neuralmpcx.Nlp` before any variable, parameters, etc. is defined.

The following wrappers are provided in this module:

- :class:`neuralmpcx.wrappers.Mpc`: a wrapper that facilities the creation of MPC
  optimization problems :cite:`rawlings_model_2017`
"""

__all__ = [
    "Mpc",
    "NonRetroactiveWrapper",
    "Wrapper",
]

from .mpc.mpc import Mpc
from .wrapper import NonRetroactiveWrapper, Wrapper
