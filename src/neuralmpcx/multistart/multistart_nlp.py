# Copyright 2024-2026 Helmholtz-Zentrum Dresden-Rossendorf e.V. (HZDR)
# Author: Ênio Lopes Júnior
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multistart NLP: solve the same problem from several initial guesses and
keep the best solution.

:class:`MultistartNlp` is a drop-in :class:`neuralmpcx.Nlp` whose ``solve``
also accepts an *iterable* of initial-guess dicts (as produced by
:meth:`neuralmpcx.core.warmstart.WarmStartStrategy.generate`). Each candidate
is solved serially with the same parameters; the returned solution is the
best one — successful solutions are preferred over failed ones, ties broken
by the lower objective. This is the consumer of the
``structured_points``/``random_points`` machinery in
:mod:`neuralmpcx.multistart.startpoints`.

The :class:`neuralmpcx.wrappers.Mpc` wrapper engages it automatically: when
its NLP ``is_multi``, ``solve_mpc`` expands the warm start into
``starts``-many candidates via the registered ``WarmStartStrategy``.
"""

from collections.abc import Iterable
from typing import Any, ClassVar, Optional, TypeVar, Union

import casadi as cs
import numpy.typing as npt

from ..core.solutions import Solution
from ..nlps.nlp import Nlp

SymType = TypeVar("SymType", cs.SX, cs.MX)


class MultistartNlp(Nlp[SymType]):
    """An :class:`Nlp` solved from multiple starting points (serial best-of-N).

    Parameters
    ----------
    starts : int
        The number of starting points the solver should expect per solve
        (typically ``WarmStartStrategy.n_points + 1`` so the plain warm start
        itself is included).
    args, kwargs
        Forwarded to :class:`neuralmpcx.Nlp`.
    """

    is_multi: ClassVar[bool] = True

    def __init__(self, *args: Any, starts: int = 1, **kwargs: Any) -> None:
        if starts < 1:
            raise ValueError("starts must be >= 1")
        super().__init__(*args, **kwargs)
        self._starts = int(starts)

    @property
    def starts(self) -> int:
        """Number of starting points per solve."""
        return self._starts

    def solve_multi(
        self,
        pars: Optional[dict[str, npt.ArrayLike]] = None,
        vals0: Union[
            None,
            dict[str, npt.ArrayLike],
            Iterable[Optional[dict[str, npt.ArrayLike]]],
        ] = None,
    ) -> Solution[SymType]:
        """Solves the NLP from every candidate in ``vals0``, returning the
        best solution (success first, then lowest objective)."""
        if vals0 is None or isinstance(vals0, dict):
            return super().solve(pars, vals0)
        best: Optional[Solution[SymType]] = None
        for candidate in vals0:
            sol = super().solve(pars, candidate)
            if best is None or (sol.success, -sol.f) > (best.success, -best.f):
                best = sol
        if best is None:  # empty iterable: fall back to the plain solve
            return super().solve(pars, None)
        return best

    def solve(
        self,
        pars: Optional[dict[str, npt.ArrayLike]] = None,
        vals0: Union[
            None,
            dict[str, npt.ArrayLike],
            Iterable[Optional[dict[str, npt.ArrayLike]]],
        ] = None,
    ) -> Solution[SymType]:
        return self.solve_multi(pars, vals0)
