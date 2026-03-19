# Copyright 2024-2026 Helmholtz-Zentrum Dresden-Rossendorf e.V. (HZDR)
# Authors: 
# - Ênio Lopes Júnior
# - Sebastian Felix Reinecke
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

from typing import Generic, Literal, TypeVar

import casadi as cs

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HasParameters(Generic[SymType]):
    """Class for the creation and storage of symbolic parameters in an NLP problem.

    Parameters
    ----------
    sym_type : {"SX", "MX"}, optional
        The CasADi symbolic variable type to use in the NLP, by default ``"SX"``.
    """

    def __init__(self, sym_type: Literal["SX", "MX"] = "SX") -> None:
        super().__init__()
        self._sym_type: type[SymType] = getattr(cs, sym_type)
        self._pars: dict[str, SymType] = {}
        self._p = self._sym_type()

    @property
    def p(self) -> SymType:
        """Gets the parameters of the NLP scheme."""
        return self._p

    @property
    def np(self) -> int:
        """Number of parameters in the NLP scheme."""
        return self._p.shape[0]

    @property
    def parameters(self) -> dict[str, SymType]:
        """Gets the parameters of the NLP scheme."""
        return self._pars

    def parameter(self, name: str, shape: tuple[int, int] = (1, 1)) -> SymType:
        """Adds a parameter to the NLP scheme.

        Parameters
        ----------
        name : str
            Name of the new parameter. Must not be already in use.
        shape : tuple of 2 ints, optional
            Shape of the new parameter. By default a scalar, i.e., ``(1, 1)``.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new parameter.

        Raises
        ------
        ValueError
            Raises if there is already another parameter with the same name ``name``.
        """
        if name in self._pars:
            raise ValueError(f"Parameter name '{name}' already exists.")
        par = self._sym_type.sym(name, *shape)
        self._pars[name] = par
        self._p = cs.veccat(self._p, par)
        return par
