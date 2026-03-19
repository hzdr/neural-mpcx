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

from typing import Literal, TypeVar

import casadi as cs

from .parameters import HasParameters

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HasVariables(HasParameters[SymType]):
    """Class for the creation and storage of symbolic variables in an NLP problem. It
    builds on top of :class:`HasParameters`, which handles parameters.

    Parameters
    ----------
    sym_type : {"SX", "MX"}, optional
        The CasADi symbolic variable type to use in the NLP, by default ``"SX"``.
    """

    def __init__(self, sym_type: Literal["SX", "MX"] = "SX") -> None:
        super().__init__(sym_type)
        self._vars: dict[str, SymType] = {}
        self._x = self._sym_type()

    @property
    def x(self) -> SymType:
        """Gets the primary variables of the NLP scheme in vector form."""
        return self._x

    @property
    def nx(self) -> int:
        """Number of variables in the NLP scheme."""
        return self._x.shape[0]

    @property
    def variables(self) -> dict[str, SymType]:
        """Gets the primal variables of the NLP scheme."""
        return self._vars

    def variable(self, name: str, shape: tuple[int, int] = (1, 1)) -> SymType:
        """Adds a variable to the NLP problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new parameter. By default a scalar, i.e., ``(1, 1)``.

        Returns
        -------
        var : casadi.SX
            The symbol of the new variable.

        Raises
        ------
        ValueError
            Raises if there is already another variable with the same name ``name``.
        """
        if name in self._vars:
            raise ValueError(f"Variable name '{name}' already exists.")
        var = self._sym_type.sym(name, *shape)
        self._vars[name] = var
        self._x = cs.veccat(self._x, var)
        return var
