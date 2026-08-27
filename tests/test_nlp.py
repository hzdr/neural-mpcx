# Copyright 2024-2026 Ênio Lopes Júnior
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

"""``Nlp``: symbolic construction and the IPOPT solve pipeline.

Every problem here is small enough to solve by hand, so each docstring is the
problem and its answer. A regression in variable stacking, bound masking or
parameter binding moves one of these numbers.
"""

import numpy as np
import numpy.testing as npt

from neuralmpcx import Nlp


def test_unconstrained_quadratic(ipopt_opts):
    """min (x - 3)^2 => x* = 3, f* = 0."""
    nlp = Nlp(sym_type="SX")
    x, _, _ = nlp.variable("x", (1, 1))
    nlp.minimize((x - 3) ** 2)
    nlp.init_solver(ipopt_opts)
    sol = nlp.solve()

    assert sol.success
    npt.assert_allclose(float(sol.vals["x"]), 3.0, atol=1e-6)
    npt.assert_allclose(sol.f, 0.0, atol=1e-8)


def test_constrained_quadratic(ipopt_opts):
    """min (x - 5)^2 s.t. x <= 3 => x* = 3, f* = 4."""
    nlp = Nlp(sym_type="SX")
    x, _, _ = nlp.variable("x", (1, 1), ub=3.0)
    nlp.minimize((x - 5) ** 2)
    nlp.init_solver(ipopt_opts)
    sol = nlp.solve()

    assert sol.success
    npt.assert_allclose(float(sol.vals["x"]), 3.0, atol=1e-5)
    npt.assert_allclose(sol.f, 4.0, atol=1e-5)


def test_equality_constraint(ipopt_opts):
    """min x1^2 + x2^2 s.t. x1 + x2 = 1 => x* = [0.5, 0.5]."""
    nlp = Nlp(sym_type="SX")
    x, _, _ = nlp.variable("x", (2, 1))
    nlp.constraint("eq", x[0] + x[1], "==", 1.0)
    nlp.minimize(x[0] ** 2 + x[1] ** 2)
    nlp.init_solver(ipopt_opts)
    sol = nlp.solve()

    assert sol.success
    x_opt = np.array(sol.vals["x"]).flatten()
    npt.assert_allclose(x_opt, [0.5, 0.5], atol=1e-6)


def test_parameter_substitution(ipopt_opts):
    """min (x - p)^2 with p=7 => x*=7, then p=2 => x*=2."""
    nlp = Nlp(sym_type="SX")
    x, _, _ = nlp.variable("x", (1, 1))
    p = nlp.parameter("p", (1, 1))
    nlp.minimize((x - p) ** 2)
    nlp.init_solver(ipopt_opts)

    sol1 = nlp.solve(pars={"p": 7.0})
    assert sol1.success
    npt.assert_allclose(float(sol1.vals["x"]), 7.0, atol=1e-6)

    sol2 = nlp.solve(pars={"p": 2.0})
    assert sol2.success
    npt.assert_allclose(float(sol2.vals["x"]), 2.0, atol=1e-6)


def test_bounded_variable(ipopt_opts):
    """min (x - 15)^2 with x in [0, 10] => x* = 10."""
    nlp = Nlp(sym_type="SX")
    x, _, _ = nlp.variable("x", (1, 1), lb=0.0, ub=10.0)
    nlp.minimize((x - 15) ** 2)
    nlp.init_solver(ipopt_opts)
    sol = nlp.solve()

    assert sol.success
    npt.assert_allclose(float(sol.vals["x"]), 10.0, atol=1e-6)


def test_solution_value_method(ipopt_opts):
    """sol.value(expr) evaluates symbolic expressions at the solution."""
    nlp = Nlp(sym_type="SX")
    x, _, _ = nlp.variable("x", (1, 1))
    p = nlp.parameter("p", (1, 1))
    nlp.minimize((x - p) ** 2)
    nlp.init_solver(ipopt_opts)
    sol = nlp.solve(pars={"p": 4.0})

    assert sol.success
    val = float(sol.value(x + p))
    npt.assert_allclose(val, 8.0, atol=1e-6)  # x*=4, p=4, sum=8


def test_multivariable_2d(ipopt_opts):
    """min x[0]^2 + x[1]^2 s.t. x[0] + x[1] >= 1 => x* = [0.5, 0.5]."""
    nlp = Nlp(sym_type="SX")
    x, _, _ = nlp.variable("x", (2, 1))
    nlp.constraint("ineq", x[0] + x[1], ">=", 1.0)
    nlp.minimize(x[0] ** 2 + x[1] ** 2)
    nlp.init_solver(ipopt_opts)
    sol = nlp.solve()

    assert sol.success
    x_opt = np.array(sol.vals["x"]).flatten()
    npt.assert_allclose(x_opt, [0.5, 0.5], atol=1e-5)


def test_mx_sym_type(ipopt_opts):
    """Same unconstrained quadratic but with MX (used by neural MPC)."""
    nlp = Nlp(sym_type="MX")
    x, _, _ = nlp.variable("x", (1, 1))
    nlp.minimize((x - 3) ** 2)
    nlp.init_solver(ipopt_opts)
    sol = nlp.solve()

    assert sol.success
    npt.assert_allclose(float(sol.vals["x"]), 3.0, atol=1e-6)
