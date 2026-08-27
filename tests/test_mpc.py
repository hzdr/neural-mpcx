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

"""The ``Mpc`` wrapper, in both conventional and neural mode.

The two modes lay out their columns differently: conventional MPC pins an
anchor column ``x[:, 0] == x0`` and spans ``N + 1`` columns, neural MPC spans
``N`` and rolls every one of them forward from the persisted LSTM state. Most
of what follows is about keeping that difference honest, plus the plumbing
around it: input spacing, slacks, disturbances and the solver-failure paths.
"""

import casadi as cs
import numpy as np
import numpy.testing as npt
import pytest

from neuralmpcx import Nlp
from neuralmpcx.wrappers import Mpc


# ===========================================================================
# Part A: Conventional (Non-Neural) MPC
# ===========================================================================


class TestConventionalMpcConstruction:
    """Horizons and dimensions land where the constructor arguments say."""

    def test_horizons_and_dimensions(self, ipopt_opts):
        """MPC wrapper initializes with correct horizons and dimensions."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=1, shooting="multi", neural=False)

        x, x0 = mpc.state("x", 2, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=-1.0, ub=1.0)

        assert mpc.prediction_horizon == 5
        assert mpc.ns == 2
        assert mpc.na == 1


class TestConventionalMpcSolve:
    """A linear plant, solved open loop and around a closed loop."""

    @pytest.fixture()
    def linear_mpc(self, ipopt_opts):
        """Build a minimal linear MPC: x_{k+1} = 0.9*x + 0.1*u, setpoint tracking."""
        pars_init = {
            "SP": np.array([0.0]),
        }
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters=pars_init,
                  n_context=1, shooting="multi", neural=False)

        x, x0 = mpc.state("x", 1, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=-1.0, ub=1.0)
        SP = mpc.parameter("SP", (1,))

        # Linear dynamics: x_{k+1} = 0.9*x_k + 0.1*u_k
        x_sym = cs.MX.sym("x_in", 1)
        u_sym = cs.MX.sym("u_in", 1)
        F = cs.Function("F_linear", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])
        mpc.set_dynamics(F)

        # Quadratic cost: sum of tracking error + control effort
        cost = 0
        for k in range(5):
            e_k = x[:, k] - SP
            cost += e_k.T @ e_k + 0.01 * u_exp[:, k] ** 2
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        return mpc

    def test_solve_succeeds(self, linear_mpc):
        """Conventional MPC solver converges and returns feasible action."""
        state = np.array([5.0])
        state_indices = [0]
        setpoint = np.array([0.0])

        u_opt = linear_mpc.solve_mpc(
            state=state, state_indices=state_indices, setpoint=setpoint,
        )

        u_val = float(u_opt)
        assert -1.0 - 1e-6 <= u_val <= 1.0 + 1e-6, f"Action {u_val} out of bounds"

    def test_multi_step_closed_loop(self, linear_mpc):
        """3 closed-loop steps all succeed with warm-starting."""
        state = np.array([5.0])
        state_indices = [0]
        setpoint = np.array([0.0])

        for _ in range(3):
            u_opt = linear_mpc.solve_mpc(
                state=state, state_indices=state_indices, setpoint=setpoint,
            )
            u_val = float(u_opt)
            assert -1.0 - 1e-6 <= u_val <= 1.0 + 1e-6

            # Simulate step: x_{k+1} = 0.9*x + 0.1*u
            state = np.array([0.9 * float(state[0]) + 0.1 * u_val])

    def test_solution_obeys_the_dynamics_and_the_anchor(self, linear_mpc):
        """Every state column is the plant's image of the previous one.

        Solver success and in-bounds actions both hold for a formulation whose
        multi-shooting equalities were never registered, so this reads the
        solved trajectory back and re-applies ``F`` to it. Column 0 is the
        anchor the neural path deliberately drops.
        """
        x0 = 5.0
        linear_mpc.solve_mpc(
            state=np.array([x0]), state_indices=[0], setpoint=np.array([0.0]),
        )

        sol = linear_mpc._last_solution
        x = np.array(sol.vals["x"]).ravel()
        u = np.array(sol.vals["u"]).ravel()

        # Conventional MPC spans N + 1 columns, anchored on the measurement.
        assert x.size == linear_mpc.prediction_horizon + 1
        npt.assert_allclose(x[0], x0, atol=1e-6)
        npt.assert_allclose(x[1:], 0.9 * x[:-1] + 0.1 * u, atol=1e-6)

    def test_closed_loop_reaches_the_setpoint(self, linear_mpc):
        """Forty steps against the plant drive the state to the setpoint.

        The three-step test above only asserts that each solve succeeded, which
        a controller pushing in the wrong direction also does.
        """
        state = np.array([5.0])

        for _ in range(40):
            u_val = float(linear_mpc.solve_mpc(
                state=state, state_indices=[0], setpoint=np.array([0.0]),
            ))
            assert -1.0 - 1e-6 <= u_val <= 1.0 + 1e-6
            state = np.array([0.9 * float(state[0]) + 0.1 * u_val])

        assert abs(float(state[0])) < 1e-3

    def test_respects_bounds(self, linear_mpc):
        """Action stays within bounds even when large effort is needed."""
        # Large initial state -> would need |u| > 1 without bounds
        state = np.array([100.0])
        state_indices = [0]
        setpoint = np.array([0.0])

        u_opt = linear_mpc.solve_mpc(
            state=state, state_indices=state_indices, setpoint=setpoint,
        )
        u_val = float(u_opt)
        assert -1.0 - 1e-6 <= u_val <= 1.0 + 1e-6


# ===========================================================================
# Part B: Neural MPC
# ===========================================================================


def _build_neural_mpc_cts(cts_state_dict, ipopt_opts, n_context=2, horizon=5):
    """Helper to build a minimal CTS neural MPC."""
    from neuralmpcx.neural import CasadiLSTM

    nx, nu = 1, 1
    hidden_size = 8

    pars_init = {
        "x_lb": np.array(0.0),
        "x_ub": np.array(10.0),
        "x_lb_f": np.array(0.0),
        "x_ub_f": np.array(10.0),
        "b": np.array(-0.5),
        "H_s": np.array(1e3),
        "h_s": np.array(0.0),
        "c_s": np.array(0.0),
        "H_lt": np.array([[1.0, 0], [0, 1e-3]]),
        "h_lt": np.array([0, 0]),
        "c_lt": np.array(0.0),
        "H_0": np.array(0.0),
        "h_0": np.array(0.0),
        "c_0": np.array(0.0),
        "w": np.array(100.0),
        "x_scaling": np.array([0.1], dtype=float),
        "u_scaling": np.array([0.1], dtype=float),
    }

    nlp = Nlp(sym_type="MX")
    mpc = Mpc(nlp, prediction_horizon=horizon, tuning_parameters=pars_init,
              n_context=n_context, shooting="multi", neural=True)

    N = horizon
    gamma = 1.0

    x_lb = mpc.parameter("x_lb", (nx,))
    x_ub = mpc.parameter("x_ub", (nx,))
    x_lb_f = mpc.parameter("x_lb_f", (nx,))
    x_ub_f = mpc.parameter("x_ub_f", (nx,))
    b = mpc.parameter("b")
    H_s = mpc.parameter("H_s", (nx, nx))
    h_s = mpc.parameter("h_s", (nx,))
    c_s = mpc.parameter("c_s")
    H_lt = mpc.parameter("H_lt", (nx + nu, nx + nu))
    h_lt = mpc.parameter("h_lt", (nx + nu,))
    c_lt = mpc.parameter("c_lt")
    H_0 = mpc.parameter("H_0", (nx, nx))
    h_0 = mpc.parameter("h_0", (nx,))
    c_0 = mpc.parameter("c_0")
    w = mpc.parameter("w", (nx, 1))
    x_scaling = mpc.parameter("x_scaling", (nx, 1))
    mpc.parameter("u_scaling", (nu, 1))
    SP = mpc.parameter("SP", (nx, 1))

    x, _ = mpc.state("x", nx, bound_initial=False)
    u, u_exp, u0 = mpc.action("u", nu, lb=0.0, ub=10.0)
    s1, _, _ = mpc.variable("s1", (nx, N), lb=0)
    s2, _, _ = mpc.variable("s2", (nx, 1), lb=0)

    model = CasadiLSTM(
        n_context=n_context, n_inputs=nu, hidden_size=hidden_size,
        horizon=horizon, proj_size=nx,
    )
    model.load_state_dict(cts_state_dict)

    mpc.set_neural_dynamics(
        model=model, output_bias=b,
        name="F_neural",
    )

    # Constraints (following the CTS benchmark pattern)
    xlb_rep = cs.repmat(x_lb, 1, N)
    xub_rep = cs.repmat(x_ub, 1, N)
    hard_indices = [0]
    mpc.constraint("s1_hard", s1[hard_indices, :], "==", 0)
    mpc.constraint("s2_hard", s2[hard_indices, :], "==", 0)
    mpc.constraint("x_lb", xlb_rep * x_scaling - s1, "<=",
                   x[:, :] * x_scaling)
    mpc.constraint("x_ub", x[:, :] * x_scaling, "<=",
                   xub_rep * x_scaling + s1)
    mpc.constraint("x_lb_f", x_lb_f * x_scaling - s2, "<=", x[:, -1] * x_scaling)
    mpc.constraint("x_ub_f", x[:, -1] * x_scaling, "<=", x_ub_f * x_scaling + s2)

    # Cost function
    e_N = x[:, -1] - SP
    e_N = e_N * x_scaling
    S = (gamma ** N) * (0.5 * cs.bilin(H_s, e_N) + h_s.T @ e_N + c_s + w.T @ s2)

    e_0 = x[:, 0] - SP
    e_0 = e_0 * x_scaling
    V0 = 0.5 * cs.bilin(H_0, e_0) + h_0.T @ e_0 + c_0

    Lt = 0.0
    for k in range(0, N):
        e_k = x[:, k] - SP
        e_k = e_k * x_scaling
        k_abs = k - 0
        Lt += (gamma ** k_abs) * (
            0.5 * cs.bilin(H_lt, cs.vertcat(e_k, u_exp[:, k]))
            + h_lt.T @ cs.vertcat(e_k, u_exp[:, k])
            + c_lt
        )
        Lt += (gamma ** k_abs) * (w.T @ s1[:, k_abs])

    nlp.minimize(V0 + S + Lt)
    nlp.init_solver(ipopt_opts)

    return mpc


class TestNeuralMpcConstruction:
    """The neural column layout: N columns, no anchor, x0 and u0 as parameters."""

    def test_state_shapes_no_anchor(self, ipopt_opts):
        """Neural state variable spans exactly the horizon (no anchor column)."""
        pytest.importorskip("torch")
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=2, shooting="multi", neural=True)

        x, x0 = mpc.state("x", 1, bound_initial=False)
        assert x.shape == (1, 5)  # (nx, N) = (1, 5)

        # x0 survives only as a cost parameter, shape (nx, 1)
        x0_sym = mpc.initial_states["x_0"]
        assert x0_sym.shape == (1, 1)

    def test_action_layout_and_first_action(self, ipopt_opts):
        """Action spans N columns; first free action is column 0."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=3, shooting="multi", neural=True)

        u, u_exp, u0 = mpc.action("u", 1)
        assert u.shape == (1, 5)            # (nu, N) = (1, 5)
        assert u0.shape == (1, 1)           # first action parameter
        # first effective action is column 0 (the action applied now)
        first = mpc.first_actions["u"]
        assert first.shape == (1, 1)
        assert str(first) == str(u[:, 0])


class TestNeuralMpcSetDynamics:
    """``set_neural_dynamics`` registers ``F(u, h0, c0)``, with no state input."""

    def test_set_neural_dynamics_cts(self, cts_state_dict, ipopt_opts):
        """set_neural_dynamics succeeds and registers dynamics."""
        pytest.importorskip("torch")
        from neuralmpcx.neural import CasadiLSTM

        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=2, shooting="multi", neural=True)

        x, x0 = mpc.state("x", 1, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=0.0, ub=10.0)

        model = CasadiLSTM(
            n_context=2, n_inputs=1, hidden_size=8,
            horizon=5, proj_size=1,
        )
        model.load_state_dict(cts_state_dict)

        mpc.set_neural_dynamics(
            model=model,
        )

        assert mpc.dynamics is not None
        # F is driven by the controls and the persisted LSTM state only
        assert mpc.dynamics.name_in() == ["u", "h0", "c0"]


class TestNeuralMpcSolve:
    """End-to-end neural solves, and the warmup state machine behind them."""

    def test_solve_cts(self, cts_state_dict, ipopt_opts):
        """Full CTS neural MPC pipeline: load, build, solve."""
        pytest.importorskip("torch")
        mpc = _build_neural_mpc_cts(cts_state_dict, ipopt_opts)

        state = np.array([[5.0]])
        state_context = np.array([[5.0], [5.0]])  # (n_context, nx_full=1)
        state_indices = [0]
        action_context = np.array([[3.0], [3.0]])  # (n_context, nu=1)
        setpoint = np.array([[5.0]])  # (nx, 1)

        u_opt = mpc.solve_mpc(
            state, state_context, state_indices,
            action_context, setpoint,
        )

        assert u_opt.shape == (1, 1)
        u_val = float(u_opt)
        assert 0.0 - 1e-6 <= u_val <= 10.0 + 1e-6, f"Action {u_val} out of bounds"

    def test_solution_matches_the_lstm_rollout(self, cts_state_dict, ipopt_opts):
        """The solved state columns are the LSTM's own rollout of the optimal u.

        The multi-shooting equalities are what tie ``x`` to the network. Without
        them the solver would drive ``x`` straight onto the setpoint and every
        other neural test here would still pass, so this re-rolls ``F`` outside
        the NLP from the same persisted ``(h, c)`` and compares.

        Built without ``output_bias`` on purpose: passing an NLP parameter there
        leaves a free MX inside ``F``, and a function with a free symbol cannot
        be evaluated numerically. That is why ``_build_neural_mpc_cts`` is not
        reused here.
        """
        pytest.importorskip("torch")
        from neuralmpcx.neural import CasadiLSTM

        n_context, horizon = 2, 5
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=horizon,
                  tuning_parameters={"SP": np.array([5.0])},
                  n_context=n_context, shooting="multi", neural=True)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=0.0, ub=10.0)
        SP = mpc.parameter("SP", (1,))

        model = CasadiLSTM(n_context=n_context, n_inputs=1, hidden_size=8,
                           horizon=horizon, proj_size=1)
        model.load_state_dict(cts_state_dict)
        mpc.set_neural_dynamics(model=model)

        nlp.minimize(sum((x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2
                         for k in range(horizon)))
        nlp.init_solver(ipopt_opts)

        mpc.solve_mpc(
            np.array([[5.0]]), np.array([[5.0], [5.0]]), [0],
            np.array([[3.0], [3.0]]), np.array([[5.0]]),
        )

        sol = mpc._last_solution
        assert not mpc.dynamics.has_free()
        rollout = np.array(mpc.dynamics(
            sol.vals["u"], cs.horzcat(*mpc._lstm_h), cs.horzcat(*mpc._lstm_c)
        ))

        x_opt = np.array(sol.vals["x"])
        # Neural MPC has no anchor column: all N columns are predictions.
        assert x_opt.shape[1] == horizon
        npt.assert_allclose(x_opt, rollout, atol=1e-5)

    def test_one_dimensional_action_context_is_refused(
        self, cts_state_dict, ipopt_opts
    ):
        """A flat `(T_ctx,)` action history is rejected by name, not by IndexError.

        The column-selection arithmetic downstream indexes ``shape[1]``, so a
        1-D history used to die inside NumPy with no mention of the argument
        that caused it.
        """
        pytest.importorskip("torch")
        mpc = _build_neural_mpc_cts(cts_state_dict, ipopt_opts)

        with pytest.raises(ValueError, match=r"`action_context` must be 2-D"):
            mpc.solve_mpc(
                np.array([[5.0]]), np.array([[5.0], [5.0]]), [0],
                np.array([3.0, 3.0]), np.array([[5.0]]),
            )

    def test_solve_cstr(self, cstr_state_dict, ipopt_opts):
        """Full CSTR neural MPC: 4 outputs, 2 inputs."""
        pytest.importorskip("torch")
        from neuralmpcx.neural import CasadiLSTM

        nx, nu = 4, 2
        n_context, horizon = 2, 5
        hidden_size = 8

        pars_init = {
            "x_lb": np.array([0.02, 0.02, 0.36, 0.36]),
            "x_ub": np.array([0.39, 0.39, 0.96, 1.00]),
            "x_lb_f": np.array([0.02, 0.02, 0.36, 0.36]),
            "x_ub_f": np.array([0.39, 0.39, 0.96, 1.00]),
            "b": np.array([0, 0, 0, 0], dtype=float),
            "Q": np.diag([1.0, 1.0, 1e-6, 1e-6]),
            "R": np.array([[1, 0], [0, 1e-4]], dtype=float),
            "w": np.array([0, 0, 1e2, 0], dtype=np.float64),
            "x_scaling": np.array([1, 1, 1, 1], dtype=float),
            "u_scaling": np.array([1, 1], dtype=float),
        }

        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=horizon, tuning_parameters=pars_init,
                  n_context=n_context, shooting="multi", neural=True)

        x_lb = mpc.parameter("x_lb", (nx, 1))
        x_ub = mpc.parameter("x_ub", (nx, 1))
        x_lb_f = mpc.parameter("x_lb_f", (nx, 1))
        x_ub_f = mpc.parameter("x_ub_f", (nx, 1))
        b = mpc.parameter("b", (nx, 1))
        Q = mpc.parameter("Q", (nx, nx))
        R = mpc.parameter("R", (nu, nu))
        SP = mpc.parameter("SP", (nx, 1))
        w = mpc.parameter("w", (nx, 1))
        x_scaling = mpc.parameter("x_scaling", (nx, 1))
        u_scaling = mpc.parameter("u_scaling", (nu, 1))

        x, _ = mpc.state("x", nx, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", nu, lb=np.array([[0.0], [0.0]]),
                                   ub=np.array([[1.0], [1.0]]))
        s1, _, _ = mpc.variable("s1", (nx, horizon), lb=0)
        s2, _, _ = mpc.variable("s2", (nx, 1), lb=0)

        model = CasadiLSTM(
            n_context=n_context, n_inputs=nu, hidden_size=hidden_size,
            horizon=horizon, proj_size=nx,
        )
        model.load_state_dict(cstr_state_dict)

        mpc.set_neural_dynamics(
            model=model, output_bias=b,
            name="F_neural",
        )

        # Constraints
        N = horizon
        xlb_rep = cs.repmat(x_lb, 1, N)
        xub_rep = cs.repmat(x_ub, 1, N)
        hard_indices = [0, 1, 3]
        mpc.constraint("s1_hard", s1[hard_indices, :], "==", 0)
        mpc.constraint("s2_hard", s2[hard_indices, :], "==", 0)
        mpc.constraint("x_lb", xlb_rep * x_scaling - s1, "<=",
                       x[:, :] * x_scaling)
        mpc.constraint("x_ub", x[:, :] * x_scaling, "<=",
                       xub_rep * x_scaling + s1)
        mpc.constraint("x_lb_f", x_lb_f * x_scaling - s2, "<=", x[:, -1] * x_scaling)
        mpc.constraint("x_ub_f", x[:, -1] * x_scaling, "<=", x_ub_f * x_scaling + s2)

        # Cost
        gamma = 1.0
        du = []
        du.append(u_exp[:, 0] - u0[:, -1])
        for t in range(1, N):
            du.append((u_exp[:, t] - u_exp[:, t - 1]) * u_scaling)
        du = cs.hcat(du)

        e_N = x[:, -1] - SP
        e_N = e_N * x_scaling
        S = (gamma ** N) * (cs.bilin(Q, e_N) + w.T @ s2)

        Lt = 0.0
        for k in range(0, 0 + N):
            e_k = x[:, k] - SP
            e_k = e_k * x_scaling
            k_abs = k - 0
            Lt += (gamma ** k_abs) * cs.bilin(Q, e_k)
            Lt += (gamma ** k_abs) * cs.bilin(R, du[:, k_abs])
            Lt += (gamma ** k_abs) * (w.T @ s1[:, k_abs])

        nlp.minimize(S + Lt)
        nlp.init_solver(ipopt_opts)

        # Solve with CSTR steady-state-like initial conditions (normalized)
        ss = np.array([[0.039], [0.098], [0.857], [0.857]])  # approx normalized [0.2, 0.5, 120, 120]
        state_context = np.tile(ss.T, (n_context, 1))  # (n_context, 4)
        action_context = np.tile(np.array([[0.5, 0.5]]), (n_context, 1))  # (n_context, 2)
        setpoint = np.array([[0.294], [0.196], [0.714], [0.714]])  # normalized [1.5, 1.0, 100, 100]
        state_indices = [0, 1, 2, 3]

        u_opt = mpc.solve_mpc(
            ss, state_context, state_indices,
            action_context, setpoint,
        )

        assert u_opt.shape == (2, 1), f"Expected (2,1), got {u_opt.shape}"
        u_arr = np.array(u_opt).flatten()
        for i in range(2):
            assert -1e-6 <= u_arr[i] <= 1.0 + 1e-6, f"Action[{i}]={u_arr[i]} out of bounds"

    def test_action_within_bounds(self, cts_state_dict, ipopt_opts):
        """Neural MPC respects action bounds [0, 10]."""
        pytest.importorskip("torch")
        mpc = _build_neural_mpc_cts(cts_state_dict, ipopt_opts)

        # Extreme setpoint to push controller to limits
        state = np.array([[0.0]])
        state_context = np.array([[0.0], [0.0]])
        state_indices = [0]
        action_context = np.array([[0.0], [0.0]])
        setpoint = np.array([[10.0]])

        u_opt = mpc.solve_mpc(
            state, state_context, state_indices,
            action_context, setpoint,
        )
        u_val = float(u_opt)
        assert 0.0 - 1e-6 <= u_val <= 10.0 + 1e-6

    def test_stateful_solve_persists_h_c(self, cts_state_dict, ipopt_opts):
        """Stateful neural MPC: solve populates _lstm_h/_lstm_c and advances them."""
        pytest.importorskip("torch")
        from neuralmpcx.neural import CasadiLSTM

        n_context, horizon = 2, 5
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=horizon, tuning_parameters={"SP": np.array([5.0])},
                  n_context=n_context, shooting="multi", neural=True)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=0.0, ub=10.0)
        SP = mpc.parameter("SP", (1,))

        model = CasadiLSTM(
            n_context=n_context, n_inputs=1, hidden_size=8,
            horizon=horizon, proj_size=1,
        )
        model.load_state_dict(cts_state_dict)
        mpc.set_neural_dynamics(
            model=model,
            n_warmup=2,
        )
        cost = sum((x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2
                   for k in range(0, horizon))
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        state = np.array([[5.0]])
        sc = np.array([[5.0], [5.0]])
        ac = np.array([[3.0], [3.0]])
        sp = np.array([[5.0]])
        state_indices = [0]

        assert mpc._lstm_h is None and mpc._lstm_c is None
        assert not mpc.is_warmed_up

        # 1st solve: warmup (re-estimate from zeros)
        mpc.solve_mpc(state, sc, state_indices, ac, sp)
        assert mpc._lstm_h is not None and mpc._lstm_c is not None
        assert mpc._solve_count == 1
        assert not mpc.is_warmed_up
        h_after_solve1 = np.array(mpc._lstm_h[0]).copy()

        # 2nd solve: still warmup (re-estimate seeded from previous h,c)
        mpc.solve_mpc(state, sc, state_indices, ac, sp)
        assert mpc._solve_count == 2
        assert mpc.is_warmed_up
        h_after_solve2 = np.array(mpc._lstm_h[0]).copy()
        # h should have changed because the seed was different
        assert not np.allclose(h_after_solve2, h_after_solve1)

        # 3rd solve: post-warmup (single incremental step)
        mpc.solve_mpc(state, sc, state_indices, ac, sp)
        assert mpc._solve_count == 3
        assert mpc._lstm_h is not None and mpc._lstm_c is not None

    def test_reset_lstm_state(self, cts_state_dict, ipopt_opts):
        """reset_lstm_state zeros the buffers and forces re-warmup."""
        pytest.importorskip("torch")
        from neuralmpcx.neural import CasadiLSTM

        n_context, horizon = 2, 5
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=horizon, tuning_parameters={"SP": np.array([5.0])},
                  n_context=n_context, shooting="multi", neural=True)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=0.0, ub=10.0)
        SP = mpc.parameter("SP", (1,))

        model = CasadiLSTM(
            n_context=n_context, n_inputs=1, hidden_size=8,
            horizon=horizon, proj_size=1,
        )
        model.load_state_dict(cts_state_dict)
        mpc.set_neural_dynamics(
            model=model,
        )
        nlp.minimize(sum((x[:, k] - SP) ** 2 for k in range(0, horizon)))
        nlp.init_solver(ipopt_opts)

        sc = np.array([[5.0], [5.0]])
        ac = np.array([[3.0], [3.0]])
        sp = np.array([[5.0]])

        mpc.solve_mpc(np.array([[5.0]]), sc, [0], ac, sp)
        mpc.solve_mpc(np.array([[5.0]]), sc, [0], ac, sp)
        assert mpc.is_warmed_up
        assert mpc._solve_count == 2

        mpc.reset_lstm_state()
        assert mpc._lstm_h is None and mpc._lstm_c is None
        assert mpc._solve_count == 0
        assert not mpc.is_warmed_up

    def test_invalid_n_warmup_raises(self, cts_state_dict, ipopt_opts):
        """set_neural_dynamics rejects non-positive n_warmup."""
        pytest.importorskip("torch")
        from neuralmpcx.neural import CasadiLSTM

        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=2, shooting="multi", neural=True)
        mpc.state("x", 1, bound_initial=False)
        mpc.action("u", 1, lb=0.0, ub=10.0)

        model = CasadiLSTM(
            n_context=2, n_inputs=1, hidden_size=8,
            horizon=5, proj_size=1,
        )
        model.load_state_dict(cts_state_dict)

        with pytest.raises(ValueError, match="n_warmup"):
            mpc.set_neural_dynamics(
                model=model,
                n_warmup=0,
            )

    def test_stateful_F_built_once(self, cts_state_dict, ipopt_opts):
        """The dynamics function F is shared across solves (no rebuild)."""
        pytest.importorskip("torch")
        from neuralmpcx.neural import CasadiLSTM

        n_context, horizon = 2, 5
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=horizon, tuning_parameters={"SP": np.array([5.0])},
                  n_context=n_context, shooting="multi", neural=True)
        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=0.0, ub=10.0)
        SP = mpc.parameter("SP", (1,))

        model = CasadiLSTM(
            n_context=n_context, n_inputs=1, hidden_size=8,
            horizon=horizon, proj_size=1,
        )
        model.load_state_dict(cts_state_dict)
        mpc.set_neural_dynamics(
            model=model,
        )
        nlp.minimize(sum((x[:, k] - SP) ** 2 for k in range(0, horizon)))
        nlp.init_solver(ipopt_opts)

        f_id = id(mpc.dynamics)
        sc = np.array([[5.0], [5.0]])
        ac = np.array([[3.0], [3.0]])
        sp = np.array([[5.0]])
        for _ in range(3):
            mpc.solve_mpc(np.array([[5.0]]), sc, [0], ac, sp)
        assert id(mpc.dynamics) == f_id

    def test_context_affects_solution(self, cts_state_dict, ipopt_opts):
        """Different state contexts produce different optimal actions."""
        pytest.importorskip("torch")
        mpc = _build_neural_mpc_cts(cts_state_dict, ipopt_opts)

        state_indices = [0]
        action_context = np.array([[3.0], [3.0]])
        setpoint = np.array([[5.0]])

        # Context 1: low state
        u1 = float(mpc.solve_mpc(
            np.array([[1.0]]),
            np.array([[1.0], [1.0]]),
            state_indices,
            action_context,
            setpoint,
        ))

        # Context 2: high state
        u2 = float(mpc.solve_mpc(
            np.array([[9.0]]),
            np.array([[9.0], [9.0]]),
            state_indices,
            action_context,
            setpoint,
        ))

        assert u1 != u2, "Different contexts should produce different actions"


# ===========================================================================
# Part C: Dynamics Validation
# ===========================================================================


class TestDynamicsGuards:
    """Dynamics can be set once, and each setter rejects the other's argument."""

    def test_set_dynamics_rejects_double_call(self, ipopt_opts):
        """Calling set_dynamics twice raises RuntimeError."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=1, shooting="multi", neural=False)

        mpc.state("x", 1, bound_initial=False)
        mpc.action("u", 1, lb=-1.0, ub=1.0)

        x_sym = cs.MX.sym("x", 1)
        u_sym = cs.MX.sym("u", 1)
        F = cs.Function("F", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])

        mpc.set_dynamics(F)

        with pytest.raises(RuntimeError, match="already set"):
            mpc.set_dynamics(F)

    def test_neural_dynamics_rejects_casadi_function(self, ipopt_opts):
        """Passing cs.Function to set_neural_dynamics raises RuntimeError."""
        pytest.importorskip("torch")

        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=2, shooting="multi", neural=True)

        mpc.state("x", 1, bound_initial=False)
        mpc.action("u", 1, lb=-1.0, ub=1.0)

        x_sym = cs.MX.sym("x", 1, 7)
        u_sym = cs.MX.sym("u", 1, 7)
        F = cs.Function("F", [x_sym, u_sym], [x_sym])

        with pytest.raises(RuntimeError, match="casadi.Function"):
            mpc.set_neural_dynamics(model=F)


# ===========================================================================
# Part D: Input Spacing
# ===========================================================================


class TestInputSpacing:
    """``input_spacing`` blocks the controls into zero-order-hold segments."""

    def test_action_reduces_free_actions(self):
        """With input_spacing=2 and horizon=6, only ceil(6/2)=3 free actions."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=6, tuning_parameters={},
                  n_context=1, input_spacing=2, shooting="multi", neural=False)
        mpc.state("x", 1, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=-1.0, ub=1.0)
        assert u.shape == (1, 3)  # ceil(6/2) = 3 free variables

    def test_expanded_action_has_horizon_length(self):
        """The expanded action spans the full prediction horizon."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=6, tuning_parameters={},
                  n_context=1, input_spacing=2, shooting="multi", neural=False)
        mpc.state("x", 1, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=-1.0, ub=1.0)
        assert u_exp.shape == (1, 6)  # expanded to full horizon

    def test_invalid_spacing_raises(self):
        """Non-positive input_spacing raises ValueError."""
        with pytest.raises(ValueError, match="Input spacing"):
            Mpc(Nlp(sym_type="MX"), prediction_horizon=5, tuning_parameters={},
                n_context=1, input_spacing=0, shooting="multi", neural=False)

    def test_solve_with_input_spacing(self, ipopt_opts):
        """MPC with input_spacing=2 solves and returns feasible action."""
        pars_init = {"SP": np.array([0.0])}
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=6, tuning_parameters=pars_init,
                  n_context=1, input_spacing=2, shooting="multi", neural=False)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=-1.0, ub=1.0)
        SP = mpc.parameter("SP", (1,))

        x_sym = cs.MX.sym("x_sp2", 1)
        u_sym = cs.MX.sym("u_sp2", 1)
        F = cs.Function("F_sp2", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])
        mpc.set_dynamics(F)

        cost = sum((x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(6))
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        u_opt = mpc.solve_mpc(
            np.array([5.0]), np.array([[5.0]]), [0],
            np.array([[0.0]]), np.array([0.0]),
        )
        assert -1.0 - 1e-6 <= float(u_opt) <= 1.0 + 1e-6


# ===========================================================================
# Part E: Single Shooting
# ===========================================================================


class TestSingleShooting:
    """Single shooting: the states are expressions, not decision variables."""

    def test_state_returns_none_before_dynamics(self):
        """Single-shooting state() returns None before set_dynamics."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=1, shooting="single", neural=False)
        x, x0 = mpc.state("x", 1, bound_initial=False)
        assert x is None
        assert x0 is not None

    def test_state_with_bounds_raises(self):
        """Specifying state bounds in single-shooting raises RuntimeError."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=1, shooting="single", neural=False)
        with pytest.raises(RuntimeError, match="single shooting"):
            mpc.state("x", 1, lb=-10.0, ub=10.0)

    def test_conventional_solve(self, ipopt_opts):
        """Single-shooting conventional MPC solves and returns feasible action."""
        pars_init = {"SP": np.array([0.0])}
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters=pars_init,
                  n_context=1, shooting="single", neural=False)

        _, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=-1.0, ub=1.0)
        SP = mpc.parameter("SP", (1,))

        x_sym = cs.MX.sym("x_ss", 1)
        u_sym = cs.MX.sym("u_ss", 1)
        F = cs.Function("F_ss", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])
        mpc.set_dynamics(F)

        # After set_dynamics, states["x"] contains the symbolic trajectory (1, N+1)
        x = mpc.states["x"]
        cost = sum(
            (x[:, k + 1] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(5)
        )
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        u_opt = mpc.solve_mpc(
            np.array([5.0]), np.array([[5.0]]), [0],
            np.array([[0.0]]), np.array([0.0]),
        )
        assert -1.0 - 1e-6 <= float(u_opt) <= 1.0 + 1e-6

    def test_neural_solve(self, cts_state_dict, ipopt_opts):
        """Single-shooting neural MPC builds an (nx, N) trajectory and solves."""
        pytest.importorskip("torch")
        from neuralmpcx.neural import CasadiLSTM

        nx, nu = 1, 1
        n_context, horizon = 2, 5
        hidden_size = 8

        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=horizon, tuning_parameters={},
                  n_context=n_context, shooting="single", neural=True)

        x, _ = mpc.state("x", nx, bound_initial=False)
        assert x is None  # single shooting: trajectory not available yet
        u, u_exp, _ = mpc.action("u", nu, lb=0.0, ub=10.0)
        SP = mpc.parameter("SP", (nx, 1))

        model = CasadiLSTM(
            n_context=n_context, n_inputs=nu, hidden_size=hidden_size,
            horizon=horizon, proj_size=nx,
        )
        model.load_state_dict(cts_state_dict)
        mpc.set_neural_dynamics(model=model, name="F_neural")

        # F has no state input; it is driven by the controls and h0/c0 only.
        assert mpc.dynamics.name_in() == ["u", "h0", "c0"]

        # After set_neural_dynamics the trajectory exists with no anchor: (nx, N)
        x = mpc.states["x"]
        assert x.shape == (nx, horizon)

        cost = sum(
            (x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(horizon)
        )
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        u_opt = mpc.solve_mpc(
            np.array([[5.0]]), np.array([[5.0], [5.0]]), [0],
            np.array([[3.0], [3.0]]), np.array([[5.0]]),
        )
        assert u_opt.shape == (nu, 1)
        assert 0.0 - 1e-6 <= float(u_opt) <= 10.0 + 1e-6


# ===========================================================================
# Part F: Soft Constraints
# ===========================================================================


class TestSoftConstraints:
    """A soft constraint buys feasibility with a penalized slack variable."""

    def test_soft_constraint_creates_slack(self):
        """Soft inequality constraint registers a slack in mpc.slacks."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=3, tuning_parameters={},
                  n_context=1, shooting="multi", neural=False)
        x, _ = mpc.state("x", 1, bound_initial=False)
        mpc.action("u", 1, lb=-2.0, ub=2.0)
        mpc.constraint("x_soft_lb", x[:, 1:], ">=", 0.0, soft=True)
        assert "slack_x_soft_lb" in mpc.slacks

    def test_nslacks_count(self):
        """nslacks reports the sum of first dimensions of registered slack groups."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=3, tuning_parameters={},
                  n_context=1, shooting="multi", neural=False)
        x, _ = mpc.state("x", 1, bound_initial=False)
        mpc.action("u", 1, lb=-2.0, ub=2.0)
        mpc.constraint("x_soft_ub", x[:, 1:], "<=", 100.0, soft=True)
        # slack shape is (1, N) -> shape[0] = 1
        assert mpc.nslacks == 1

    def test_solve_with_soft_constraint(self, ipopt_opts):
        """Soft constraint allows solver to converge even when hard bound would fail."""
        pars_init = {"SP": np.array([0.0])}
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=3, tuning_parameters=pars_init,
                  n_context=1, shooting="multi", neural=False)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=-1.0, ub=1.0)
        SP = mpc.parameter("SP", (1,))

        x_sym = cs.MX.sym("x_soft", 1)
        u_sym = cs.MX.sym("u_soft", 1)
        F = cs.Function("F_soft", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])
        mpc.set_dynamics(F)

        # Soft upper-bound: x <= 0.1 (infeasible from x0=5 with u in [-1,1])
        out = mpc.constraint("x_ub_soft", x[:, 1:], "<=", 0.1, soft=True)
        slack = out[2]

        cost = (
            sum((x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(3))
            + 1000 * cs.sumsqr(slack)
        )
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        u_opt = mpc.solve_mpc(
            np.array([5.0]), np.array([[5.0]]), [0],
            np.array([[0.0]]), np.array([0.0]),
        )
        assert -1.0 - 1e-6 <= float(u_opt) <= 1.0 + 1e-6


# ===========================================================================
# Part G: Disturbances
# ===========================================================================


class TestDisturbances:
    """Measured disturbances on the conventional path: an extra input to F."""

    def test_disturbance_shape(self):
        """disturbance() creates a parameter with shape (size, prediction_horizon)."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=1, shooting="multi", neural=False)
        d = mpc.disturbance("d", 1)
        assert d.shape == (1, 5)

    def test_nd_property(self):
        """nd counts the sum of first dimensions of registered disturbance groups."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=1, shooting="multi", neural=False)
        mpc.disturbance("d", 1)
        assert mpc.nd == 1

    def test_solve_with_disturbance(self, ipopt_opts):
        """A 3-input ``F(x, u, d)`` solves and returns a feasible first action."""
        N = 5
        pars_init = {"SP": np.array([0.0])}
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=N, tuning_parameters=pars_init,
                  n_context=1, shooting="multi", neural=False)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=-1.0, ub=1.0)
        mpc.disturbance("d", 1)
        SP = mpc.parameter("SP", (1,))

        x_sym = cs.MX.sym("x_dist", 1)
        u_sym = cs.MX.sym("u_dist", 1)
        d_sym = cs.MX.sym("d_dist", 1)
        F = cs.Function(
            "F_dist", [x_sym, u_sym, d_sym], [0.9 * x_sym + 0.1 * u_sym + 0.1 * d_sym]
        )
        mpc.set_dynamics(F)

        cost = sum((x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(N))
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        u_opt = mpc.solve_mpc(
            state=np.array([5.0]), state_indices=[0], setpoint=np.array([0.0]),
            dynamic_pars={"d": np.zeros((1, N))},
        )
        assert -1.0 - 1e-6 <= float(u_opt) <= 1.0 + 1e-6

    def test_disturbance_hold_constant(self, ipopt_opts):
        """Conventional MPC: the `disturbance` input holds the latest value constant.

        Equivalent to tiling the latest measured disturbance over the horizon, and
        overridable by an explicit dynamic_pars forecast.
        """
        N = 5
        pars_init = {"SP": np.array([0.0])}
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=N, tuning_parameters=pars_init,
                  n_context=1, shooting="multi", neural=False)

        # Start at the setpoint with wide control bounds so the optimum is
        # interior and sensitive to the disturbance (not saturated at a bound).
        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=-10.0, ub=10.0)
        mpc.disturbance("d", 1)
        SP = mpc.parameter("SP", (1,))

        x_sym = cs.MX.sym("x_dc", 1)
        u_sym = cs.MX.sym("u_dc", 1)
        d_sym = cs.MX.sym("d_dc", 1)
        F = cs.Function(
            "F_dc", [x_sym, u_sym, d_sym], [0.9 * x_sym + 0.1 * u_sym + 0.1 * d_sym]
        )
        mpc.set_dynamics(F)

        cost = sum((x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(N))
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        common = {"state": np.array([0.0]), "state_indices": [0],
                  "setpoint": np.array([0.0])}

        # Hold-constant via the `disturbance` input matches the explicit tiled
        # dynamic_pars trajectory.
        u_ctx = mpc.solve_mpc(**common, disturbance=np.array([[2.0]]))
        u_dp = mpc.solve_mpc(**common, dynamic_pars={"d": np.full((1, N), 2.0)})
        npt.assert_allclose(np.array(u_ctx), np.array(u_dp), atol=1e-6)

        # A different held-constant disturbance changes the optimal first action.
        u_zero = mpc.solve_mpc(**common, disturbance=np.array([[0.0]]))
        assert not np.allclose(np.array(u_ctx), np.array(u_zero))
        assert -10.0 - 1e-6 <= float(u_ctx) <= 10.0 + 1e-6

        # An explicit dynamic_pars forecast overrides the hold-constant default.
        forecast = np.linspace(0.0, 2.0, N).reshape(1, N)
        u_over = mpc.solve_mpc(
            **common, disturbance=np.array([[2.0]]),
            dynamic_pars={"d": forecast},
        )
        u_fc = mpc.solve_mpc(**common, dynamic_pars={"d": forecast})
        npt.assert_allclose(np.array(u_over), np.array(u_fc), atol=1e-6)


def _build_neural_disturbance_mpc(
    make_lstm_state_dict, ipopt_opts, *,
    shooting="multi", allow_disturbances=True,
    horizon=4, n_context=2, seed=0, dist_name="d",
):
    """Minimal neural MPC with a measured-disturbance (feedforward) channel."""
    from neuralmpcx.neural import CasadiLSTM

    nx, nu, nd = 1, 1, 1
    hidden = 8
    pars_init = {"SP": np.array([0.0])}
    nlp = Nlp(sym_type="MX")
    mpc = Mpc(nlp, prediction_horizon=horizon, tuning_parameters=pars_init,
              n_context=n_context, shooting=shooting, neural=True)

    SP = mpc.parameter("SP", (nx, 1))
    mpc.state("x", nx, bound_initial=False)
    u, u_exp, _ = mpc.action("u", nu, lb=-5.0, ub=5.0)
    mpc.disturbance(dist_name, nd)

    model = CasadiLSTM(
        n_context=n_context, n_inputs=nu, hidden_size=hidden,
        horizon=horizon, proj_size=nx, n_disturbances=nd,
    )
    model.load_state_dict(
        make_lstm_state_dict(nu + nd, hidden, proj_size=nx, seed=seed)
    )

    mpc.set_neural_dynamics(
        model=model, allow_disturbances=allow_disturbances, name="F_d"
    )

    # Build the tracking cost after dynamics so single shooting (where the state
    # trajectory is materialized by set_neural_dynamics) works too.
    xt = mpc.states["x"]
    cost = sum((xt[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(horizon))
    nlp.minimize(cost)
    nlp.init_solver(ipopt_opts)
    return mpc


class TestNeuralDisturbances:
    """Neural MPC feedforward: disturbance fed to the LSTM and held constant."""

    def _contexts(self, n_context, nd, d_value=0.0):
        state = np.array([[0.0]])
        state_context = np.zeros((n_context, 1))
        state_indices = [0]
        action_context = np.zeros((n_context, 1))
        setpoint = np.array([[1.0]])
        disturbance_context = np.full((n_context, nd), float(d_value))
        return (state, state_context, state_indices, action_context, setpoint,
                disturbance_context)

    def test_dynamics_exposes_d_input(self, make_lstm_state_dict, ipopt_opts):
        """allow_disturbances wires `d` into the registered dynamics function."""
        pytest.importorskip("torch")
        mpc = _build_neural_disturbance_mpc(make_lstm_state_dict, ipopt_opts)
        assert mpc.dynamics.name_in() == ["u", "h0", "c0", "d"]

    def test_hold_constant_solve(self, make_lstm_state_dict, ipopt_opts):
        """A hold-constant disturbance solve returns a feasible first action."""
        pytest.importorskip("torch")
        mpc = _build_neural_disturbance_mpc(make_lstm_state_dict, ipopt_opts)
        args = self._contexts(mpc._n_context, nd=1, d_value=1.0)
        state, sc, si, ac, sp, dc = args
        u_opt = mpc.solve_mpc(state, sc, si, ac, sp, disturbance_context=dc)
        assert u_opt.shape == (1, 1)
        assert -5.0 - 1e-6 <= float(u_opt) <= 5.0 + 1e-6

    def test_disturbance_changes_action(self, make_lstm_state_dict, ipopt_opts):
        """Different disturbance contexts yield a different optimal first action."""
        pytest.importorskip("torch")
        mpc0 = _build_neural_disturbance_mpc(make_lstm_state_dict, ipopt_opts, seed=1)
        mpc1 = _build_neural_disturbance_mpc(make_lstm_state_dict, ipopt_opts, seed=1)

        s, sc, si, ac, sp, dc0 = self._contexts(mpc0._n_context, nd=1, d_value=0.0)
        _, _, _, _, _, dc1 = self._contexts(mpc1._n_context, nd=1, d_value=2.0)

        u0 = mpc0.solve_mpc(s, sc, si, ac, sp, disturbance_context=dc0)
        u1 = mpc1.solve_mpc(s, sc, si, ac, sp, disturbance_context=dc1)
        assert not np.allclose(np.array(u0), np.array(u1))

    def test_explicit_forecast_override(self, make_lstm_state_dict, ipopt_opts):
        """An explicit dynamic_pars['d'] forecast is accepted and overrides default."""
        pytest.importorskip("torch")
        mpc = _build_neural_disturbance_mpc(make_lstm_state_dict, ipopt_opts)
        N = mpc.prediction_horizon
        s, sc, si, ac, sp, dc = self._contexts(mpc._n_context, nd=1, d_value=1.0)
        forecast = np.linspace(0.0, 1.0, N).reshape(1, N)
        u_opt = mpc.solve_mpc(
            s, sc, si, ac, sp, disturbance_context=dc,
            dynamic_pars={"d": forecast},
        )
        assert -5.0 - 1e-6 <= float(u_opt) <= 5.0 + 1e-6

    def test_missing_disturbance_context_raises(self, make_lstm_state_dict, ipopt_opts):
        """Declaring a disturbance but omitting disturbance_context raises."""
        pytest.importorskip("torch")
        mpc = _build_neural_disturbance_mpc(make_lstm_state_dict, ipopt_opts)
        s, sc, si, ac, sp, _ = self._contexts(mpc._n_context, nd=1)
        with pytest.raises(ValueError, match="disturbance_context"):
            mpc.solve_mpc(s, sc, si, ac, sp)

    def test_missing_context_windows_raise(self, make_lstm_state_dict, ipopt_opts):
        """Neural MPC requires both state_context and action_context."""
        pytest.importorskip("torch")
        mpc = _build_neural_disturbance_mpc(make_lstm_state_dict, ipopt_opts)
        _, sc, si, ac, sp, dc = self._contexts(mpc._n_context, nd=1)
        with pytest.raises(ValueError, match="state_context"):
            mpc.solve_mpc(state_indices=si, action_context=ac, setpoint=sp,
                          disturbance_context=dc)
        with pytest.raises(ValueError, match="action_context"):
            mpc.solve_mpc(state_context=sc, state_indices=si, setpoint=sp,
                          disturbance_context=dc)

    def test_allow_disturbances_flag_required(self, make_lstm_state_dict, ipopt_opts):
        """Declaring a disturbance without allow_disturbances=True raises."""
        pytest.importorskip("torch")
        with pytest.raises(RuntimeError, match="allow_disturbances"):
            _build_neural_disturbance_mpc(
                make_lstm_state_dict, ipopt_opts, allow_disturbances=False
            )

    def test_named_disturbance_parameter_resolves(self, make_lstm_state_dict, ipopt_opts):
        """A disturbance named other than 'd' resolves to its NLP parameter.

        solve_mpc must assign the hold-constant horizon values by the declared
        disturbance name (regression: previously hardcoded as 'd').
        """
        pytest.importorskip("torch")
        mpc = _build_neural_disturbance_mpc(
            make_lstm_state_dict, ipopt_opts, dist_name="Q_dot"
        )
        s, sc, si, ac, sp, dc = self._contexts(mpc._n_context, nd=1, d_value=1.0)
        u_opt = mpc.solve_mpc(s, sc, si, ac, sp, disturbance_context=dc)
        assert -5.0 - 1e-6 <= float(u_opt) <= 5.0 + 1e-6

    def test_single_shooting_parity(self, make_lstm_state_dict, ipopt_opts):
        """Single shooting threads the disturbance and returns a feasible action."""
        pytest.importorskip("torch")
        mpc = _build_neural_disturbance_mpc(
            make_lstm_state_dict, ipopt_opts, shooting="single"
        )
        assert mpc.dynamics.name_in() == ["u", "h0", "c0", "d"]
        s, sc, si, ac, sp, dc = self._contexts(mpc._n_context, nd=1, d_value=1.0)
        u_opt = mpc.solve_mpc(s, sc, si, ac, sp, disturbance_context=dc)
        assert u_opt.shape == (1, 1)
        assert -5.0 - 1e-6 <= float(u_opt) <= 5.0 + 1e-6


# ===========================================================================
# Part H: solve_mpc Edge Cases
# ===========================================================================


class TestSolveMpcEdgeCases:
    """``solve_mpc``'s optional arguments and its two solver-failure branches."""

    @pytest.fixture()
    def simple_mpc(self, ipopt_opts):
        """Build and return a ready-to-solve simple linear MPC."""
        pars_init = {"SP": np.array([0.0])}
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters=pars_init,
                  n_context=1, shooting="multi", neural=False)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=-1.0, ub=1.0)
        SP = mpc.parameter("SP", (1,))

        x_sym = cs.MX.sym("x_ec", 1)
        u_sym = cs.MX.sym("u_ec", 1)
        F = cs.Function("F_ec", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])
        mpc.set_dynamics(F)

        cost = sum((x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(5))
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)
        return mpc

    def test_store_solution_false(self, simple_mpc):
        """store_solution=False prevents updating the internal warmstart cache."""
        simple_mpc.solve_mpc(
            state=np.array([5.0]), state_indices=[0], setpoint=np.array([0.0]),
            store_solution=False,
        )
        assert simple_mpc._last_solution is None

    def test_conventional_requires_state(self, simple_mpc):
        """Conventional MPC raises when `state` is omitted."""
        with pytest.raises(ValueError, match="Conventional MPC requires `state`"):
            simple_mpc.solve_mpc(state_indices=[0], setpoint=np.array([0.0]))

    def test_requires_state_indices(self, simple_mpc):
        """solve_mpc raises when `state_indices` is omitted."""
        with pytest.raises(ValueError, match="state_indices"):
            simple_mpc.solve_mpc(state=np.array([5.0]), setpoint=np.array([0.0]))

    def test_conventional_u0_uses_last_action(self, ipopt_opts):
        """Conventional u0 comes from the previously applied action (Delta-u cost)."""
        N = 5
        pars_init = {"SP": np.array([0.0])}
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=N, tuning_parameters=pars_init,
                  n_context=1, shooting="multi", neural=False)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, u0 = mpc.action("u", 1, lb=-10.0, ub=10.0)
        SP = mpc.parameter("SP", (1,))

        x_sym = cs.MX.sym("x_du", 1)
        u_sym = cs.MX.sym("u_du", 1)
        F = cs.Function("F_du", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])
        mpc.set_dynamics(F)

        # Heavy move-suppression penalty on the first step ties u[:, 0] to u0,
        # which solve_mpc must populate from self._last_action.
        du0 = u_exp[:, 0] - u0
        cost = 1e3 * du0 ** 2 + sum(
            (x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(N)
        )
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        # First solve: no last action yet -> u0 defaults to zeros.
        common = {"state": np.array([5.0]), "state_indices": [0],
                  "setpoint": np.array([0.0])}
        u_first = float(mpc.solve_mpc(**common))
        # The strong Delta-u penalty keeps the first move close to u0 == 0.
        assert abs(u_first) < 0.5
        # Second solve reuses the stored last action as u0.
        u_second = float(mpc.solve_mpc(**common))
        assert abs(u_second - u_first) < 0.5

    def test_use_last_action_on_fail_attribute(self):
        """``use_last_action_on_fail`` lands on the wrapper as given."""
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={},
                  n_context=1, shooting="multi", neural=False)
        mpc.state("x", 1, bound_initial=False)
        mpc.action("u", 1)

        x_sym = cs.MX.sym("x_fail", 1)
        u_sym = cs.MX.sym("u_fail", 1)
        F = cs.Function("F_fail", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])
        mpc.set_dynamics(F, use_last_action_on_fail=True)

        assert mpc._last_action_on_fail is True

    @pytest.mark.parametrize("warmstart", ["last-successful", "last"])
    @pytest.mark.parametrize("use_last_action_on_fail", [True, False])
    def test_failed_solve_fallback_and_storage_policy(
        self, ipopt_opts, warmstart, use_last_action_on_fail
    ):
        """A failed solve picks its return value and its cache policy by flag.

        Two independent branches meet on the failure path. With
        ``use_last_action_on_fail`` the wrapper returns the previously applied
        action instead of the failed solve's first column, which is what keeps
        a control loop from writing garbage to the plant. Separately,
        ``warmstart="last"`` warm-starts from a failed solution while
        ``"last-successful"`` keeps the last good one.
        """
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters={"SP": np.array([0.0])},
                  n_context=1, shooting="multi", neural=False)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=-1.0, ub=1.0)
        SP = mpc.parameter("SP", (1,))

        x_sym = cs.MX.sym("x_fb", 1)
        u_sym = cs.MX.sym("u_fb", 1)
        mpc.set_dynamics(
            cs.Function("F_fb", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym]),
            warmstart=warmstart,
            use_last_action_on_fail=use_last_action_on_fail,
        )
        nlp.minimize(sum((x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2
                         for k in range(5)))
        nlp.init_solver(ipopt_opts)

        call = {"state": np.array([5.0]), "state_indices": [0],
                "setpoint": np.array([0.0])}
        u_good = float(mpc.solve_mpc(**call))
        sol_good = mpc._last_solution
        assert sol_good is not None

        # Starve the solver so the next solve reports failure.
        starved = {**ipopt_opts, "ipopt": {**ipopt_opts["ipopt"], "max_iter": 0}}
        nlp.init_solver(starved)
        u_failed = float(mpc.solve_mpc(**call))

        if use_last_action_on_fail:
            assert u_failed == pytest.approx(u_good)
        else:
            assert u_failed != pytest.approx(u_good)

        replaced = mpc._last_solution is not sol_good
        assert replaced is (warmstart == "last")

    def test_dynamic_pars_override(self, ipopt_opts):
        """dynamic_pars overrides tuning parameters at solve time."""
        pars_init = {"SP": np.array([0.0]), "Q_gain": np.array([1.0])}
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp, prediction_horizon=5, tuning_parameters=pars_init,
                  n_context=1, shooting="multi", neural=False)

        x, _ = mpc.state("x", 1, bound_initial=False)
        u, u_exp, _ = mpc.action("u", 1, lb=-1.0, ub=1.0)
        SP = mpc.parameter("SP", (1,))
        Q_gain = mpc.parameter("Q_gain", (1,))

        x_sym = cs.MX.sym("x_dp", 1)
        u_sym = cs.MX.sym("u_dp", 1)
        F = cs.Function("F_dp", [x_sym, u_sym], [0.9 * x_sym + 0.1 * u_sym])
        mpc.set_dynamics(F)

        cost = sum(
            Q_gain * (x[:, k] - SP) ** 2 + 0.01 * u_exp[:, k] ** 2 for k in range(5)
        )
        nlp.minimize(cost)
        nlp.init_solver(ipopt_opts)

        u_opt = mpc.solve_mpc(
            state=np.array([5.0]), state_indices=[0], setpoint=np.array([0.0]),
            dynamic_pars={"Q_gain": np.array([10.0])},
        )
        assert -1.0 - 1e-6 <= float(u_opt) <= 1.0 + 1e-6
