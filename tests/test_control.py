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

"""Transfer-function assembly, discretization and LQR in ``neuralmpcx.util.control``.

Walks the four private stages ``mimo_tf2ss`` composes, then pins the assembled
realization against the analytic step response of the transfer function it was
built from.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neuralmpcx.util.control import (
    DiscreteStateSpace,
    TransferFunctionTerm,
    _assemble_mimo_ss,
    _pade_delay_poly,
    _poly_mul_desc,
    _term_to_tf,
    dlqr,
    mimo_tf2ss,
)


class TestPolynomialUtilities:
    """Descending-order polynomial arithmetic and the Pade delay factor."""

    def test_poly_mul_desc_simple(self):
        """(s + 1)(s + 2) = s^2 + 3s + 2."""
        # (s + 1) * (s + 2) = s^2 + 3s + 2
        p = np.array([1.0, 1.0])  # s + 1
        q = np.array([1.0, 2.0])  # s + 2
        result = _poly_mul_desc(p, q)
        expected = np.array([1.0, 3.0, 2.0])
        assert_allclose(result, expected)

    def test_poly_mul_desc_with_constant(self):
        """A degree-zero operand scales the coefficients and leaves the degree alone."""
        p = np.array([2.0, 1.0])  # 2s + 1
        q = np.array([3.0])  # 3
        result = _poly_mul_desc(p, q)
        expected = np.array([6.0, 3.0])
        assert_allclose(result, expected)

    def test_pade_delay_zero(self):
        """Zero delay collapses the Pade factor to 1/1."""
        num, den = _pade_delay_poly(0.0, 2)
        assert_allclose(num, [1.0])
        assert_allclose(den, [1.0])

    def test_pade_delay_nonzero(self):
        """Order 2 gives degree-2 numerator and denominator with unit DC gain."""
        num, den = _pade_delay_poly(10.0, 2)
        # Padé order 2 should give degree 2 polynomials
        assert len(num) == 3
        assert len(den) == 3
        # At s=0, e^0 = 1, so num(0)/den(0) should be 1
        assert_allclose(num[-1] / den[-1], 1.0)


class TestTransferFunctionTerm:
    """The term dataclass: field defaults and full construction."""

    def test_term_default_values(self):
        """An unspecified term is a pure gain: no delay, no poles, no integrator."""
        term = TransferFunctionTerm(gain=1.0)
        assert term.gain == 1.0
        assert term.delay == 0.0
        assert term.time_constants == []
        assert term.second_order_factors == []
        assert term.has_integrator is False

    def test_term_with_all_fields(self):
        """Every field round-trips through the constructor."""
        term = TransferFunctionTerm(
            gain=2.5,
            delay=10.0,
            time_constants=[5.0, 10.0],
            second_order_factors=[(0.5, 2.0)],
            has_integrator=True,
        )
        assert term.gain == 2.5
        assert term.delay == 10.0
        assert term.time_constants == [5.0, 10.0]
        assert term.second_order_factors == [(0.5, 2.0)]
        assert term.has_integrator is True


class TestTermToTf:
    """``_term_to_tf`` expands one term into descending num/den coefficients."""

    def test_simple_first_order(self):
        """K/(tau*s + 1) becomes num [K], den [tau, 1]."""
        # G(s) = K / (tau*s + 1)
        term = TransferFunctionTerm(gain=2.0, time_constants=[5.0])
        num, den = _term_to_tf(term, pade_order=2)
        # num = [2.0], den = [5.0, 1.0]
        assert_allclose(num, [2.0])
        assert_allclose(den, [5.0, 1.0])

    def test_first_order_with_delay(self):
        """A delay multiplies in the Pade factor and raises both degrees."""
        term = TransferFunctionTerm(gain=1.0, delay=5.0, time_constants=[10.0])
        num, den = _term_to_tf(term, pade_order=2)
        # Should have higher order due to Padé
        assert len(num) == 3  # Padé order 2 numerator (degree 2 -> 3 coeffs)
        # Denominator: (tau*s + 1) * Padé_den = degree 1 + degree 2 = degree 3 -> 4 coeffs
        assert len(den) == 4

    def test_integrator(self):
        """``has_integrator`` puts a free s in the denominator."""
        term = TransferFunctionTerm(gain=0.5, has_integrator=True)
        num, den = _term_to_tf(term, pade_order=2)
        # G(s) = 0.5/s -> num=[0.5], den=[1, 0]
        assert_allclose(num, [0.5])
        assert_allclose(den, [1.0, 0.0])

    def test_second_order(self):
        """A (zeta, wn) pair expands to s^2 + 2*zeta*wn*s + wn^2."""
        # G(s) = 1 / (s^2 + 2*zeta*wn*s + wn^2)
        # with zeta=0.5, wn=2: s^2 + 2*s + 4
        term = TransferFunctionTerm(gain=1.0, second_order_factors=[(0.5, 2.0)])
        num, den = _term_to_tf(term, pade_order=2)
        assert_allclose(num, [1.0])
        assert_allclose(den, [1.0, 2.0, 4.0])


class TestAssembleMimoSS:
    """``_assemble_mimo_ss`` stacks the per-entry realizations into one MIMO block."""

    def test_siso_assembly(self):
        """A single entry gives square A, with B, C and D sized to one channel."""
        G = {(0, 0): [TransferFunctionTerm(gain=1.0, time_constants=[1.0])]}
        A, B, C, D = _assemble_mimo_ss(G, ny=1, nu=1, pade_order=2)
        assert A.shape[0] == A.shape[1]  # Square A
        assert B.shape == (A.shape[0], 1)
        assert C.shape == (1, A.shape[0])
        assert D.shape == (1, 1)

    def test_mimo_2x2_assembly(self):
        """Four first-order entries contribute one state each."""
        G = {
            (0, 0): [TransferFunctionTerm(gain=1.0, time_constants=[1.0])],
            (0, 1): [TransferFunctionTerm(gain=2.0, time_constants=[2.0])],
            (1, 0): [TransferFunctionTerm(gain=3.0, time_constants=[3.0])],
            (1, 1): [TransferFunctionTerm(gain=4.0, time_constants=[4.0])],
        }
        A, B, C, D = _assemble_mimo_ss(G, ny=2, nu=2, pade_order=2)
        # 4 first-order systems -> 4 states total
        assert A.shape == (4, 4)
        assert B.shape == (4, 2)
        assert C.shape == (2, 4)
        assert D.shape == (2, 2)

    def test_sparse_mimo(self):
        """Missing entries in G cost no states."""
        # Only G11 and G22 are non-zero
        G = {
            (0, 0): [TransferFunctionTerm(gain=1.0, time_constants=[1.0])],
            (1, 1): [TransferFunctionTerm(gain=2.0, time_constants=[2.0])],
        }
        A, B, C, D = _assemble_mimo_ss(G, ny=2, nu=2, pade_order=2)
        # 2 first-order systems -> 2 states
        assert A.shape == (2, 2)
        assert B.shape == (2, 2)
        assert C.shape == (2, 2)


class TestMimoTf2ss:
    """The public entry point: assemble, discretize, return a ``DiscreteStateSpace``."""

    def test_siso_discretization(self):
        """A SISO system returns a ``DiscreteStateSpace`` with consistent block shapes."""
        G = {(0, 0): [TransferFunctionTerm(gain=1.0, time_constants=[1.0])]}
        ss = mimo_tf2ss(G, ny=1, nu=1, Ts=0.1, pade_order=2)

        assert isinstance(ss, DiscreteStateSpace)
        assert ss.ny == 1
        assert ss.nu == 1
        assert ss.Ts == 0.1
        assert ss.Ad.shape == (ss.nx, ss.nx)
        assert ss.Bd.shape == (ss.nx, 1)
        assert ss.Cd.shape == (1, ss.nx)
        assert ss.Dd.shape == (1, 1)

    def test_mimo_4x4_grinding_circuit_shape(self):
        """The full 4x4 grinding circuit discretizes to consistent block shapes.

        Its 16 entries carry delays, repeated second-order pairs and integrators
        together, which is the combination the assembly stage has to survive.
        """
        # Simplified version of grinding circuit (just shapes)
        TF = TransferFunctionTerm
        G = {
            (0, 0): [TF(gain=-0.58, delay=41.0, time_constants=[83.0])],
            (0, 1): [
                TF(gain=0.97, delay=40.0, time_constants=[125.0, 195.0]),
                TF(gain=-0.97 * 1.08, delay=272.0, time_constants=[125.0, 195.0]),
            ],
            (0, 2): [TF(gain=0.67, delay=8.0, time_constants=[20.0, 92.0])],
            (0, 3): [TF(gain=0.50, delay=2.0, time_constants=[18.0])],
            (1, 0): [TF(gain=0.62, time_constants=[123.0])],
            (1, 1): [TF(gain=-1.75, time_constants=[118.0])],
            (1, 2): [TF(gain=0.51, delay=87.0, time_constants=[81.0, 182.0])],
            (1, 3): [TF(gain=0.64, delay=9.0, time_constants=[137.0])],
            (2, 0): [TF(gain=2.61, delay=45.0, time_constants=[110.0])],
            (2, 1): [TF(gain=9.52, delay=93.0, time_constants=[98.0, 137.0])],
            (2, 2): [TF(gain=2.83, delay=8.0, time_constants=[128.0])],
            (2, 3): [TF(gain=2.81, delay=5.0, time_constants=[108.0])],
            (3, 0): [
                TF(gain=0.001, delay=30.0, time_constants=[150.0], has_integrator=True)
            ],
            (3, 1): [
                TF(gain=0.011, delay=30.0, time_constants=[100.0], has_integrator=True)
            ],
            (3, 2): [TF(gain=0.032, has_integrator=True)],
            (3, 3): [TF(gain=-0.031, has_integrator=True)],
        }

        ss = mimo_tf2ss(G, ny=4, nu=4, Ts=30.0, pade_order=2, balanced=False)

        assert ss.ny == 4
        assert ss.nu == 4
        assert ss.Ts == 30.0
        assert ss.pade_order == 2
        # Check that we have a reasonable number of states
        # (depends on Padé order and system complexity)
        assert ss.nx > 0
        assert ss.Ad.shape == (ss.nx, ss.nx)
        assert ss.Bd.shape == (ss.nx, 4)
        assert ss.Cd.shape == (4, ss.nx)
        assert ss.Dd.shape == (4, 4)

    def test_continuous_matrices_stored(self):
        """``store_continuous=True`` keeps the pre-discretization A, B, C, D."""
        G = {(0, 0): [TransferFunctionTerm(gain=1.0, time_constants=[1.0])]}
        ss = mimo_tf2ss(G, ny=1, nu=1, Ts=0.1, store_continuous=True)
        assert ss.A is not None
        assert ss.B is not None
        assert ss.C is not None
        assert ss.D is not None

    def test_continuous_matrices_not_stored(self):
        """``store_continuous=False`` leaves them ``None``."""
        G = {(0, 0): [TransferFunctionTerm(gain=1.0, time_constants=[1.0])]}
        ss = mimo_tf2ss(G, ny=1, nu=1, Ts=0.1, store_continuous=False)
        assert ss.A is None
        assert ss.B is None
        assert ss.C is None
        assert ss.D is None

    def test_empty_system(self):
        """A pure gain still realizes, with its feedthrough block sized (ny, nu)."""
        # A constant gain has no dynamics
        G = {(0, 0): [TransferFunctionTerm(gain=5.0)]}
        ss = mimo_tf2ss(G, ny=1, nu=1, Ts=0.1, balanced=False)
        # Note: tf2ss will still create a minimal realization
        assert ss.Dd.shape == (1, 1)

    def test_second_order_system(self):
        """A single second-order factor realizes in two states."""
        # Underdamped system: zeta=0.3, wn=1
        G = {
            (0, 0): [TransferFunctionTerm(gain=1.0, second_order_factors=[(0.3, 1.0)])]
        }
        ss = mimo_tf2ss(G, ny=1, nu=1, Ts=0.1)
        # Second-order system has 2 states
        assert ss.nx == 2

    def test_different_discretization_methods(self):
        """Both ``zoh`` and ``tustin`` produce a realization."""
        G = {(0, 0): [TransferFunctionTerm(gain=1.0, time_constants=[1.0])]}

        for method in ["zoh", "tustin"]:
            ss = mimo_tf2ss(G, ny=1, nu=1, Ts=0.1, method=method)
            assert ss.nx > 0

    def test_step_response_matches_the_transfer_function(self):
        """The realization reproduces the step response of the term it came from.

        Every other test in this class asserts block shapes, which a wrong
        realization passes. Compared against the analytic first-order step
        ``K(1 - exp(-(t - theta)/tau))``, sampled past the delay and one time
        constant so the Pade ripple has died out.
        """
        from scipy.signal import dlsim

        K, tau, theta, Ts = 2.0, 10.0, 6.0, 1.0
        G = {(0, 0): [TransferFunctionTerm(gain=K, delay=theta, time_constants=[tau])]}
        ss = mimo_tf2ss(G, ny=1, nu=1, Ts=Ts, pade_order=4, balanced=False)

        n_steps = 200
        t, y, _ = dlsim(
            (ss.Ad, ss.Bd, ss.Cd, ss.Dd, Ts),
            np.ones((n_steps, 1)),
            x0=np.zeros(ss.nx),
        )
        t, y = t.ravel(), y.ravel()
        analytic = np.where(t >= theta, K * (1.0 - np.exp(-(t - theta) / tau)), 0.0)

        settled = t >= theta + 2 * tau
        assert_allclose(y[settled], analytic[settled], atol=1e-4)
        # Once settled the response is the term's gain.
        assert_allclose(y[-1], K, atol=1e-6)


class TestBalancedRealization:
    """``balanced=True`` rescales the realization without changing its I/O."""

    def test_balanced_stable_same_response(self):
        """Balanced and unbalanced realizations give the same step response."""
        from scipy.signal import dlsim

        G = {
            (0, 0): [TransferFunctionTerm(gain=-0.58, delay=41.0, time_constants=[83.0])],
            (0, 1): [TransferFunctionTerm(gain=0.97, delay=40.0, time_constants=[125.0, 195.0])],
            (1, 0): [TransferFunctionTerm(gain=0.62, time_constants=[123.0])],
            (1, 1): [TransferFunctionTerm(gain=-1.75, time_constants=[118.0])],
        }
        ss_bal = mimo_tf2ss(G, ny=2, nu=2, Ts=30.0, balanced=True)
        ss_nobal = mimo_tf2ss(G, ny=2, nu=2, Ts=30.0, balanced=False)

        n_steps = 50
        u = np.ones((n_steps, 2))
        x0_bal = np.zeros(ss_bal.nx)
        x0_nobal = np.zeros(ss_nobal.nx)

        _, y_bal, _ = dlsim(
            (ss_bal.Ad, ss_bal.Bd, ss_bal.Cd, ss_bal.Dd, ss_bal.Ts), u, x0=x0_bal
        )
        _, y_nobal, _ = dlsim(
            (ss_nobal.Ad, ss_nobal.Bd, ss_nobal.Cd, ss_nobal.Dd, ss_nobal.Ts),
            u,
            x0=x0_nobal,
        )
        assert_allclose(y_bal, y_nobal, rtol=1e-6, atol=1e-8)

    def test_balanced_improves_condition_number(self):
        """Balancing does not worsen the condition number of Ad."""
        G = {
            (0, 0): [TransferFunctionTerm(gain=-0.58, delay=41.0, time_constants=[83.0])],
            (1, 0): [TransferFunctionTerm(gain=0.62, time_constants=[123.0])],
            (0, 1): [TransferFunctionTerm(gain=-1.75, time_constants=[118.0])],
            (1, 1): [TransferFunctionTerm(gain=0.97, delay=40.0, time_constants=[125.0, 195.0])],
        }
        ss_bal = mimo_tf2ss(G, ny=2, nu=2, Ts=30.0, balanced=True)
        ss_nobal = mimo_tf2ss(G, ny=2, nu=2, Ts=30.0, balanced=False)
        assert np.linalg.cond(ss_bal.Ad) <= np.linalg.cond(ss_nobal.Ad) + 1.0

    def test_balanced_with_integrator_warns(self):
        """An integrator warns about marginally stable modes and still realizes."""
        G = {
            (0, 0): [TransferFunctionTerm(gain=0.001, delay=30.0, time_constants=[150.0], has_integrator=True)],
        }
        with pytest.warns(UserWarning, match="marginally stable or unstable modes"):
            ss = mimo_tf2ss(G, ny=1, nu=1, Ts=30.0, balanced=True)
        assert isinstance(ss, DiscreteStateSpace)
        assert ss.nx > 0

    def test_balanced_false_no_warning(self):
        """``balanced=False`` stays silent, integrators included."""
        G = {
            (0, 0): [TransferFunctionTerm(gain=0.001, has_integrator=True)],
        }
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            ss = mimo_tf2ss(G, ny=1, nu=1, Ts=30.0, balanced=False)
        assert isinstance(ss, DiscreteStateSpace)


class TestDlqr:
    """``dlqr`` returns the stabilizing gain and its cost-to-go."""

    def test_dlqr_simple(self):
        """K and P come back at the shapes the stage cost implies, with P positive definite."""
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.0], [0.1]])
        Q = np.eye(2)
        R = np.array([[1.0]])

        K, P = dlqr(A, B, Q, R)

        assert K.shape == (1, 2)
        assert P.shape == (2, 2)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0)

    def test_solves_the_riccati_equation_and_stabilizes(self):
        """P satisfies the discrete ARE, K is its gain, and A - BK is stable.

        Shapes and positive definiteness pass for any symmetric matrix, so
        without this the Riccati iteration could return the wrong fixed point
        and nothing would notice.
        """
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.0], [0.1]])
        Q = np.eye(2)
        R = np.array([[1.0]])

        K, P = dlqr(A, B, Q, R)

        residual = (
            A.T @ P @ A
            - (A.T @ P @ B) @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            + Q
            - P
        )
        assert_allclose(residual, np.zeros((2, 2)), atol=1e-9)

        assert_allclose(K, np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A), atol=1e-12)
        assert np.abs(np.linalg.eigvals(A - B @ K)).max() < 1.0


class TestIntegration:
    """A ``mimo_tf2ss`` realization drops straight into a CasADi graph."""

    def test_mimo_to_casadi_compatible(self):
        """Ad and Bd wrap into ``cs.DM`` and evaluate inside a ``cs.Function``."""
        try:
            import casadi as cs
        except ImportError:
            pytest.skip("CasADi not available")

        G = {
            (0, 0): [TransferFunctionTerm(gain=1.0, time_constants=[1.0])],
            (0, 1): [TransferFunctionTerm(gain=0.5, time_constants=[2.0])],
            (1, 0): [TransferFunctionTerm(gain=0.3, time_constants=[1.5])],
            (1, 1): [TransferFunctionTerm(gain=2.0, time_constants=[3.0])],
        }
        ss = mimo_tf2ss(G, ny=2, nu=2, Ts=0.1)

        # Convert to CasADi
        Ad_cs = cs.DM(ss.Ad)
        Bd_cs = cs.DM(ss.Bd)

        # Create symbolic state and input
        x = cs.MX.sym("x", ss.nx)
        u = cs.MX.sym("u", ss.nu)

        # Dynamics should work
        x_next = Ad_cs @ x + Bd_cs @ u
        F = cs.Function("F", [x, u], [x_next])

        # Test with numerical values
        x_val = np.zeros(ss.nx)
        u_val = np.ones(ss.nu)
        result = F(x_val, u_val)
        assert result.shape == (ss.nx, 1)
