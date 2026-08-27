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

"""State and bias estimation in ``neuralmpcx.util.estimators``.

Five estimators that must agree with each other where their assumptions
overlap: the EKF reduces to the Kalman filter on a linear model, the AEKF
reduces to the EKF with no bias channels, and the moving-horizon estimator
reproduces the AEKF when no bound is active. Those equivalences are the
load-bearing tests here; the rest pin the bias selection, the detectability
budget and the failure paths.
"""

import casadi as cs
import numpy as np
import pytest
from numpy.testing import assert_allclose

from neuralmpcx.util.estimators import (
    AugmentedExtendedKalmanFilter,
    AugmentedKalmanFilter,
    ExtendedKalmanFilter,
    KalmanFilter,
    MovingHorizonEstimator,
    bias_detectability,
)


def _make_linear_f(Ad, Bd):
    """Build a CasADi function f(x, u) = Ad @ x + Bd @ u."""
    x = cs.MX.sym("x", Ad.shape[0])
    u = cs.MX.sym("u", Bd.shape[1])
    return cs.Function("f", [x, u], [cs.DM(Ad) @ x + cs.DM(Bd) @ u])


class TestKalmanFilter:
    """The linear Kalman filter: recursion, reset and shape validation."""

    def test_initialization_default(self):
        """Dimensions come from the matrices; the estimate starts at zero, P at identity."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.array([[1.0, 0.0]])

        kf = KalmanFilter(Ad, Bd, Cd)

        assert kf.nx == 2
        assert kf.nu == 1
        assert kf.ny == 1
        assert kf.x_est.shape == (2, 1)
        assert kf.P.shape == (2, 2)
        assert_allclose(kf.x_est, np.zeros((2, 1)))
        assert_allclose(kf.P, np.eye(2))

    def test_initialization_with_all_parameters(self):
        """x0 and P0 seed the filter as given."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.array([[1.0, 0.0]])
        Dd = np.array([[0.1]])
        Q = np.eye(2) * 0.05
        R = np.eye(1) * 0.5
        x0 = np.array([1.0, 2.0])
        P0 = np.eye(2) * 2.0

        kf = KalmanFilter(Ad, Bd, Cd, Dd=Dd, Q=Q, R=R, x0=x0, P0=P0)

        assert_allclose(kf.x_est.flatten(), [1.0, 2.0])
        assert_allclose(kf.P, np.eye(2) * 2.0)

    def test_predict_step(self):
        """One predict is ``Ad @ x + Bd @ u`` and ``Ad @ P @ Ad.T + Q``, hand-computed."""
        Ad = np.array([[0.9, 0.0], [0.0, 0.8]])
        Bd = np.array([[1.0], [0.5]])
        Cd = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1

        kf = KalmanFilter(Ad, Bd, Cd, Q=Q, x0=np.array([1.0, 1.0]))
        kf.predict(u=np.array([0.5]))

        # x_pred = Ad @ x0 + Bd @ u = [[0.9, 0], [0, 0.8]] @ [1, 1] + [[1], [0.5]] @ [0.5]
        #        = [0.9, 0.8] + [0.5, 0.25] = [1.4, 1.05]
        expected_x = np.array([[1.4], [1.05]])
        assert_allclose(kf.x_est, expected_x)

        # P_pred = Ad @ P @ Ad.T + Q, P was identity
        expected_P = Ad @ np.eye(2) @ Ad.T + Q
        assert_allclose(kf.P, expected_P)

    def test_update_step(self):
        """An update pulls the estimate toward the measurement."""
        Ad = np.eye(2)
        Bd = np.zeros((2, 1))
        Cd = np.array([[1.0, 0.0]])
        R = np.eye(1) * 0.1

        kf = KalmanFilter(Ad, Bd, Cd, R=R, x0=np.array([0.0, 0.0]))

        # After update with measurement y=1.0, state should move towards measurement
        kf.update(y=np.array([1.0]))

        # State estimate should have increased (moved towards measurement)
        assert kf.x_est[0, 0] > 0.0

    def test_predict_update_cycle(self):
        """Ten predict/update cycles leave the estimate at its declared shape."""
        Ad = np.array([[0.95]])
        Bd = np.array([[0.1]])
        Cd = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])

        kf = KalmanFilter(Ad, Bd, Cd, Q=Q, R=R)

        # Run several predict-update cycles
        for _ in range(10):
            kf.predict(u=np.array([1.0]))
            kf.update(y=np.array([1.0]))

        # State should converge towards a steady value
        assert kf.x_est.shape == (1, 1)

    def test_reset(self):
        """``reset(x0, P0)`` replaces both the estimate and the covariance."""
        Ad = np.eye(2)
        Bd = np.zeros((2, 1))
        Cd = np.array([[1.0, 0.0]])

        kf = KalmanFilter(Ad, Bd, Cd, x0=np.array([5.0, 5.0]))

        # Run some updates
        kf.predict(u=np.array([0.0]))
        kf.update(y=np.array([1.0]))

        # Reset to new values
        kf.reset(x0=np.array([0.0, 0.0]), P0=np.eye(2) * 10.0)

        assert_allclose(kf.x_est, np.zeros((2, 1)))
        assert_allclose(kf.P, np.eye(2) * 10.0)

    def test_reset_default(self):
        """A bare ``reset()`` falls back to zeros and identity."""
        Ad = np.eye(2)
        Bd = np.zeros((2, 1))
        Cd = np.array([[1.0, 0.0]])

        kf = KalmanFilter(Ad, Bd, Cd, x0=np.array([5.0, 5.0]))
        kf.reset()

        assert_allclose(kf.x_est, np.zeros((2, 1)))
        assert_allclose(kf.P, np.eye(2))

    def test_invalid_Ad_shape(self):
        """A non-square ``Ad`` is refused at construction."""
        Ad = np.array([[0.9, 0.1, 0.0], [0.0, 0.95, 0.0]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="Ad must be square"):
            KalmanFilter(Ad, Bd, Cd)

    def test_invalid_Bd_rows(self):
        """``Bd`` must carry one row per state."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1], [0.2]])  # 3 rows instead of 2
        Cd = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="Bd must have 2 rows"):
            KalmanFilter(Ad, Bd, Cd)

    def test_invalid_Cd_columns(self):
        """``Cd`` must carry one column per state."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.array([[1.0, 0.0, 0.0]])  # 3 columns instead of 2

        with pytest.raises(ValueError, match="Cd must have 2 columns"):
            KalmanFilter(Ad, Bd, Cd)

    def test_casadi_dm_input(self):
        """Matrices and signals may arrive as ``cs.DM``."""
        try:
            import casadi as cs
        except ImportError:
            pytest.skip("CasADi not available")

        Ad = cs.DM([[0.9, 0.1], [0.0, 0.95]])
        Bd = cs.DM([[0.0], [0.1]])
        Cd = cs.DM([[1.0, 0.0]])

        kf = KalmanFilter(Ad, Bd, Cd)

        assert kf.nx == 2
        assert kf.nu == 1
        assert kf.ny == 1

        # Test predict/update with CasADi DM
        kf.predict(u=cs.DM([1.0]))
        kf.update(y=cs.DM([0.5]))

        assert kf.x_est.shape == (2, 1)


class TestExtendedKalmanFilter:
    """The EKF: dimensions inferred from f and h, and agreement with the linear filter."""

    def test_initialization_default(self):
        """nx, nu and ny are read off ``f`` and ``h`` rather than passed in."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        ekf = ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C)

        assert ekf.nx == 2
        assert ekf.nu == 1
        assert ekf.ny == 1
        assert ekf.x_est.shape == (2, 1)
        assert ekf.P.shape == (2, 2)
        assert_allclose(ekf.x_est, np.zeros((2, 1)))
        assert_allclose(ekf.P, np.eye(2))

    def test_initialization_with_all_parameters(self):
        """x0 and P0 seed the filter as given."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.05
        R = np.eye(1) * 0.5
        x0 = np.array([1.0, 2.0])
        P0 = np.eye(2) * 2.0

        ekf = ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C, Q=Q, R=R, x0=x0, P0=P0)

        assert_allclose(ekf.x_est.flatten(), [1.0, 2.0])
        assert_allclose(ekf.P, np.eye(2) * 2.0)

    def test_linear_equivalence_with_kalman_filter(self):
        """EKF on a linear system must match the linear KalmanFilter exactly."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.eye(1) * 0.1
        x0 = np.array([0.5, -0.5])
        P0 = np.eye(2) * 2.0

        kf = KalmanFilter(Ad, Bd, Cd, Q=Q, R=R, x0=x0, P0=P0)
        ekf = ExtendedKalmanFilter(
            _make_linear_f(Ad, Bd), Cd, Q=Q, R=R, x0=x0, P0=P0
        )

        rng = np.random.default_rng(42)
        for _ in range(10):
            u = rng.standard_normal((1, 1))
            y = rng.standard_normal((1, 1))
            kf.predict(u=u)
            ekf.predict(u=u)
            assert_allclose(ekf.x_est, kf.x_est, rtol=1e-10)
            assert_allclose(ekf.P, kf.P, rtol=1e-10)
            kf.update(y=y)
            ekf.update(y=y)
            assert_allclose(ekf.x_est, kf.x_est, rtol=1e-10)
            assert_allclose(ekf.P, kf.P, rtol=1e-10)

    def test_predict_step(self):
        """With a linear ``f`` the Jacobian is ``Ad``, so predict is hand-computable."""
        Ad = np.array([[0.9, 0.0], [0.0, 0.8]])
        Bd = np.array([[1.0], [0.5]])
        C = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1

        ekf = ExtendedKalmanFilter(
            _make_linear_f(Ad, Bd), C, Q=Q, x0=np.array([1.0, 1.0])
        )
        ekf.predict(u=np.array([0.5]))

        # x_pred = Ad @ x0 + Bd @ u = [1.4, 1.05]
        assert_allclose(ekf.x_est, np.array([[1.4], [1.05]]))

        # Jacobian of a linear f is exactly Ad, so P_pred = Ad @ P @ Ad.T + Q
        assert_allclose(ekf.P, Ad @ np.eye(2) @ Ad.T + Q)

    def test_h_as_function_matches_matrix(self):
        """h given as a CasADi function must match the equivalent matrix C."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        x = cs.MX.sym("x", 2)
        h_fun = cs.Function("h", [x], [cs.DM(C) @ x])

        ekf_mat = ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C)
        ekf_fun = ExtendedKalmanFilter(_make_linear_f(Ad, Bd), h_fun)

        for _ in range(5):
            ekf_mat.predict(u=np.array([1.0]))
            ekf_fun.predict(u=np.array([1.0]))
            ekf_mat.update(y=np.array([0.5]))
            ekf_fun.update(y=np.array([0.5]))

        assert_allclose(ekf_fun.x_est, ekf_mat.x_est, rtol=1e-12)
        assert_allclose(ekf_fun.P, ekf_mat.P, rtol=1e-12)

    def test_nonlinear_convergence(self):
        """EKF converges to the true state of a nonlinear system."""
        dt = 0.1
        x = cs.MX.sym("x", 1)
        u = cs.MX.sym("u", 1)
        f = cs.Function("f", [x, u], [x + dt * (-(x**3) + u)])
        C = np.array([[1.0]])

        ekf = ExtendedKalmanFilter(
            f,
            C,
            Q=np.array([[1e-4]]),
            R=np.array([[1e-2]]),
            x0=np.array([2.0]),  # wrong initial estimate
            P0=np.array([[1.0]]),
        )

        rng = np.random.default_rng(0)
        x_true = np.array([[0.5]])
        initial_error = abs(ekf.x_est[0, 0] - x_true[0, 0])
        for _ in range(100):
            u_k = np.array([[0.3]])
            x_true = np.asarray(f(x_true, u_k)).reshape(1, 1)
            y_meas = x_true + rng.normal(0.0, 0.1, size=(1, 1))
            ekf.predict(u=u_k)
            ekf.update(y=y_meas)

        final_error = abs(ekf.x_est[0, 0] - x_true[0, 0])
        assert final_error < initial_error
        assert final_error < 0.1

    def test_partial_measurement_state_reconstruction(self):
        """EKF reconstructs unmeasured states of a nonlinear MIMO system."""
        # Two-state system where only the second state is measured; the
        # first state is observable through the coupling term.
        dt = 0.05
        x = cs.MX.sym("x", 2)
        u = cs.MX.sym("u", 1)
        dx = cs.vertcat(-0.5 * x[0] + u, x[0] - 0.2 * x[1] ** 2)
        f = cs.Function("f", [x, u], [x + dt * dx])
        C = np.array([[0.0, 1.0]])

        ekf = ExtendedKalmanFilter(
            f,
            C,
            Q=np.eye(2) * 1e-5,
            R=np.array([[1e-3]]),
            x0=np.array([2.0, 0.0]),  # wrong x[0] on purpose
            P0=np.diag([1.0, 0.01]),
        )

        rng = np.random.default_rng(1)
        x_true = np.array([[0.2], [0.5]])
        for _ in range(200):
            u_k = np.array([[0.4]])
            x_true = np.asarray(f(x_true, u_k)).reshape(2, 1)
            y_meas = x_true[1:2] + rng.normal(0.0, 0.03, size=(1, 1))
            ekf.predict(u=u_k)
            ekf.update(y=y_meas)

        # The unmeasured first state must be reconstructed
        assert abs(ekf.x_est[0, 0] - x_true[0, 0]) < 0.05

    def test_f_not_function_raises(self):
        """``f`` must be a ``casadi.Function``, not a matrix."""
        with pytest.raises(TypeError, match="f must be a casadi.Function"):
            ExtendedKalmanFilter(np.eye(2), np.array([[1.0, 0.0]]))

    def test_f_wrong_n_in_raises(self):
        """``f`` takes exactly ``(x, u)``; a third input is refused."""
        x = cs.MX.sym("x", 2)
        u = cs.MX.sym("u", 1)
        d = cs.MX.sym("d", 1)
        f = cs.Function("f", [x, u, d], [x])

        with pytest.raises(ValueError, match="f must take exactly 2 inputs"):
            ExtendedKalmanFilter(f, np.array([[1.0, 0.0]]))

    def test_h_wrong_columns_raises(self):
        """``h`` must carry one column per state."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0, 0.0]])  # 3 columns instead of 2

        with pytest.raises(ValueError, match="h must have 2 columns"):
            ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C)

    def test_invalid_Q_shape(self):
        """``Q`` is sized ``(nx, nx)``."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="Q must have shape"):
            ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C, Q=np.eye(3))

    def test_invalid_R_shape(self):
        """``R`` is sized ``(ny, ny)``."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="R must have shape"):
            ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C, R=np.eye(2))

    def test_invalid_P0_shape(self):
        """``P0`` is sized ``(nx, nx)``."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="P0 must have shape"):
            ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C, P0=np.eye(3))

    def test_invalid_x0_length(self):
        """``x0`` carries one element per state."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="x0 must have 2 elements"):
            ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C, x0=np.array([1.0]))

    def test_reset(self):
        """``reset(x0, P0)`` replaces both the estimate and the covariance."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        ekf = ExtendedKalmanFilter(
            _make_linear_f(Ad, Bd), C, x0=np.array([5.0, 5.0])
        )

        ekf.predict(u=np.array([0.0]))
        ekf.update(y=np.array([1.0]))

        ekf.reset(x0=np.array([0.0, 0.0]), P0=np.eye(2) * 10.0)

        assert_allclose(ekf.x_est, np.zeros((2, 1)))
        assert_allclose(ekf.P, np.eye(2) * 10.0)

    def test_reset_default(self):
        """A bare ``reset()`` falls back to zeros and identity."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        ekf = ExtendedKalmanFilter(
            _make_linear_f(Ad, Bd), C, x0=np.array([5.0, 5.0])
        )
        ekf.reset()

        assert_allclose(ekf.x_est, np.zeros((2, 1)))
        assert_allclose(ekf.P, np.eye(2))

    def test_casadi_dm_input(self):
        """``cs.DM`` signals go in, NumPy comes back out."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])

        ekf = ExtendedKalmanFilter(_make_linear_f(Ad, Bd), C)

        ekf.predict(u=cs.DM([1.0]))
        ekf.update(y=cs.DM([0.5]))

        assert isinstance(ekf.x_est, np.ndarray)
        assert ekf.x_est.shape == (2, 1)


class TestAugmentedKalmanFilter:
    """The linear filter augmented with selected input and output biases."""

    def test_initialization_default(self):
        """Default augmentation is an output bias per output and no input bias.

        Input bias is opt-in because it enters the dynamics and competes with
        the state for the same innovation; and on nu=ny=1 biasing both would
        exceed the detectability budget anyway.
        """
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.array([[1.0, 0.0]])

        kf = AugmentedKalmanFilter(Ad, Bd, Cd)

        assert kf.nx == 2
        assert kf.nu == 1
        assert kf.ny == 1
        assert (kf.n_du, kf.n_dy) == (0, 1)
        assert kf.du_index == () and kf.dy_index == (0,)
        assert kf.n_aug == 3  # 2 + 0 + 1

        assert kf.x_est.shape == (2, 1)
        # the bias estimates stay at *signal* width even though the augmented
        # state is compact, so a consumer indexing by signal position is unaware
        # of the selection
        assert kf.du_bias_est.shape == (1, 1)
        assert kf.dy_bias_est.shape == (1, 1)
        assert kf.z_est.shape == (3, 1)
        assert kf.P.shape == (3, 3)

    def test_initialization_with_all_parameters(self):
        """Seeds arrive at full signal width; the unselected input channel is dropped."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.array([[1.0, 0.0]])
        Dd = np.array([[0.1]])
        Q_x = np.eye(2) * 0.05
        Q_dy = np.array([[0.03]])
        R = np.array([[0.5]])
        x0 = np.array([1.0, 2.0])
        du_bias0 = np.array([0.1])
        dy_bias0 = np.array([0.2])

        kf = AugmentedKalmanFilter(
            Ad,
            Bd,
            Cd,
            Dd=Dd,
            Q_x=Q_x,
            Q_dy=Q_dy,
            R=R,
            x0=x0,
            du_bias0=du_bias0,
            dy_bias0=dy_bias0,
        )

        assert_allclose(kf.x_est.flatten(), [1.0, 2.0])
        assert_allclose(kf.dy_bias_est.flatten(), [0.2])
        # the seed is given at full width; the unselected input channel is
        # dropped rather than silently creating a state for it
        assert_allclose(kf.du_bias_est.flatten(), [0.0])

    def test_augmented_state_dimensions(self):
        """z grows by the selected bias count while the estimates stay at signal width."""
        nx, nu, ny = 4, 2, 3
        rng = np.random.default_rng(0)
        Ad = np.eye(nx) * 0.9
        Bd = rng.standard_normal((nx, nu)) * 0.1
        Cd = rng.standard_normal((ny, nx))

        # the full budget of ny = 3 states, one of them on an input
        kf = AugmentedKalmanFilter(Ad, Bd, Cd, du_index=[0], dy_index=[0, 1])

        assert kf.nx == nx
        assert kf.nu == nu
        assert kf.ny == ny
        assert kf.n_aug == nx + 1 + 2
        assert kf.z_est.shape == (nx + 3, 1)
        assert kf.P.shape == (nx + 3, nx + 3)
        # ...while the bias estimates stay at signal width
        assert kf.du_bias_est.shape == (nu, 1)
        assert kf.dy_bias_est.shape == (ny, 1)

    def test_predict_step(self):
        """The plant state advances and the bias random walks stay where they started."""
        Ad = np.array([[0.9]])
        Bd = np.array([[1.0]])
        Cd = np.array([[1.0]])

        kf = AugmentedKalmanFilter(Ad, Bd, Cd, x0=np.array([1.0]))
        kf.predict(u=np.array([0.5]))

        # Plant state: x_pred = 0.9 * 1.0 + 1.0 * 0.5 = 1.4
        assert_allclose(kf.x_est.flatten(), [1.4], rtol=1e-10)

        # Biases remain at zero (random walk with zero initial)
        assert_allclose(kf.du_bias_est.flatten(), [0.0], atol=1e-10)
        assert_allclose(kf.dy_bias_est.flatten(), [0.0], atol=1e-10)

    def test_bias_estimation(self):
        """A persistent output error is absorbed by the output bias.

        Deliberately *not* an integrator, though it used to be: an output bias
        on a channel the model already integrates is unidentifiable. Both are
        free integrators feeding the same output, so the detectability check
        now refuses it, and the loose tolerance below is what let the old
        version of this test pass anyway.
        """
        Ad = np.array([[0.95]])
        Bd = np.array([[0.1]])
        Cd = np.array([[1.0]])

        kf = AugmentedKalmanFilter(
            Ad,
            Bd,
            Cd,
            Q_x=np.array([[0.01]]),
            Q_dy=np.array([[0.01]]),
            R=np.array([[0.1]]),
        )

        # Simulate with constant output bias of 0.5
        true_bias = 0.5
        for _ in range(50):
            kf.predict(u=np.array([0.0]))
            # Measurement includes bias
            y_meas = kf.x_est[0, 0] + true_bias
            kf.update(y=np.array([y_meas]))

        # Output bias estimate should converge towards true bias
        assert abs(kf.dy_bias_est[0, 0] - true_bias) < 0.3

    def test_get_mpc_biases_format(self):
        """``get_mpc_biases`` returns ``du_bias`` and ``dy_bias`` at signal width."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0, 0.1], [0.1, 0.0]])
        Cd = np.array([[1.0, 0.0], [0.0, 1.0]])

        kf = AugmentedKalmanFilter(Ad, Bd, Cd)

        biases = kf.get_mpc_biases()

        assert isinstance(biases, dict)
        assert "du_bias" in biases
        assert "dy_bias" in biases
        assert biases["du_bias"].shape == (2, 1)
        assert biases["dy_bias"].shape == (2, 1)

    def test_get_mpc_biases_returns_copies(self):
        """The returned dict does not alias internal state."""
        Ad = np.array([[0.9]])
        Bd = np.array([[0.1]])
        Cd = np.array([[1.0]])

        kf = AugmentedKalmanFilter(Ad, Bd, Cd)

        biases1 = kf.get_mpc_biases()
        biases1["du_bias"][0, 0] = 999.0

        biases2 = kf.get_mpc_biases()
        assert biases2["du_bias"][0, 0] != 999.0

    def test_reset(self):
        """A reset seeds the plant state and both biases.

        The bias seeds go in at full signal width and are gathered onto the
        selection, so this is also the round-trip check that the two widths do
        not get crossed.
        """
        Ad = np.array([[0.9, 0.0], [0.0, 0.8]])
        Bd = np.array([[0.1, 0.0], [0.0, 0.1]])
        Cd = np.eye(2)

        kf = AugmentedKalmanFilter(
            Ad, Bd, Cd, du_index=[1], dy_index=[0], x0=np.array([5.0, 5.0])
        )

        # Run some updates
        for _ in range(5):
            kf.predict(u=np.array([1.0, 1.0]))
            kf.update(y=np.array([1.0, 1.0]))

        # Reset to new values, given at signal width
        kf.reset(
            x0=np.array([0.0, 0.0]),
            du_bias0=np.array([0.9, 0.1]),
            dy_bias0=np.array([0.2, 0.7]),
        )

        assert_allclose(kf.x_est.flatten(), [0.0, 0.0])
        # only the selected channels survive; the rest read a hard zero
        assert_allclose(kf.du_bias_est.flatten(), [0.0, 0.1])
        assert_allclose(kf.dy_bias_est.flatten(), [0.2, 0.0])
        assert kf.z_est.shape == (4, 1)  # 2 states + 1 du + 1 dy

    def test_invalid_Q_x_shape(self):
        """``Q_x`` is sized to the plant state, not to the augmented one."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="Q_x must have shape"):
            AugmentedKalmanFilter(Ad, Bd, Cd, Q_x=np.eye(3))

    def test_invalid_Q_du_shape(self):
        """Q_du is sized to the *selection*, not to the full input width."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0, 0.1], [0.1, 0.0]])
        Cd = np.array([[1.0, 0.0]])

        with pytest.raises(
            ValueError, match=r"Q_du must have shape \(1, 1\).*selected bias channels"
        ):
            AugmentedKalmanFilter(
                Ad, Bd, Cd, du_index=[0], dy_index=[], Q_du=np.eye(2)
            )

    def test_casadi_dm_input(self):
        """Matrices and signals may arrive as ``cs.DM``."""
        try:
            import casadi as cs
        except ImportError:
            pytest.skip("CasADi not available")

        Ad = cs.DM([[0.9, 0.1], [0.0, 0.95]])
        Bd = cs.DM([[0.0], [0.1]])
        Cd = cs.DM([[1.0, 0.0]])

        kf = AugmentedKalmanFilter(Ad, Bd, Cd)

        assert kf.nx == 2
        assert kf.nu == 1
        assert kf.ny == 1

        # Test predict/update with CasADi DM
        kf.predict(u=cs.DM([1.0]))
        kf.update(y=cs.DM([0.5]))

        assert kf.x_est.shape == (2, 1)
        assert isinstance(kf.get_mpc_biases()["du_bias"], np.ndarray)

    # ------------------------------------------------------------------
    # the input bias reaches the *dynamics* (P29)
    # ------------------------------------------------------------------

    def test_input_bias_recovers_an_actuator_offset(self):
        """A constant offset on the actuator is recovered by ``du_bias``.

        The property the filter could not satisfy at all before ``A_aug`` grew
        its ``Bd @ S_du`` cross-block: the input bias reached the model only
        through ``Dd``, so on a plant with no feedthrough it was a dead state
        that random-walked under ``Q_du``, corrected nothing, and was reported
        as an estimate regardless.
        """
        Ad = np.array([[0.8]])
        Bd = np.array([[0.5]])
        Cd = np.array([[1.0]])
        true_offset = 0.7

        kf = AugmentedKalmanFilter(
            Ad,
            Bd,
            Cd,
            du_index=[0],
            dy_index=[],
            Q_x=np.array([[1e-4]]),
            Q_du=np.array([[1e-2]]),
            R=np.array([[1e-3]]),
        )

        x = 0.0
        for _ in range(300):
            u = np.array([[0.3]])
            kf.predict(u)
            x = float((Ad @ [[x]] + Bd @ (u + true_offset)).item())
            kf.update(np.array([[x]]), u)

        assert kf.du_bias_est[0, 0] == pytest.approx(true_offset, abs=1e-3)

    def test_zero_feedthrough_no_longer_makes_the_input_bias_inert(self):
        """With ``Dd = 0`` the input bias must still move the prediction.

        The shipped grinding template converts to ``D = 0``, so this is the
        configuration the defect bit in production.
        """
        Ad = np.array([[0.8]])
        Bd = np.array([[0.5]])
        Cd = np.array([[1.0]])
        Dd = np.zeros((1, 1))

        kf = AugmentedKalmanFilter(Ad, Bd, Cd, Dd, du_index=[0], dy_index=[])
        kf.reset(x0=np.array([1.0]), du_bias0=np.array([2.0]))
        kf.predict(np.array([[0.0]]))

        # x⁺ = 0.8·1 + 0.5·(0 + 2) = 1.8, not the 0.8 a zero cross-block gives
        assert kf.x_est[0, 0] == pytest.approx(1.8)

    def test_bias_budget_is_enforced(self):
        """More bias states than measurements is never detectable."""
        Ad = np.array([[0.9]])
        Bd = np.array([[0.1]])
        Cd = np.array([[1.0]])

        with pytest.raises(ValueError, match="2 bias states were requested"):
            AugmentedKalmanFilter(Ad, Bd, Cd, du_index=[0], dy_index=[0])

    def test_output_bias_on_an_integrating_channel_is_refused(self):
        """The incremental rank test's headline case.

        A bias and an integrator on the same output are both free integrators
        feeding the same measurement, so no data separates them: the split is
        set by the ratio of ``Q_dy`` to ``Q_x``. The channel is named, because
        "deficiency 1" alone does not tell anyone which one to drop.
        """
        Ad = np.diag([1.0, 0.5])  # channel 0 integrates
        Bd = np.array([[1.0], [1.0]])
        Cd = np.eye(2)

        with pytest.raises(ValueError, match="never separate output channel 0"):
            AugmentedKalmanFilter(Ad, Bd, Cd, dy_index=[0, 1])

        # ...but the non-integrating channel is fine
        kf = AugmentedKalmanFilter(Ad, Bd, Cd, dy_index=[1])
        assert kf.detectability_report()["detectable"]

    def test_check_detectability_can_be_turned_off(self):
        Ad = np.diag([1.0, 0.5])
        Bd = np.array([[1.0], [1.0]])
        Cd = np.eye(2)

        kf = AugmentedKalmanFilter(
            Ad, Bd, Cd, dy_index=[0, 1], check_detectability=False
        )
        assert kf.detectability_report()["detectable"] is False

    def test_unobservable_integrator_warns_but_does_not_block(self):
        """A rank-deficient *realization* is a model property, not a selection one.

        It is equally true of the plain Kalman filter on the same model, so
        refusing only the augmented one would be incoherent. The bias states
        are checked against that baseline instead.
        """
        # state 1 integrates and no measurement sees it
        Ad = np.diag([0.9, 1.0])
        Bd = np.array([[0.1], [0.1]])
        Cd = np.array([[1.0, 0.0]])

        with pytest.warns(UserWarning, match="rank-deficient"):
            kf = AugmentedKalmanFilter(Ad, Bd, Cd, dy_index=[0])

        report = kf.detectability_report()
        assert report["plant_rank_deficiency"] == 1
        assert report["detectable"] is True  # the bias itself is separable


class TestBiasDetectability:
    """The shared rank test both augmented filters delegate to."""

    def test_incremental_and_absolute_agree_on_a_detectable_plant(self):
        """Where the plant is detectable the baseline IS nx, so the incremental
        statement reduces to the textbook ``rank == nx + n_bias``."""
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        B = np.array([[0.1], [0.2]])
        C = np.array([[1.0, 0.0]])

        report = bias_detectability(A, B, C, dy_index=[0])
        assert report["baseline_rank"] == 2  # == nx
        assert report["rank"] == report["required"] == 3
        assert report["detectable"] is True

    def test_names_every_redundant_channel(self):
        A = np.diag([1.0, 1.0])
        B = np.array([[1.0], [0.0]])
        C = np.eye(2)

        report = bias_detectability(A, B, C, dy_index=[0, 1])
        assert report["detectable"] is False
        assert report["redundant_dy"] == (0, 1)

    def test_no_bias_is_vacuous(self):
        A = np.array([[0.9]])
        B = np.array([[0.1]])
        C = np.array([[1.0]])

        report = bias_detectability(A, B, C)
        assert report["n_bias"] == 0
        assert report["gained"] == 0
        assert report["detectable"] is True


class TestAugmentedExtendedKalmanFilter:
    """The nonlinear counterpart, with the bias states carried through CasADi AD."""

    def test_initialization_default(self):
        """Default selection is output bias on every channel, no input bias."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        aekf = AugmentedExtendedKalmanFilter(f, np.eye(2))

        assert aekf.nx == 2
        assert aekf.nu == 1
        assert aekf.ny == 2
        assert aekf.du_index == ()
        assert aekf.dy_index == (0, 1)
        assert aekf.n_du == 0
        assert aekf.n_dy == 2
        assert aekf.n_aug == 4
        assert_allclose(aekf.x_est, np.zeros((2, 1)))
        assert_allclose(aekf.P, np.eye(4))

    def test_linear_equivalence_with_kalman_filter(self):
        """The augmented model matches a hand-built linear KF exactly.

        This is the load-bearing check on the symbolic augmentation: with a
        linear f, the AEKF must reproduce a plain KalmanFilter running on
        z = [x; du; dy] with A_aug carrying the Bd @ S_du cross-block.
        """
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        Cd = np.eye(2)
        S_du = np.array([[1.0]])  # input bias on channel 0
        S_dy = np.array([[1.0], [0.0]])  # output bias on channel 0
        Q = np.diag([0.1, 0.1, 0.01, 0.01])
        R = np.eye(2) * 0.1

        A_aug = np.block(
            [
                [Ad, Bd @ S_du, np.zeros((2, 1))],
                [np.zeros((1, 2)), np.eye(1), np.zeros((1, 1))],
                [np.zeros((1, 2)), np.zeros((1, 1)), np.eye(1)],
            ]
        )
        B_aug = np.vstack([Bd, np.zeros((1, 1)), np.zeros((1, 1))])
        C_aug = np.hstack([Cd, np.zeros((2, 1)), S_dy])

        aekf = AugmentedExtendedKalmanFilter(
            _make_linear_f(Ad, Bd), Cd, du_index=[0], dy_index=[0], R=R
        )
        kf = KalmanFilter(A_aug, B_aug, C_aug, Q=Q, R=R)

        rng = np.random.default_rng(0)
        for _ in range(10):
            u_k = rng.normal(size=(1, 1))
            y_k = rng.normal(size=(2, 1))

            aekf.predict(u=u_k)
            kf.predict(u=u_k)
            assert_allclose(aekf.z_est, kf.x_est, rtol=1e-10, atol=1e-12)
            assert_allclose(aekf.P, kf.P, rtol=1e-10, atol=1e-12)

            aekf.update(y=y_k)
            kf.update(y=y_k)
            assert_allclose(aekf.z_est, kf.x_est, rtol=1e-10, atol=1e-12)
            assert_allclose(aekf.P, kf.P, rtol=1e-10, atol=1e-12)

    def test_no_bias_channels_matches_plain_ekf(self):
        """With no bias channels the filter degenerates to the plain EKF."""
        dt = 0.1
        x = cs.MX.sym("x", 2)
        u = cs.MX.sym("u", 1)
        f = cs.Function("f", [x, u], [x + dt * cs.vertcat(x[1], -cs.sin(x[0]) + u)])
        C = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])

        aekf = AugmentedExtendedKalmanFilter(
            f, C, du_index=[], dy_index=[], Q_x=Q, R=R
        )
        ekf = ExtendedKalmanFilter(f, C, Q=Q, R=R)

        assert aekf.n_aug == 2
        rng = np.random.default_rng(3)
        for _ in range(10):
            u_k = rng.normal(size=(1, 1))
            y_k = rng.normal(size=(1, 1))
            aekf.predict(u=u_k)
            ekf.predict(u=u_k)
            aekf.update(y=y_k)
            ekf.update(y=y_k)
            assert_allclose(aekf.x_est, ekf.x_est, rtol=1e-12, atol=1e-14)
            assert_allclose(aekf.P, ekf.P, rtol=1e-12, atol=1e-14)

    def test_input_bias_recovers_actuator_offset(self):
        """A constant actuator offset is recovered on a nonlinear plant.

        This is what the linear AugmentedKalmanFilter cannot do: there the
        input bias only reaches the output through Dd, so with no feedthrough
        it never moves. Here it enters f(x, u + du_bias) and is identifiable
        from the state response alone.
        """
        dt = 0.1
        x = cs.MX.sym("x", 1)
        u = cs.MX.sym("u", 1)
        f = cs.Function("f", [x, u], [x + dt * (-(x**3) + u)])
        C = np.array([[1.0]])
        true_offset = 0.3

        aekf = AugmentedExtendedKalmanFilter(
            f,
            C,
            du_index=[0],
            dy_index=[],  # the whole budget (ny = 1) goes to the input bias
            Q_x=np.array([[1e-6]]),
            Q_du=np.array([[1e-4]]),
            R=np.array([[1e-3]]),
            P0=np.diag([0.1, 1.0]),
        )

        rng = np.random.default_rng(2)
        x_true = np.array([[0.5]])
        for _ in range(400):
            u_k = np.array([[0.4]])
            # the plant sees the offset control; the filter is only told u_k
            x_true = np.asarray(f(x_true, u_k + true_offset)).reshape(1, 1)
            y_meas = x_true + rng.normal(0.0, 0.01, size=(1, 1))
            aekf.predict(u=u_k)
            aekf.update(y=y_meas)

        assert abs(aekf.du_bias_est[0, 0] - true_offset) < 0.05
        assert abs(aekf.x_est[0, 0] - x_true[0, 0]) < 0.05

    def test_output_bias_estimation(self):
        """A constant output bias converges (nonlinear counterpart of the AKF test).

        The plant is damped on purpose. On an integrating plant an output bias
        is genuinely indistinguishable from a state offset, and the
        detectability check refuses that selection; see
        ``test_output_bias_on_an_integrator_is_undetectable``.
        """
        dt = 0.1
        x = cs.MX.sym("x", 1)
        u = cs.MX.sym("u", 1)
        f = cs.Function("f", [x, u], [x + dt * (-x - x**3 + u)])
        C = np.array([[1.0]])
        true_bias = 0.5

        aekf = AugmentedExtendedKalmanFilter(
            f,
            C,
            du_index=[],
            dy_index=[0],
            Q_x=np.array([[1e-6]]),
            Q_dy=np.array([[1e-4]]),
            R=np.array([[1e-3]]),
            P0=np.diag([0.1, 1.0]),
        )

        rng = np.random.default_rng(4)
        x_true = np.array([[0.5]])
        for _ in range(400):
            u_k = np.array([[0.4]])
            x_true = np.asarray(f(x_true, u_k)).reshape(1, 1)
            # the sensor reads high by a constant amount
            y_meas = x_true + true_bias + rng.normal(0.0, 0.01, size=(1, 1))
            aekf.predict(u=u_k)
            aekf.update(y=y_meas)

        assert abs(aekf.dy_bias_est[0, 0] - true_bias) < 0.05
        assert abs(aekf.x_est[0, 0] - x_true[0, 0]) < 0.05

    def test_over_budget_bias_selection_raises(self):
        """More bias states than measurements is refused."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        with pytest.raises(ValueError, match="3 bias states were requested"):
            AugmentedExtendedKalmanFilter(f, np.eye(2), du_index=[0], dy_index=[0, 1])

    def test_over_budget_counts_measurements_not_states(self):
        """The cap is the number of measured channels, not the state count."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)
        C = np.array([[1.0, 0.0]])  # only one channel measured

        with pytest.raises(ValueError, match="only 1 measured output"):
            AugmentedExtendedKalmanFilter(f, C, du_index=[0], dy_index=[0])

    def test_undetectable_selection_raises(self):
        """A bias on a channel the model cannot separate is refused."""
        # x[1] does not respond to u at all, so an input bias cannot explain
        # anything the output bias on the same channel does not already cover
        Ad = np.array([[0.9, 0.0], [0.0, 0.95]])
        Bd = np.array([[0.1], [0.0]])
        f = _make_linear_f(Ad, Bd)
        C = np.array([[0.0, 1.0], [0.0, 1.0]])  # both rows read the same state

        with pytest.raises(ValueError, match="not detectable"):
            AugmentedExtendedKalmanFilter(f, C, du_index=[0], dy_index=[0])

    def test_output_bias_on_an_integrator_is_undetectable(self):
        """An output bias cannot be separated from the state of an integrator.

        The same plant accepts an input bias, which is identifiable from how
        the state ramps, so the check distinguishes the two cases rather than
        counting channels.
        """
        dt = 0.1
        x = cs.MX.sym("x", 1)
        u = cs.MX.sym("u", 1)
        f = cs.Function("f", [x, u], [x + dt * u])  # pure integrator
        C = np.array([[1.0]])

        with pytest.raises(ValueError, match="not detectable"):
            AugmentedExtendedKalmanFilter(f, C, dy_index=[0])

        aekf = AugmentedExtendedKalmanFilter(f, C, du_index=[0], dy_index=[])
        assert aekf.detectability_report()["detectable"] is True

    def test_an_unobservable_mode_no_longer_vetoes_a_sound_bias(self):
        """The incremental rank test, delegated to the shared helper.

        A realization with a mode at z = 1 that no output sees is rank-deficient
        in ``[[I - A], [C]]`` before any augmentation. Measured absolutely, that
        deficiency was charged to the bias selection and every augmentation on
        such a model was refused; measured against the plant's own baseline, a
        bias that genuinely adds rank is accepted and the model earns a warning
        instead. The AKF and the AEKF call the same helper, so they agree.
        """
        Ad = np.diag([0.9, 1.0])  # state 1 integrates...
        Bd = np.array([[0.1], [0.1]])
        f = _make_linear_f(Ad, Bd)
        C = np.array([[1.0, 0.0]])  # ...and no output observes it

        with pytest.warns(UserWarning, match="rank-deficient"):
            aekf = AugmentedExtendedKalmanFilter(f, C, dy_index=[0])

        report = aekf.detectability_report()
        assert report["plant_rank_deficiency"] == 1
        assert report["detectable"] is True

    def test_check_detectability_false_builds(self):
        """The escape hatch builds the filter and the report still reads false."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        aekf = AugmentedExtendedKalmanFilter(
            f, np.eye(2), du_index=[0], dy_index=[0, 1], check_detectability=False
        )

        report = aekf.detectability_report()
        assert aekf.n_aug == 5
        assert report["n_bias"] == 3
        assert report["max_bias_states"] == 2
        assert report["deficiency"] == 1
        assert report["detectable"] is False

    def test_detectability_report_on_valid_selection(self):
        """A within-budget, separable selection reports detectable."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        aekf = AugmentedExtendedKalmanFilter(f, np.eye(2), du_index=[0], dy_index=[0])

        report = aekf.detectability_report()
        assert report["detectable"] is True
        assert report["rank"] == report["required"] == 4
        assert report["deficiency"] == 0

    def test_no_bias_report_is_vacuous(self):
        """With nothing augmented the report makes no claim about the plant."""
        f = _make_linear_f(np.array([[1.0]]), np.array([[0.1]]))

        aekf = AugmentedExtendedKalmanFilter(f, np.array([[1.0]]), dy_index=[])

        assert aekf.detectability_report() == {
            "n_bias": 0,
            "max_bias_states": 1,
            "baseline_rank": 1,
            "rank": 1,
            "gained": 0,
            "required": 1,
            "deficiency": 0,
            "plant_rank_deficiency": 0,
            "detectable": True,
            "redundant_du": (),
            "redundant_dy": (),
        }

    def test_biases_are_full_width_and_zero_off_selection(self):
        """Bias estimates come back at signal width, zero on unselected channels."""
        Ad = np.array([[0.9, 0.0], [0.0, 0.95]])
        Bd = np.array([[0.1, 0.0], [0.0, 0.1]])
        f = _make_linear_f(Ad, Bd)

        aekf = AugmentedExtendedKalmanFilter(
            f, np.eye(2), du_index=[1], dy_index=[0], R=np.eye(2) * 0.1
        )
        aekf.predict(u=np.array([[1.0], [1.0]]))
        aekf.update(y=np.array([[0.5], [0.5]]))

        biases = aekf.get_mpc_biases()
        assert biases["du_bias"].shape == (2, 1)
        assert biases["dy_bias"].shape == (2, 1)
        # only the selected channels can be non-zero
        assert biases["du_bias"][0, 0] == 0.0
        assert biases["dy_bias"][1, 0] == 0.0
        assert biases["du_bias"][1, 0] != 0.0
        assert biases["dy_bias"][0, 0] != 0.0
        # z_est stays compact: nx + n_du + n_dy
        assert aekf.z_est.shape == (4, 1)

    def test_get_mpc_biases_returns_copies(self):
        """The returned dict does not alias internal state."""
        f = _make_linear_f(np.array([[0.9]]), np.array([[0.1]]))

        aekf = AugmentedExtendedKalmanFilter(f, np.array([[1.0]]))

        biases1 = aekf.get_mpc_biases()
        biases1["dy_bias"][0, 0] = 999.0

        biases2 = aekf.get_mpc_biases()
        assert biases2["dy_bias"][0, 0] != 999.0

    def test_reset_round_trips_the_full_estimate(self):
        """A fresh filter reset from another's estimate reproduces it exactly.
        """
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        def build():
            return AugmentedExtendedKalmanFilter(
                f, np.eye(2), du_index=[0], dy_index=[0], R=np.eye(2) * 0.1
            )

        old = build()
        rng = np.random.default_rng(5)
        for _ in range(5):
            old.predict(u=rng.normal(size=(1, 1)))
            old.update(y=rng.normal(size=(2, 1)))

        new = build()
        new.reset(
            x0=old.x_est,
            du_bias0=old.du_bias_est,
            dy_bias0=old.dy_bias_est,
            P0=old.P,
        )

        assert_allclose(new.z_est, old.z_est, rtol=1e-14, atol=1e-16)
        assert_allclose(new.P, old.P, rtol=1e-14, atol=1e-16)

    def test_reset_defaults_to_zeros_and_identity(self):
        """A bare ``reset()`` zeros the estimate and restores an identity covariance."""
        f = _make_linear_f(np.array([[0.9]]), np.array([[0.1]]))
        aekf = AugmentedExtendedKalmanFilter(f, np.array([[1.0]]), du_index=[])

        aekf.predict(u=np.array([1.0]))
        aekf.update(y=np.array([5.0]))
        aekf.reset()

        assert_allclose(aekf.z_est, np.zeros((2, 1)))
        assert_allclose(aekf.P, np.eye(2))

    def test_du_index_out_of_range_raises(self):
        """An input channel outside ``range(nu)`` is refused."""
        f = _make_linear_f(np.array([[0.9]]), np.array([[0.1]]))

        with pytest.raises(ValueError, match="du_index contains out-of-range"):
            AugmentedExtendedKalmanFilter(f, np.array([[1.0]]), du_index=[3])

    def test_dy_index_repeated_raises(self):
        """A repeated output channel is refused."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        with pytest.raises(ValueError, match="dy_index contains repeated"):
            AugmentedExtendedKalmanFilter(f, np.eye(2), dy_index=[0, 0])

    def test_bias_noise_shape_is_over_selected_channels(self):
        """Q_du/Q_dy are sized by the selection, and say so when they are not."""
        Ad = np.array([[0.9, 0.0], [0.0, 0.95]])
        Bd = np.array([[0.1, 0.0], [0.0, 0.1]])
        f = _make_linear_f(Ad, Bd)

        with pytest.raises(ValueError, match=r"Q_du must have shape \(1, 1\)"):
            AugmentedExtendedKalmanFilter(
                f, np.eye(2), du_index=[0], dy_index=[1], Q_du=np.eye(2) * 0.01
            )

    def test_bias_seeds_are_full_width(self):
        """Bias seeds are given at signal width so estimates round-trip."""
        Ad = np.array([[0.9, 0.0], [0.0, 0.95]])
        Bd = np.array([[0.1, 0.0], [0.0, 0.1]])
        f = _make_linear_f(Ad, Bd)

        aekf = AugmentedExtendedKalmanFilter(
            f,
            np.eye(2),
            du_index=[1],
            dy_index=[0],
            du_bias0=np.array([7.0, 0.25]),  # channel 0 is not estimated
            dy_bias0=np.array([0.5, 9.0]),  # channel 1 is not estimated
        )

        assert_allclose(aekf.du_bias_est, np.array([[0.0], [0.25]]))
        assert_allclose(aekf.dy_bias_est, np.array([[0.5], [0.0]]))

        with pytest.raises(ValueError, match="du_bias0 must have 2 elements"):
            AugmentedExtendedKalmanFilter(
                f, np.eye(2), du_index=[1], dy_index=[0], du_bias0=np.array([0.25])
            )

    def test_accepts_casadi_dm_inputs(self):
        """Matrices may arrive as ``cs.DM``."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        aekf = AugmentedExtendedKalmanFilter(
            f,
            cs.DM(np.eye(2)),
            du_index=[0],
            dy_index=[0],
            Q_x=cs.DM(np.eye(2) * 0.1),
            R=cs.DM(np.eye(2) * 0.5),
            x0=cs.DM([1.0, 2.0]),
        )

        assert aekf.n_aug == 4
        assert_allclose(aekf.x_est, np.array([[1.0], [2.0]]))

    def test_multi_output_dynamics_accepted(self):
        """A multi-output f is normalized to its first output, as in the EKF."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        x = cs.MX.sym("x", 2)
        u = cs.MX.sym("u", 1)
        f = cs.Function(
            "f", [x, u], [cs.DM(Ad) @ x + cs.DM(Bd) @ u, cs.sum1(x)]
        )

        aekf = AugmentedExtendedKalmanFilter(f, np.eye(2), du_index=[0], dy_index=[0])

        assert aekf.nx == 2
        assert aekf.n_aug == 4


#: Tolerances tight enough that the moving-horizon solve is limited by the
#: arithmetic rather than by the solver stopping early, required wherever a
#: test asserts exact agreement with a Kalman filter.
_EXACT_OPTS = {"ipopt": {"tol": 1e-12, "acceptable_tol": 1e-12, "max_iter": 500}}


def _damped_scalar_f():
    """Scalar plant with a pole inside the unit circle.

    Not an integrator: an output bias on an integrator is undetectable (see
    ``test_output_bias_on_an_integrator_is_undetectable``), so the default
    ``dy_index`` would be rejected at construction.
    """
    x = cs.MX.sym("x", 1)
    u = cs.MX.sym("u", 1)
    return cs.Function("f", [x, u], [0.8 * x + 0.2 * u])


class TestMovingHorizonEstimator:
    """Bias and state estimation over a window, with bounds the filters cannot honour."""

    # ------------------------------------------------------------------
    # exact equivalences
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", [1, 3, 8])
    def test_exact_equivalence_with_augmented_ekf_on_a_linear_model(self, horizon):
        """MHE with the exact arrival cost reproduces the AEKF on a linear model.

        With no active bounds and an arrival cost carrying the correctly lagged
        prior, moving-horizon estimation is full-information estimation, which
        for a linear model is the Kalman recursion, at any horizon. Asserted
        after every predict and every update, from the first cycle on, so the
        window-fill phase is covered too.

        This is the load-bearing test of the whole formulation: it fails if the
        arrival cost is lagged wrongly, if an in-window measurement is counted
        twice, if U or Y are off by one, if a covariance is used where its
        inverse belongs, or if the augmentation differs from the filter's.
        """
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)
        C = np.eye(2)

        shared = {
            "du_index": [0],
            "dy_index": [0],
            "Q_x": np.eye(2) * 0.05,
            "Q_du": np.eye(1) * 0.02,
            "Q_dy": np.eye(1) * 0.03,
            "R": np.eye(2) * 0.1,
            "x0": np.array([0.3, -0.2]),
            "P0": np.eye(4) * 0.7,
        }
        aekf = AugmentedExtendedKalmanFilter(f, C, **shared)
        mhe = MovingHorizonEstimator(
            f, C, horizon=horizon, solver_opts=_EXACT_OPTS, **shared
        )

        rng = np.random.default_rng(0)
        for _ in range(15):
            u = rng.normal(size=(1, 1))
            y = rng.normal(size=(2, 1))

            aekf.predict(u)
            mhe.predict(u)
            assert_allclose(mhe.z_est, aekf.z_est, atol=1e-6)

            aekf.update(y)
            mhe.update(y)
            assert_allclose(mhe.z_est, aekf.z_est, atol=1e-6)
            assert_allclose(mhe.du_bias_est, aekf.du_bias_est, atol=1e-6)
            assert_allclose(mhe.dy_bias_est, aekf.dy_bias_est, atol=1e-6)

    def test_equivalence_holds_during_window_fill(self):
        """The equivalence must hold while the window is still filling.

        The masked prefix, the one-hot arrival slot and the right-alignment all
        only do anything for ``t < horizon``; asserting from cycle 0 with a long
        horizon is what exercises them.
        """
        Ad = np.array([[0.85, 0.05], [0.0, 0.9]])
        Bd = np.array([[0.1], [0.2]])
        f = _make_linear_f(Ad, Bd)
        C = np.eye(2)

        shared = {
            "dy_index": [0, 1],
            "Q_x": np.eye(2) * 0.02,
            "Q_dy": np.eye(2) * 0.01,
            "R": np.eye(2) * 0.2,
            "P0": np.eye(4) * 0.4,
        }
        aekf = AugmentedExtendedKalmanFilter(f, C, **shared)
        mhe = MovingHorizonEstimator(
            f, C, horizon=8, solver_opts=_EXACT_OPTS, **shared
        )

        rng = np.random.default_rng(11)
        for k in range(8):
            u = rng.normal(size=(1, 1))
            y = rng.normal(size=(2, 1))
            aekf.predict(u)
            mhe.predict(u)
            aekf.update(y)
            mhe.update(y)
            assert mhe.window_fill == k + 1
            assert not mhe.is_window_full
            assert_allclose(mhe.z_est, aekf.z_est, atol=1e-6)

    def test_no_bias_channels_matches_plain_ekf(self):
        """With no bias channels the estimator reduces to the plain EKF.

        Pins the degenerate ``n_aug == nx`` path, where both scatter matrices
        are empty.
        """
        Ad = np.array([[0.8, 0.1], [0.0, 0.9]])
        Bd = np.array([[0.0], [0.2]])
        f = _make_linear_f(Ad, Bd)
        C = np.eye(2)
        Q = np.eye(2) * 0.05
        R = np.eye(2) * 0.1

        mhe = MovingHorizonEstimator(
            f,
            C,
            horizon=4,
            du_index=[],
            dy_index=[],
            Q_x=Q,
            R=R,
            x0=[0.3, -0.2],
            P0=np.eye(2) * 0.7,
            solver_opts=_EXACT_OPTS,
        )
        ekf = ExtendedKalmanFilter(f, C, Q=Q, R=R, x0=[0.3, -0.2], P0=np.eye(2) * 0.7)

        assert mhe.n_aug == mhe.nx == 2
        assert mhe.n_du == 0 and mhe.n_dy == 0

        rng = np.random.default_rng(3)
        for _ in range(12):
            u = rng.normal(size=(1, 1))
            y = rng.normal(size=(2, 1))
            mhe.predict(u)
            ekf.predict(u)
            mhe.update(y)
            ekf.update(y)
            assert_allclose(mhe.x_est, ekf.x_est, atol=1e-6)

    # ------------------------------------------------------------------
    # hand-computed arithmetic
    # ------------------------------------------------------------------

    def test_hand_computed_scalar_arrival_and_measurement_blend(self):
        """One filled slot is a closed-form precision-weighted blend.

        With ``f(x, u) = x``, no bias channels, a constant arrival weight and a
        single measurement, the objective is
        ``(x - x_bar)^2 / p + (y - x)^2 / r`` whose minimizer is the
        precision-weighted mean. Isolates the covariance-versus-weight
        convention: an inverted Q/R would move this number.
        """
        x = cs.MX.sym("x", 1)
        u = cs.MX.sym("u", 1)
        f = cs.Function("f", [x, u], [x])

        p, r = 0.7, 0.25
        x_bar, y0 = 0.4, 1.6
        mhe = MovingHorizonEstimator(
            f,
            np.array([[1.0]]),
            horizon=1,
            dy_index=[],
            Q_x=np.array([[0.05]]),
            R=np.array([[r]]),
            x0=[x_bar],
            P0=np.array([[p]]),
            arrival_cost="constant",
            solver_opts=_EXACT_OPTS,
        )

        mhe.predict([0.0])
        mhe.update([y0])

        expected = (x_bar / p + y0 / r) / (1.0 / p + 1.0 / r)
        assert_allclose(mhe.x_est, [[expected]], atol=1e-9)

    def test_hand_computed_two_slot_blend_pins_the_process_weight(self):
        """A second cycle brings in the process residual, still in closed form.

        With ``f(x, u) = x`` the window minimizes
        ``(x0 - x_bar)^2/p + (y0 - x0)^2/r + (x1 - x0)^2/q + (y1 - x1)^2/r``,
        a quadratic in (x0, x1) whose stationary point is a 2x2 solve. This is
        the term ``Q`` weights, which the single-slot test cannot see.
        """
        x = cs.MX.sym("x", 1)
        u = cs.MX.sym("u", 1)
        f = cs.Function("f", [x, u], [x])

        p, q, r = 0.7, 0.3, 0.25
        x_bar, y0, y1 = 0.4, 1.6, -0.2
        mhe = MovingHorizonEstimator(
            f,
            np.array([[1.0]]),
            horizon=1,
            dy_index=[],
            Q_x=np.array([[q]]),
            R=np.array([[r]]),
            x0=[x_bar],
            P0=np.array([[p]]),
            arrival_cost="constant",
            solver_opts=_EXACT_OPTS,
        )
        mhe.predict([0.0])
        mhe.update([y0])
        mhe.predict([0.0])
        mhe.update([y1])

        # The arrival anchor after the shift is the previous solution's slot 1,
        # which for horizon=1 is the estimate the first cycle produced.
        x_bar2 = (x_bar / p + y0 / r) / (1.0 / p + 1.0 / r)
        # d/dx0: 2(x0 - x_bar2)/p + 2(x0 - y0)/r + 2(x0 - x1)/q = 0
        # d/dx1: 2(x1 - x0)/q + 2(x1 - y1)/r = 0
        A = np.array([[1 / p + 1 / r + 1 / q, -1 / q], [-1 / q, 1 / q + 1 / r]])
        b = np.array([x_bar2 / p + y0 / r, y1 / r])
        expected = np.linalg.solve(A, b)

        assert_allclose(mhe.x_est, [[expected[1]]], atol=1e-9)
        assert_allclose(mhe.x_traj.ravel(), expected, atol=1e-9)

    def test_covariance_semantics_are_covariances_not_weights(self):
        """A small R means a trusted measurement, not a discounted one."""
        f = _damped_scalar_f()
        C = np.array([[1.0]])
        shared = {
            "dy_index": [],
            "Q_x": np.array([[0.1]]),
            "x0": [0.0],
            "P0": np.array([[0.5]]),
        }
        tight = MovingHorizonEstimator(
            f, C, horizon=3, R=np.array([[1e-4]]), **shared
        )
        loose = MovingHorizonEstimator(
            f, C, horizon=3, R=np.array([[1e2]]), **shared
        )

        y = 1.0
        for _ in range(6):
            tight.predict([0.0])
            loose.predict([0.0])
            tight.update([y])
            loose.update([y])

        assert abs(float(tight.x_est.item()) - y) < abs(float(loose.x_est.item()) - y)

    # ------------------------------------------------------------------
    # what a Kalman filter cannot do
    # ------------------------------------------------------------------

    def test_bounds_clip_the_estimate(self):
        """Bounds constrain the estimate; the filter has no way to honour them."""
        f = _damped_scalar_f()
        C = np.array([[1.0]])
        shared = {
            "dy_index": [0],
            "Q_x": np.array([[0.1]]),
            "Q_dy": np.array([[1e-4]]),
            "R": np.array([[0.05]]),
            "x0": [0.5],
            "P0": np.eye(2) * 0.5,
        }
        mhe = MovingHorizonEstimator(f, C, horizon=5, x_lb=[0.0], **shared)
        aekf = AugmentedExtendedKalmanFilter(f, C, **shared)

        for _ in range(30):
            mhe.predict([0.0])
            aekf.predict([0.0])
            mhe.update([-0.5])
            aekf.update([-0.5])

        # bound_relax_factor is pinned to 0, so this is a hard inequality
        assert float(mhe.x_est.item()) >= 0.0
        assert float(aekf.x_est.item()) < 0.0

    def test_bias_bounds_are_full_width_and_gathered(self):
        """Bias bounds come in at full signal width, like the bias seeds."""
        Ad = np.array([[0.8, 0.0], [0.0, 0.85]])
        Bd = np.array([[0.2, 0.0], [0.0, 0.3]])
        f = _make_linear_f(Ad, Bd)
        C = np.eye(2)

        cap = 0.05
        mhe = MovingHorizonEstimator(
            f,
            C,
            horizon=4,
            du_index=[1],
            dy_index=[],
            Q_x=np.eye(2) * 0.05,
            Q_du=np.eye(1) * 0.1,
            R=np.eye(2) * 0.01,
            # full input width; the entry for the unselected channel 0 is ignored
            du_bias_lb=[-99.0, -cap],
            du_bias_ub=[99.0, cap],
        )

        for _ in range(20):
            mhe.predict([0.0, 0.0])
            mhe.update([0.0, 2.0])

        du = mhe.du_bias_est
        assert du.shape == (2, 1)
        assert du[0, 0] == 0.0  # unselected channel stays exactly zero
        assert abs(du[1, 0]) <= cap + 1e-9
        assert abs(du[1, 0]) > 0.5 * cap  # the cap is what is binding

        with pytest.raises(ValueError, match="du_bias_lb must have 2 elements"):
            MovingHorizonEstimator(f, C, du_index=[1], dy_index=[], du_bias_lb=[0.0])

    def test_input_bias_recovers_actuator_offset(self):
        """An unmeasured actuator offset is recovered through the dynamics."""
        x = cs.MX.sym("x", 2)
        u = cs.MX.sym("u", 1)
        dt = 0.1
        x_next = cs.vertcat(
            x[0] + dt * x[1],
            x[1] + dt * (-0.5 * x[0] - 0.2 * x[1] ** 3 + u),
        )
        f = cs.Function("f", [x, u], [x_next])
        C = np.eye(2)

        mhe = MovingHorizonEstimator(
            f,
            C,
            horizon=6,
            du_index=[0],
            dy_index=[],
            Q_x=np.eye(2) * 1e-4,
            Q_du=np.eye(1) * 1e-3,
            R=np.eye(2) * 1e-4,
        )

        true_offset = 0.3
        x_true = np.zeros((2, 1))
        for _ in range(80):
            u_cmd = np.array([[0.4]])
            mhe.predict(u_cmd)
            x_true = np.asarray(f(x_true, u_cmd + true_offset), dtype=float)
            mhe.update(x_true)

        assert abs(float(mhe.du_bias_est[0, 0]) - true_offset) < 0.05

    def test_output_bias_estimation(self):
        """A constant sensor offset lands in the output bias, not the state."""
        x = cs.MX.sym("x", 2)
        u = cs.MX.sym("u", 1)
        dt = 0.1
        x_next = cs.vertcat(
            x[0] + dt * (-0.5 * x[0] + x[1]),
            x[1] + dt * (-0.8 * x[1] + u),
        )
        f = cs.Function("f", [x, u], [x_next])
        C = np.eye(2)

        mhe = MovingHorizonEstimator(
            f,
            C,
            horizon=6,
            dy_index=[0],
            Q_x=np.eye(2) * 1e-4,
            Q_dy=np.eye(1) * 1e-3,
            R=np.eye(2) * 1e-4,
        )

        sensor_offset = 0.25
        x_true = np.zeros((2, 1))
        for _ in range(80):
            u_cmd = np.array([[0.5]])
            mhe.predict(u_cmd)
            x_true = np.asarray(f(x_true, u_cmd), dtype=float)
            y = x_true.copy()
            y[0, 0] += sensor_offset
            mhe.update(y)

        assert abs(float(mhe.dy_bias_est[0, 0]) - sensor_offset) < 0.05
        assert_allclose(mhe.x_est, x_true, atol=0.05)

    # ------------------------------------------------------------------
    # delegation to the companion filter
    # ------------------------------------------------------------------

    def test_over_budget_bias_selection_raises(self):
        """The detectability budget is the companion's, message and all."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.1, 0.0], [0.0, 0.2]])
        f = _make_linear_f(Ad, Bd)
        C = np.array([[1.0, 0.0]])  # one measurement -> budget of 1

        with pytest.raises(ValueError, match="3 bias states were requested"):
            MovingHorizonEstimator(f, C, du_index=[0, 1], dy_index=[0])

    def test_undetectable_selection_raises(self):
        """A within-budget but unobservable split is still rejected."""
        Ad = np.array([[1.0, 0.0], [0.0, 0.95]])
        Bd = np.array([[0.1], [0.2]])
        f = _make_linear_f(Ad, Bd)
        C = np.array([[1.0, 0.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="not detectable"):
            MovingHorizonEstimator(f, C, du_index=[], dy_index=[0])

    def test_detectability_report_matches_the_aekf(self):
        """The report is delegated, not reimplemented."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)
        C = np.eye(2)

        mhe = MovingHorizonEstimator(f, C, horizon=3, du_index=[0], dy_index=[0])
        aekf = AugmentedExtendedKalmanFilter(f, C, du_index=[0], dy_index=[0])
        assert mhe.detectability_report() == aekf.detectability_report()

    def test_check_detectability_false_builds(self):
        """The rank test can be waived, exactly as on the filter."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.1, 0.0], [0.0, 0.2]])
        f = _make_linear_f(Ad, Bd)
        C = np.array([[1.0, 0.0]])

        mhe = MovingHorizonEstimator(
            f, C, horizon=2, du_index=[0, 1], dy_index=[0], check_detectability=False
        )
        assert mhe.n_aug == 5
        assert mhe.detectability_report()["detectable"] is False

    def test_no_bias_report_is_vacuous(self):
        """With nothing augmented the condition has nothing to say."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        mhe = MovingHorizonEstimator(
            f, np.eye(2), horizon=2, du_index=[], dy_index=[]
        )
        assert mhe.detectability_report() == {
            "n_bias": 0,
            "max_bias_states": 2,
            "baseline_rank": 2,
            "rank": 2,
            "gained": 0,
            "required": 2,
            "deficiency": 0,
            "plant_rank_deficiency": 0,
            "detectable": True,
            "redundant_du": (),
            "redundant_dy": (),
        }

    # ------------------------------------------------------------------
    # contract and plumbing
    # ------------------------------------------------------------------

    def test_initialization_default(self):
        """Defaults mirror the augmented filter's, plus an empty window."""
        Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
        Bd = np.array([[0.0], [0.1]])
        f = _make_linear_f(Ad, Bd)

        mhe = MovingHorizonEstimator(f, np.eye(2), horizon=5)

        assert mhe.nx == 2
        assert mhe.nu == 1
        assert mhe.ny == 2
        assert mhe.du_index == ()
        assert mhe.dy_index == (0, 1)
        assert mhe.n_du == 0
        assert mhe.n_dy == 2
        assert mhe.n_aug == 4
        assert mhe.horizon == 5
        assert mhe.arrival_cost == "ekf"
        assert mhe.window_fill == 0
        assert mhe.is_window_full is False
        assert mhe.last_success is None
        assert mhe.last_status is None
        assert mhe.last_cost is None
        assert mhe.n_solver_failures == 0
        assert_allclose(mhe.x_est, np.zeros((2, 1)))
        assert_allclose(mhe.P, np.eye(4))

    def test_biases_are_full_width_and_zero_off_selection(self):
        """Bias estimates come out at full signal width."""
        Ad = np.array([[0.8, 0.0], [0.0, 0.85]])
        Bd = np.array([[0.2, 0.0], [0.0, 0.3]])
        f = _make_linear_f(Ad, Bd)

        mhe = MovingHorizonEstimator(
            f, np.eye(2), horizon=3, du_index=[1], dy_index=[0]
        )
        assert mhe.du_bias_est.shape == (2, 1)
        assert mhe.dy_bias_est.shape == (2, 1)
        assert mhe.z_est.shape == (4, 1)  # compact: nx + 1 + 1

        mhe.predict([0.5, -0.5])
        mhe.update([0.1, 0.2])
        assert mhe.du_bias_est[0, 0] == 0.0
        assert mhe.dy_bias_est[1, 0] == 0.0

    def test_get_mpc_biases_returns_copies(self):
        """The returned dict must not alias internal state."""
        f = _damped_scalar_f()
        mhe = MovingHorizonEstimator(f, np.array([[1.0]]), horizon=2)

        biases = mhe.get_mpc_biases()
        assert set(biases) == {"du_bias", "dy_bias"}
        assert biases["du_bias"].shape == (1, 1)
        assert biases["dy_bias"].shape == (1, 1)

        biases["dy_bias"][0, 0] = 99.0
        assert mhe.get_mpc_biases()["dy_bias"][0, 0] != 99.0

    def test_reset_round_trips_the_estimate_but_not_the_window(self):
        """A reset restores the estimate and deliberately drops the window."""
        f = _damped_scalar_f()
        C = np.array([[1.0]])
        mhe = MovingHorizonEstimator(f, C, horizon=3, dy_index=[0])

        for _ in range(6):
            mhe.predict([0.5])
            mhe.update([0.9])
        assert mhe.window_fill == 4

        mhe.reset(x0=[0.25], dy_bias0=[0.1], P0=np.eye(2) * 0.3)

        assert_allclose(mhe.x_est, [[0.25]])
        assert_allclose(mhe.dy_bias_est, [[0.1]])
        assert_allclose(mhe.P, np.eye(2) * 0.3)
        # the window is not transferred state; a reset means the past is gone
        assert mhe.window_fill == 0
        assert mhe.is_window_full is False
        assert mhe.z_traj.shape == (2, 0)

    def test_reset_defaults_to_zeros_and_identity(self):
        f = _damped_scalar_f()
        mhe = MovingHorizonEstimator(f, np.array([[1.0]]), horizon=2, x0=[3.0])

        mhe.reset()
        assert_allclose(mhe.z_est, np.zeros((2, 1)))
        assert_allclose(mhe.P, np.eye(2))

    def test_z_traj_shape_and_alignment(self):
        """The trajectory holds only filled slots, newest last."""
        f = _damped_scalar_f()
        C = np.array([[1.0]])
        mhe = MovingHorizonEstimator(f, C, horizon=3, dy_index=[0])

        assert mhe.z_traj.shape == (2, 0)
        for k in range(6):
            mhe.predict([0.5])
            mhe.update([0.9])
            expected = min(k + 1, 4)
            assert mhe.z_traj.shape == (2, expected)
            assert mhe.x_traj.shape == (1, expected)
            assert_allclose(mhe.z_traj[:, -1:], mhe.z_est, atol=1e-9)
        assert mhe.is_window_full

    def test_accepts_casadi_dm_inputs(self):
        f = _damped_scalar_f()
        mhe = MovingHorizonEstimator(f, np.array([[1.0]]), horizon=2, dy_index=[0])

        mhe.predict(cs.DM([0.5]))
        mhe.update(cs.DM([0.9]))
        assert np.all(np.isfinite(mhe.x_est))

    def test_multi_output_dynamics_accepted(self):
        """A multi-output f is normalized to its first output, as on the EKF."""
        x = cs.MX.sym("x", 2)
        u = cs.MX.sym("u", 1)
        x_next = cs.DM([[0.9, 0.1], [0.0, 0.95]]) @ x + cs.DM([[0.0], [0.1]]) @ u
        f = cs.Function("f", [x, u], [x_next, cs.sumsqr(x)])

        mhe = MovingHorizonEstimator(f, np.eye(2), horizon=2)
        assert mhe.nx == 2
        mhe.predict([0.1])
        mhe.update([0.2, 0.3])
        assert np.all(np.isfinite(mhe.x_est))

    # ------------------------------------------------------------------
    # failure modes
    # ------------------------------------------------------------------

    def test_solver_failure_falls_back_to_the_companion_filter(self):
        """A failed solve must not stall a control loop."""
        f = _damped_scalar_f()
        C = np.array([[1.0]])
        shared = {
            "dy_index": [0],
            "Q_x": np.array([[0.1]]),
            "Q_dy": np.array([[1e-4]]),
            "R": np.array([[0.05]]),
            "x0": [0.5],
            "P0": np.eye(2) * 0.5,
        }
        mhe = MovingHorizonEstimator(
            f, C, horizon=3, solver_opts={"ipopt": {"max_iter": 0}}, **shared
        )
        aekf = AugmentedExtendedKalmanFilter(f, C, **shared)

        mhe.predict([0.0])
        aekf.predict([0.0])
        mhe.update([0.9])
        aekf.update([0.9])

        assert mhe.last_success is False
        assert mhe.n_solver_failures == 1
        assert mhe.last_cost is None
        assert_allclose(mhe.z_est, aekf.z_est)

    def test_solver_failure_raises_when_asked(self):
        f = _damped_scalar_f()
        mhe = MovingHorizonEstimator(
            f,
            np.array([[1.0]]),
            horizon=3,
            dy_index=[0],
            on_solver_failure="raise",
            solver_opts={"ipopt": {"max_iter": 0}},
        )
        mhe.predict([0.0])
        with pytest.raises(RuntimeError, match="the moving-horizon NLP failed"):
            mhe.update([0.9])

    def test_masked_slots_do_not_poison_the_objective(self):
        """Unfilled slots are pinned where the model is defined.

        Pinning them at the origin would be the obvious choice and would make
        this model return NaN from cycle 0, because the mask is applied to the
        residual after ``f`` has already been evaluated.
        """
        x = cs.MX.sym("x", 1)
        u = cs.MX.sym("u", 1)
        f = cs.Function("f", [x, u], [cs.sqrt(x) + 0.1 * u])

        mhe = MovingHorizonEstimator(
            f,
            np.array([[1.0]]),
            horizon=6,
            dy_index=[],
            Q_x=np.array([[0.1]]),
            R=np.array([[0.1]]),
            x0=[1.0],
            x_lb=[1e-3],
        )
        mhe.predict([0.5])
        mhe.update([1.1])

        assert np.all(np.isfinite(mhe.x_est))
        assert mhe.last_success is True
        assert float(mhe.x_est.item()) >= 1e-3

    def test_update_without_predict_raises(self):
        f = _damped_scalar_f()
        mhe = MovingHorizonEstimator(f, np.array([[1.0]]), horizon=2, dy_index=[0])
        with pytest.raises(RuntimeError, match="without a preceding predict"):
            mhe.update([0.5])

    def test_singular_process_noise_raises(self):
        """A covariance the filters tolerate is rejected here, by name.

        The moving-horizon cost weights by the inverse, so a singular block is
        an infinite weight. This is the one migration trap going from the
        augmented filter to this class, so the asymmetry is asserted directly.
        """
        f = _damped_scalar_f()
        C = np.array([[1.0]])

        with pytest.raises(ValueError, match="Q_x must be positive definite"):
            MovingHorizonEstimator(f, C, horizon=2, Q_x=np.zeros((1, 1)))

        with pytest.raises(ValueError, match="Q_dy must be positive definite"):
            MovingHorizonEstimator(
                f, C, horizon=2, dy_index=[0], Q_dy=np.zeros((1, 1))
            )

        # the same tuning is perfectly legal on the filter
        AugmentedExtendedKalmanFilter(f, C, Q_x=np.zeros((1, 1)))

    def test_non_symmetric_covariance_raises(self):
        Ad = np.array([[0.8, 0.1], [0.0, 0.9]])
        Bd = np.array([[0.0], [0.2]])
        f = _make_linear_f(Ad, Bd)
        with pytest.raises(ValueError, match="Q_x must be symmetric"):
            MovingHorizonEstimator(
                f, np.eye(2), horizon=2, Q_x=np.array([[1.0, 0.5], [0.0, 1.0]])
            )

    def test_ill_conditioned_covariance_warns(self):
        Ad = np.array([[0.8, 0.1], [0.0, 0.9]])
        Bd = np.array([[0.0], [0.2]])
        f = _make_linear_f(Ad, Bd)
        with pytest.warns(UserWarning, match="ill-conditioned"):
            MovingHorizonEstimator(
                f, np.eye(2), horizon=2, Q_x=np.diag([1.0, 1e-12])
            )

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "3"])
    def test_horizon_must_be_positive(self, bad):
        f = _damped_scalar_f()
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            MovingHorizonEstimator(f, np.array([[1.0]]), horizon=bad)

    def test_crossed_bounds_raise(self):
        f = _damped_scalar_f()
        with pytest.raises(ValueError, match="x_lb must be <= x_ub"):
            MovingHorizonEstimator(
                f, np.array([[1.0]]), horizon=2, x_lb=[1.0], x_ub=[0.0]
            )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"arrival_cost": "bogus"}, "arrival_cost must be"),
            ({"on_solver_failure": "bogus"}, "on_solver_failure must be"),
            ({"solver": "bogus"}, "unknown NLP solver"),
        ],
    )
    def test_invalid_mode_arguments_raise(self, kwargs, match):
        f = _damped_scalar_f()
        with pytest.raises(ValueError, match=match):
            MovingHorizonEstimator(f, np.array([[1.0]]), horizon=2, **kwargs)

    def test_solution_is_independent_of_the_initial_guess(self):
        """A linear model makes the problem strictly convex, so it has one answer.

        Guards against a warm start quietly deciding the estimate.
        """
        Ad = np.array([[0.85, 0.05], [0.0, 0.9]])
        Bd = np.array([[0.1], [0.2]])
        f = _make_linear_f(Ad, Bd)
        C = np.eye(2)
        shared = {
            "horizon": 5,
            "du_index": [],
            "dy_index": [],
            "Q_x": np.eye(2) * 0.05,
            "R": np.eye(2) * 0.1,
            "P0": np.eye(2) * 1e4,  # a deliberately weak prior
            "arrival_cost": "constant",
            "solver_opts": _EXACT_OPTS,
        }
        near = MovingHorizonEstimator(f, C, x0=[0.0, 0.0], **shared)
        far = MovingHorizonEstimator(f, C, x0=[50.0, -80.0], **shared)

        rng = np.random.default_rng(7)
        for _ in range(12):
            u = rng.normal(size=(1, 1))
            y = rng.normal(size=(2, 1))
            near.predict(u)
            far.predict(u)
            near.update(y)
            far.update(y)

        assert_allclose(near.x_est, far.x_est, atol=1e-6)

    def test_retune_preserves_the_window_and_sharpens_the_gain(self):
        """Retuning rewrites weights in place; the window survives."""
        f = _damped_scalar_f()
        C = np.array([[1.0]])
        mhe = MovingHorizonEstimator(
            f,
            C,
            horizon=5,
            dy_index=[0],
            Q_x=np.array([[0.1]]),
            Q_dy=np.array([[1e-4]]),
            R=np.array([[0.5]]),
            x0=[0.5],
            P0=np.eye(2) * 0.5,
        )
        for _ in range(12):
            mhe.predict([1.0])
            mhe.update([0.9])

        fill_before = mhe.window_fill
        assert mhe.is_window_full

        mhe.retune(R=np.array([[0.005]]))
        assert mhe.window_fill == fill_before

        before = float(mhe.x_est.item())
        mhe.predict([1.0])
        mhe.update([2.5])
        after = float(mhe.x_est.item())
        # a hundredfold tighter R must pull the estimate hard toward the reading
        assert after - before > 0.5 * (2.5 - before)

    def test_retune_rejects_a_bad_covariance_without_changing_state(self):
        f = _damped_scalar_f()
        mhe = MovingHorizonEstimator(
            f, np.array([[1.0]]), horizon=3, dy_index=[0], R=np.array([[0.5]])
        )
        for _ in range(4):
            mhe.predict([1.0])
            mhe.update([0.9])
        before = mhe.z_est.copy()

        with pytest.raises(ValueError, match="R must be positive definite"):
            mhe.retune(R=np.zeros((1, 1)))

        assert_allclose(mhe.z_est, before)
        assert_allclose(mhe._R, np.array([[0.5]]))


class TestIntegration:
    """Estimators at the dimensions the grinding-circuit example runs at."""

    def test_grinding_circuit_dimensions(self):
        """The filters run at the grinding circuit's 4x4 dimensions."""
        nx = 20  # Typical state dimension for grinding circuit
        nu = 4
        ny = 4
        rng = np.random.default_rng(3)

        Ad = np.eye(nx) * 0.95
        Bd = rng.standard_normal((nx, nu)) * 0.1
        Cd = rng.standard_normal((ny, nx)) * 0.5
        Dd = np.zeros((ny, nu))

        # the full budget of ny bias states, split across inputs and outputs
        kf = AugmentedKalmanFilter(
            Ad,
            Bd,
            Cd,
            Dd=Dd,
            du_index=[0, 1],
            dy_index=[0, 1],
            Q_x=np.eye(nx) * 0.1,
            Q_du=np.eye(2) * 0.01,
            Q_dy=np.eye(2) * 0.01,
            R=np.eye(ny) * 1.0,
        )

        assert kf.n_aug == nx + 2 + 2

        # Run a few cycles
        for _ in range(10):
            u = np.random.randn(nu, 1)
            y = np.random.randn(ny, 1)
            kf.predict(u=u)
            kf.update(y=y, u=u)

        biases = kf.get_mpc_biases()
        assert biases["du_bias"].shape == (nu, 1)
        assert biases["dy_bias"].shape == (ny, 1)

    def test_mhe_grinding_circuit_dimensions(self):
        """Moving-horizon estimation at grinding-circuit dimensions (4x4 MIMO).

        Deliberately a smaller nx than the filter's version of this test: the
        moving-horizon problem carries ``n_aug * (horizon + 1)`` variables, so a
        state dimension that is free for a filter is not free here.
        """
        nx, nu, ny, horizon = 8, 4, 4, 10
        rng = np.random.default_rng(5)

        Ad = np.eye(nx) * 0.95
        Bd = rng.normal(size=(nx, nu)) * 0.1
        Cd = np.zeros((ny, nx))
        Cd[:, :ny] = np.eye(ny)  # the first four states are measured
        f = _make_linear_f(Ad, Bd)

        mhe = MovingHorizonEstimator(
            f,
            Cd,
            horizon=horizon,
            du_index=[],
            dy_index=[0, 1, 2, 3],
            Q_x=np.eye(nx) * 0.1,
            Q_dy=np.eye(ny) * 0.01,
            R=np.eye(ny) * 1.0,
        )

        assert mhe.n_aug == nx + ny
        assert mhe.horizon == horizon

        for _ in range(10):
            mhe.predict(rng.normal(size=(nu, 1)))
            mhe.update(rng.normal(size=(ny, 1)))

        assert np.all(np.isfinite(mhe.x_est))
        assert mhe.x_est.shape == (nx, 1)
        assert mhe.n_solver_failures == 0
        assert mhe.window_fill == 10
        assert mhe.x_traj.shape == (nx, 10)

        biases = mhe.get_mpc_biases()
        assert biases["du_bias"].shape == (nu, 1)
        assert biases["dy_bias"].shape == (ny, 1)

    def test_with_mimo_tf2ss(self):
        """A ``mimo_tf2ss`` realization feeds the filter directly."""
        try:
            from neuralmpcx.util.control import TransferFunctionTerm, mimo_tf2ss
        except ImportError:
            pytest.skip("Control module not available")

        # Simple 2x2 system
        TF = TransferFunctionTerm
        G = {
            (0, 0): [TF(gain=1.0, time_constants=[1.0])],
            (0, 1): [TF(gain=0.5, time_constants=[2.0])],
            (1, 0): [TF(gain=0.3, time_constants=[1.5])],
            (1, 1): [TF(gain=2.0, time_constants=[3.0])],
        }
        ss = mimo_tf2ss(G, ny=2, nu=2, Ts=0.1)

        kf = AugmentedKalmanFilter(
            Ad=ss.Ad,
            Bd=ss.Bd,
            Cd=ss.Cd,
            Dd=ss.Dd,
            Q_x=np.eye(ss.nx) * 0.1,
            Q_dy=np.eye(ss.ny) * 0.01,
            R=np.eye(ss.ny) * 1.0,
        )

        assert kf.nx == ss.nx
        assert kf.nu == ss.nu
        assert kf.ny == ss.ny

        # Test predict-update cycle
        u = np.ones((ss.nu, 1))
        y = np.zeros((ss.ny, 1))

        kf.predict(u=u)
        kf.update(y=y, u=u)

        biases = kf.get_mpc_biases()
        assert "du_bias" in biases
        assert "dy_bias" in biases
