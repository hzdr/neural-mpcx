"""Input/output equivalence and detectability of reduced integrating models."""

import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import cont2discrete

from examples.MPC_Grinding_Circuit.mpc_grinding_circuit import grinding_model
from neuralmpcx.util.control import (
    TransferFunctionTerm,
    _remove_unobservable_integrators,
    mimo_tf2ss,
)
from neuralmpcx.util.estimators import AugmentedKalmanFilter, bias_detectability


def output_response(A, B, C, D, actions):
    """Simulate zero-state outputs with the standard discrete-time convention."""
    state = np.zeros(A.shape[0])
    outputs = []
    for action in actions:
        outputs.append(C @ state + D @ action)
        state = A @ state + B @ action
    return np.asarray(outputs)


class TestIntegratorRealization(unittest.TestCase):
    def test_parallel_integrators_match_analytic_response(self):
        gains = np.array([0.001, 0.011, 0.032, -0.031])
        G = {
            (0, j): [TransferFunctionTerm(gain=gain, has_integrator=True)]
            for j, gain in enumerate(gains)
        }
        original = mimo_tf2ss(G, ny=1, nu=4, Ts=30.0, balanced=False)
        self.assertEqual(original.nx, 4)
        ss = mimo_tf2ss(
            G, ny=1, nu=4, Ts=30.0, balanced=False,
            remove_unobservable_integrators=True,
        )
        self.assertEqual(ss.nx, 1)
        assert_allclose(ss.Ad, [[1.0]], atol=1e-12)
        actions = np.random.default_rng(69).normal(size=(100, 4))
        expected = np.r_[0.0, np.cumsum(30.0 * (actions[:-1] @ gains))]
        actual = output_response(ss.Ad, ss.Bd, ss.Cd, ss.Dd, actions)
        assert_allclose(actual[:, 0], expected, rtol=1e-12, atol=1e-12)

    def test_delayed_mimo_response_is_preserved(self):
        TF = TransferFunctionTerm
        G = {
            (0, 0): [TF(gain=1.3, time_constants=[5.0])],
            (1, 0): [TF(gain=0.01, time_constants=[7.0], has_integrator=True)],
            (1, 1): [TF(gain=-0.02, has_integrator=True)],
            (1, 2): [TF(gain=0.03, delay=2.0, time_constants=[10.0], has_integrator=True)],
        }
        actions = np.random.default_rng(42).normal(size=(300, 3))
        for method in ("zoh", "tustin", "foh"):
            with self.subTest(method=method):
                original = mimo_tf2ss(G, 2, 3, Ts=0.5, method=method, balanced=False)
                reduced = mimo_tf2ss(
                    G, 2, 3, Ts=0.5, method=method, balanced=False,
                    remove_unobservable_integrators=True,
                )
                self.assertEqual(reduced.nx, original.nx - 2)
                expected = output_response(
                    original.Ad, original.Bd, original.Cd, original.Dd, actions
                )
                actual = output_response(
                    reduced.Ad, reduced.Bd, reduced.Cd, reduced.Dd, actions
                )
                assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_observable_and_stable_modes_are_retained(self):
        A = np.diag([-1.0, -2.0, 0.0])
        B = np.ones((3, 1))
        C = np.array([[1.0, 0.0, 1.0]])
        reduced = _remove_unobservable_integrators(A, B, C)
        for actual, expected in zip(reduced, (A, B, C)):
            assert_array_equal(actual, expected)

    def test_unobservable_integrator_chain_is_removed(self):
        A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        B = np.ones((3, 1))
        C = np.array([[0.0, 0.0, 1.0]])
        D = np.zeros((1, 1))
        Ar, Br, Cr = _remove_unobservable_integrators(A, B, C)
        self.assertEqual(Ar.shape, (1, 1))
        full = cont2discrete((A, B, C, D), 0.1)
        reduced = cont2discrete((Ar, Br, Cr, D), 0.1)
        actions = np.ones((100, 1))
        assert_allclose(
            output_response(*reduced[:4], actions),
            output_response(*full[:4], actions),
            atol=1e-12,
        )

    def test_zero_state_realization_and_optional_continuous_matrices(self):
        G = {(0, 0): [TransferFunctionTerm(gain=5.0)]}
        ss = mimo_tf2ss(
            G, 1, 1, Ts=0.1, balanced=False, store_continuous=False,
            remove_unobservable_integrators=True,
        )
        self.assertEqual(ss.nx, 0)
        self.assertIsNone(ss.A)
        self.assertIsNone(ss.B)
        self.assertIsNone(ss.C)
        self.assertIsNone(ss.D)
        assert_array_equal(ss.Dd, [[5.0]])

    def test_grinding_response_and_unit_pole_count(self):
        original = grinding_model(remove_unobservable_integrators=False)
        reduced = grinding_model()
        self.assertEqual(original.nx, 54)
        self.assertEqual(reduced.nx, 51)
        for model, expected in ((original, 4), (reduced, 1)):
            poles = np.linalg.eigvals(model.Ad)
            self.assertEqual(np.count_nonzero(np.abs(poles - 1.0) < 1e-10), expected)
        actions = np.random.default_rng(69).normal(size=(1000, 4))
        assert_allclose(
            output_response(reduced.Ad, reduced.Bd, reduced.Cd, reduced.Dd, actions),
            output_response(original.Ad, original.Bd, original.Cd, original.Dd, actions),
            rtol=1e-9, atol=1e-9,
        )

    def test_grinding_filter_is_detectable_without_warnings(self):
        ss = grinding_model()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kf = AugmentedKalmanFilter(
                ss.Ad, ss.Bd, ss.Cd, ss.Dd,
                du_index=[], dy_index=[0, 1, 2],
                Q_x=np.eye(ss.nx) * 0.1, Q_dy=np.eye(3) * 0.01, R=np.eye(4),
            )
        report = kf.detectability_report()
        self.assertTrue(report["detectable"])
        self.assertEqual(report["plant_rank_deficiency"], 0)
        self.assertEqual(report["rank"], report["required"])
        for bias in kf.get_mpc_biases().values():
            self.assertEqual(bias.shape, (4, 1))

    def test_sump_level_bias_is_still_indistinguishable(self):
        ss = grinding_model()
        report = bias_detectability(ss.Ad, ss.Bd, ss.Cd, dy_index=[0, 1, 2, 3])
        self.assertFalse(report["detectable"])
        self.assertEqual(report["redundant_dy"], (3,))


if __name__ == "__main__":
    unittest.main()
