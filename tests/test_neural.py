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

"""``CasadiLSTM``: the PyTorch LSTM re-expressed as a CasADi graph.

The reimplementation only earns its place if it computes the same function as
the network it was trained as, so the anchor test here runs a real
``torch.nn.LSTM`` alongside and compares outputs and final ``(h, c)``. The
rest covers the stateful path the MPC drives: the numeric warmup, the
no-anchor rollout, and the ``[u, d]`` core-input layout. Everything runs on
the ``hidden_size=8`` checkpoints.
"""

import casadi as cs
import numpy as np
import numpy.testing as npt
import pytest

torch = pytest.importorskip("torch")

from neuralmpcx.neural import CasadiLSTM  # noqa: E402


# ---------------------------------------------------------------------------
# CTS model config: n_inputs=1 (pump voltage), hidden_size=8, proj_size=1
# LSTM core sees n_inputs=1, h_out=1 (proj_size=1)
# Weight shapes: W_ih (32, 1), W_hh (32, 1), W_hr (1, 8)
# ---------------------------------------------------------------------------


def _rollout(model, u_seq, h0=None, c0=None):
    """Chain `_core.step_full` numerically over an input sequence.

    Returns stacked outputs with shape (T, h_out) plus the final per-layer
    hidden/cell state lists.
    """
    core = model._core
    h_list = (
        h0
        if h0 is not None
        else [np.zeros((core.h_out, 1)) for _ in range(core.num_layers)]
    )
    c_list = (
        c0
        if c0 is not None
        else [np.zeros((core.hidden_size, 1)) for _ in range(core.num_layers)]
    )
    ys = []
    for t in range(u_seq.shape[0]):
        y_t, h_list, c_list = core.step_full(u_seq[t].reshape(1, -1), h_list, c_list)
        ys.append(np.array(y_t).ravel())
    return np.vstack(ys), h_list, c_list


class TestCasadiLSTMConstruction:
    def test_dimensions(self):
        """The constructor derives output_size, sequence_length and h_out."""
        model = CasadiLSTM(
            n_context=2, n_inputs=1, hidden_size=8,
            horizon=5, proj_size=1,
        )
        assert model.output_size == 1
        assert model.sequence_length == 5  # horizon (5), no anchor column
        assert model.h_out == 1
        assert model._core.n_inputs == 1
        assert model._core.hidden_size == 8


class TestCasadiLSTMWeightLoading:
    def test_load_state_dict_cts(self, cts_state_dict):
        """Loading CTS model produces correct weight shapes."""
        model = CasadiLSTM(
            n_context=2, n_inputs=1, hidden_size=8,
            horizon=5, proj_size=1,
        )
        model.load_state_dict(cts_state_dict)

        core = model._core
        assert core.W_ih_val[0] is not None
        assert core.W_ih_val[0].shape == (32, 1)  # 4*8 x n_inputs
        assert core.W_hh_val[0].shape == (32, 1)  # 4*8 x h_out (proj_size=1)
        assert core.W_hr_val[0].shape == (1, 8)   # proj_size x hidden_size


class TestCasadiLSTMForward:
    def test_forward_deterministic(self, cts_state_dict):
        """Same input produces identical output on repeated calls."""
        model = CasadiLSTM(
            n_context=0, n_inputs=1, hidden_size=8,
            horizon=5, proj_size=1,
        )
        model.load_state_dict(cts_state_dict)

        rng = np.random.default_rng(42)
        u_np = rng.standard_normal((5, 1))

        y1, _, _ = _rollout(model, u_np)
        y2, _, _ = _rollout(model, u_np)
        npt.assert_array_equal(y1, y2)

    def test_forward_matches_pytorch(self, cts_state_dict):
        """Outputs and final (h, c) match a real ``torch.nn.LSTM`` to ``1e-5``.

        Everything else in the suite compares the CasADi graph against itself.
        This is the one place the reimplementation is checked against the
        network whose weights it loaded.
        """
        hidden_size = 8
        proj_size = 1
        n_inputs = 1  # control inputs only (u)

        # --- PyTorch reference ---
        pt_lstm = torch.nn.LSTM(
            input_size=n_inputs, hidden_size=hidden_size,
            num_layers=1, batch_first=True, proj_size=proj_size,
        )

        # Load weights (strip 'model.' prefix)
        pt_sd = {}
        for k, v in cts_state_dict.items():
            pt_sd[k.replace("model.", "")] = v
        pt_lstm.load_state_dict(pt_sd)
        pt_lstm.eval()

        # Random input: (batch=1, seq_len=5, n_inputs=1)
        rng = np.random.default_rng(123)
        u_np = rng.standard_normal((5, n_inputs)).astype(np.float32)
        u_torch = torch.from_numpy(u_np).unsqueeze(0)  # (1, 5, 1)

        h0 = torch.zeros(1, 1, proj_size)
        c0 = torch.zeros(1, 1, hidden_size)

        with torch.no_grad():
            pt_out, (pt_hn, pt_cn) = pt_lstm(u_torch, (h0, c0))
        pt_y = pt_out.squeeze(0).numpy()  # (5, proj_size)

        # --- CasADi ---
        core = CasadiLSTM(
            n_context=0, n_inputs=n_inputs, hidden_size=hidden_size,
            horizon=5, proj_size=proj_size,
        )
        core.load_state_dict(cts_state_dict)

        cs_y, cs_hN, cs_cN = _rollout(core, u_np.astype(np.float64))

        npt.assert_allclose(cs_y, pt_y.astype(np.float64), atol=1e-5)
        npt.assert_allclose(
            np.array(cs_hN[0]).ravel(), pt_hn.numpy().ravel(), atol=1e-5
        )
        npt.assert_allclose(
            np.array(cs_cN[0]).ravel(), pt_cn.numpy().ravel(), atol=1e-5
        )


class TestCasadiLSTMStateful:
    """Stateful path: explicit h0/c0 inputs and numeric helpers."""

    def _build_model(self, state_dict, n_context=2, horizon=5):
        model = CasadiLSTM(
            n_context=n_context, n_inputs=1, hidden_size=8,
            horizon=horizon, proj_size=1,
        )
        model.load_state_dict(state_dict)
        return model

    def test_step_full_matches_pytorch_step(self, cts_state_dict):
        """One step via step_full equals one PyTorch LSTM step (num_layers=1)."""
        model = self._build_model(cts_state_dict)
        h0 = [np.zeros((1, 1))]
        c0 = [np.zeros((8, 1))]
        u_t = np.array([[0.5]])

        # via step_full
        y_sf, h_sf, c_sf = model._core.step_full(u_t, h0, c0)

        # PyTorch reference: single step from zero states
        pt_lstm = torch.nn.LSTM(
            input_size=1, hidden_size=8, num_layers=1,
            batch_first=True, proj_size=1,
        )
        pt_sd = {k.replace("model.", ""): v for k, v in cts_state_dict.items()}
        pt_lstm.load_state_dict(pt_sd)
        pt_lstm.eval()
        with torch.no_grad():
            pt_out, (pt_hn, pt_cn) = pt_lstm(
                torch.tensor([[[0.5]]]),
                (torch.zeros(1, 1, 1), torch.zeros(1, 1, 8)),
            )

        npt.assert_allclose(
            np.array(y_sf).ravel(), pt_out.numpy().ravel(), atol=1e-5
        )
        npt.assert_allclose(
            np.array(h_sf[0]).ravel(), pt_hn.numpy().ravel(), atol=1e-5
        )
        npt.assert_allclose(
            np.array(c_sf[0]).ravel(), pt_cn.numpy().ravel(), atol=1e-5
        )

    def test_context_window_updates_hidden_state(self, cts_state_dict):
        """Different context windows produce different hidden states."""
        core_model = CasadiLSTM(
            n_context=0, n_inputs=1, hidden_size=8,
            horizon=2, proj_size=1,
        )
        core_model.load_state_dict(cts_state_dict)

        # Context 1: large input values
        u1 = np.array([[5.0], [5.0]])
        _, hn1, cn1 = _rollout(core_model, u1)

        # Context 2: small input values
        u2 = np.array([[0.1], [0.1]])
        _, hn2, cn2 = _rollout(core_model, u2)

        hn1_np = np.array(hn1[0]).flatten()
        hn2_np = np.array(hn2[0]).flatten()
        assert not np.allclose(hn1_np, hn2_np), \
            "Different inputs should produce different hidden states"

    def test_estimate_numeric_produces_finite_state(self, cts_state_dict):
        """estimate_numeric returns finite h/c with correct per-layer shapes."""
        model = self._build_model(cts_state_dict)
        rng = np.random.default_rng(0)
        u_ctx = rng.standard_normal((2, 1))
        y_ctx = rng.standard_normal((2, 1))

        h, c = model.estimate_numeric(u_ctx, y_ctx)
        assert isinstance(h, list) and len(h) == model.num_layers
        assert h[0].shape == (1, 1)
        assert c[0].shape == (8, 1)
        assert np.all(np.isfinite(np.array(h[0]))) and np.all(np.isfinite(np.array(c[0])))

    def test_estimate_numeric_seed_changes_output(self, cts_state_dict):
        """Different seed for (h, c) yields a different post-warmup state."""
        model = self._build_model(cts_state_dict)
        u_ctx = np.array([[0.3], [0.4]])
        y_ctx = np.array([[0.5], [0.6]])

        h_z, c_z = model.estimate_numeric(u_ctx, y_ctx)
        h_nz, c_nz = model.estimate_numeric(
            u_ctx, y_ctx,
            h_seed=[np.ones((1, 1)) * 0.2],
            c_seed=[np.ones((8, 1)) * 0.1],
        )
        # Cell state differs because the seed is propagated through the cell.
        assert not np.allclose(np.array(c_z[0]), np.array(c_nz[0]))

    def test_step_numeric_advances_cell_state(self, cts_state_dict):
        """One step_numeric call updates c (h is teacher-forced from y)."""
        model = self._build_model(cts_state_dict)
        h_prev = [np.zeros((1, 1))]
        c_prev = [np.zeros((8, 1))]
        u_step = np.array([0.7])
        y_step = np.array([0.5])

        h_new, c_new = model.step_numeric(u_step, y_step, h_prev, c_prev)
        assert not np.allclose(np.array(c_new[0]), 0.0)

    def test_stateful_forward_symbolic_shape(self, cts_state_dict):
        """forward(u, h0=..., c0=...) returns shape (h_out, sequence_length)."""
        model = self._build_model(cts_state_dict, n_context=2, horizon=5)
        # Controls only: (n_inputs, T) = (1, horizon) = (1, 5)
        u = cs.MX.sym("u", 1, 5)
        h0 = [cs.MX.sym("h0_l0", 1, 1)]
        c0 = [cs.MX.sym("c0_l0", 8, 1)]
        y_sim = model.forward(u, h0=h0, c0=c0)
        assert y_sim.size1() == 1
        assert y_sim.size2() == 5

    def test_stateful_forward_no_anchor(self, cts_state_dict):
        """Every forward column is a genuine prediction (no zero-pad anchor)."""
        model = self._build_model(cts_state_dict, n_context=2, horizon=5)
        rng = np.random.default_rng(7)
        # Controls only: (n_inputs, T) = (1, horizon) = (1, 5)
        u_np = rng.standard_normal((1, 5))
        h0 = [cs.DM(np.zeros((1, 1)))]
        c0 = [cs.DM(np.zeros((8, 1)))]
        y_np = np.array(model.forward(cs.DM(u_np), h0=h0, c0=c0))

        # forward rolls the LSTM from h0/c0 using the controls; column 0 must
        # already be the first genuine prediction, with no leading placeholder.
        y_roll, _, _ = _rollout(model, u_np.T)  # _rollout wants (T, n_inputs)
        assert y_np.shape == (1, 5)
        npt.assert_allclose(y_np.ravel(), y_roll.ravel(), atol=1e-9)


class TestCasadiLSTMDisturbances:
    """Measured-disturbance (feedforward) channel: core input laid out as [u, d]."""

    def test_core_width_includes_disturbances(self):
        """n_disturbances widens the core input to nu + nd (controls + disturbances)."""
        model = CasadiLSTM(
            n_context=2, n_inputs=2, hidden_size=8,
            horizon=5, proj_size=1, n_disturbances=1,
        )
        assert model.n_inputs == 2          # controls only (backward compatible)
        assert model.n_disturbances == 1
        assert model.n_core_inputs == 3
        assert model._core.n_inputs == 3    # core consumes [u, d]

    def test_forward_matches_combined_input_model(self, make_lstm_state_dict):
        """forward(u, d) with [u,d] layout equals a plain (nu+nd)-input rollout."""
        hidden, proj, N = 8, 1, 5
        nu, nd = 2, 1
        sd = make_lstm_state_dict(nu + nd, hidden, proj_size=proj, seed=3)

        # Model A: explicit disturbance channel (core width nu+nd via [u, d]).
        model_d = CasadiLSTM(
            n_context=0, n_inputs=nu, hidden_size=hidden,
            horizon=N, proj_size=proj, n_disturbances=nd,
        )
        model_d.load_state_dict(sd)
        # Model B: same weights, all channels treated as "controls".
        model_full = CasadiLSTM(
            n_context=0, n_inputs=nu + nd, hidden_size=hidden,
            horizon=N, proj_size=proj,
        )
        model_full.load_state_dict(sd)

        rng = np.random.default_rng(11)
        u = rng.standard_normal((nu, N))
        d = rng.standard_normal((nd, N))
        h0 = [cs.DM(np.zeros((proj, 1)))]
        c0 = [cs.DM(np.zeros((hidden, 1)))]

        y_d = np.array(model_d.forward(cs.DM(u), h0=h0, c0=c0, d=cs.DM(d)))
        y_full = np.array(
            model_full.forward(cs.DM(np.vstack([u, d])), h0=h0, c0=c0)
        )
        npt.assert_allclose(y_d, y_full, atol=1e-9)

    def test_forward_requires_d_when_disturbances(self, make_lstm_state_dict):
        """forward raises a clear error if d is omitted but n_disturbances > 0."""
        sd = make_lstm_state_dict(2, 8, proj_size=1, seed=1)
        model = CasadiLSTM(
            n_context=0, n_inputs=1, hidden_size=8,
            horizon=3, proj_size=1, n_disturbances=1,
        )
        model.load_state_dict(sd)
        with pytest.raises(ValueError, match="requires `d`"):
            model.forward(cs.MX.sym("u", 1, 3), h0=[cs.MX.sym("h", 1)],
                          c0=[cs.MX.sym("c", 8)])

    def test_n_disturbances_zero_is_noop(self, cts_state_dict):
        """n_disturbances=0 leaves the control-only forward path unchanged."""
        m0 = CasadiLSTM(
            n_context=0, n_inputs=1, hidden_size=8, horizon=4, proj_size=1,
        )
        m0.load_state_dict(cts_state_dict)
        assert m0.n_disturbances == 0
        assert m0.n_core_inputs == 1
        rng = np.random.default_rng(5)
        u = rng.standard_normal((1, 4))
        h0 = [cs.DM(np.zeros((1, 1)))]
        c0 = [cs.DM(np.zeros((8, 1)))]
        y = np.array(m0.forward(cs.DM(u), h0=h0, c0=c0))  # no d needed
        y_roll, _, _ = _rollout(m0, u.T)
        npt.assert_allclose(y.ravel(), y_roll.ravel(), atol=1e-9)

    def test_estimate_and_step_numeric_with_disturbance(self, make_lstm_state_dict):
        """estimate_numeric/step_numeric accept d_ctx/d_step and stay finite."""
        sd = make_lstm_state_dict(3, 8, proj_size=1, seed=2)
        model = CasadiLSTM(
            n_context=2, n_inputs=2, hidden_size=8,
            horizon=5, proj_size=1, n_disturbances=1,
        )
        model.load_state_dict(sd)
        rng = np.random.default_rng(0)
        u_ctx = rng.standard_normal((2, 2))   # (n_steps, nu)
        y_ctx = rng.standard_normal((2, 1))   # (n_steps, h_out)
        d_ctx = rng.standard_normal((2, 1))   # (n_steps, nd)

        h, c = model.estimate_numeric(u_ctx, y_ctx, d_ctx=d_ctx)
        assert np.all(np.isfinite(np.array(h[0]))) and np.all(np.isfinite(np.array(c[0])))

        h2, c2 = model.step_numeric(u_ctx[-1], y_ctx[-1], h, c, d_step=d_ctx[-1])
        assert np.all(np.isfinite(np.array(c2[0])))

        # Omitting the disturbance raises when the model declares channels.
        with pytest.raises(ValueError, match="requires `d_ctx`"):
            model.estimate_numeric(u_ctx, y_ctx)
        with pytest.raises(ValueError, match="requires `d_step`"):
            model.step_numeric(u_ctx[-1], y_ctx[-1], h, c)

    def test_disturbance_changes_output(self, make_lstm_state_dict):
        """A different disturbance trajectory changes the predicted output."""
        sd = make_lstm_state_dict(2, 8, proj_size=1, seed=4)
        model = CasadiLSTM(
            n_context=0, n_inputs=1, hidden_size=8,
            horizon=4, proj_size=1, n_disturbances=1,
        )
        model.load_state_dict(sd)
        u = cs.DM(np.ones((1, 4)) * 0.5)
        h0 = [cs.DM(np.zeros((1, 1)))]
        c0 = [cs.DM(np.zeros((8, 1)))]
        y_zero = np.array(model.forward(u, h0=h0, c0=c0, d=cs.DM(np.zeros((1, 4)))))
        y_one = np.array(model.forward(u, h0=h0, c0=c0, d=cs.DM(np.ones((1, 4)))))
        assert not np.allclose(y_zero, y_one)
