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

import casadi as cs
import numpy as np
import torch
from itertools import count


class _CasadiLSTMCore:
    _instance_counter = count(0)

    def __init__(
        self,
        n_inputs,
        hidden_size,
        proj_size,
        num_layers=1,
        bias=True,
    ):
        """
        Multi-layer LSTM with optional projection (LSTMP) built in CasADi.

        It mirrors PyTorch's LSTM(+projection) layout so you can copy trained weights
        via `load_state_dict`. The class builds a single CasADi function:

        - `openlstm_casadi_step_full(U_step, h0_l*, c0_l*, weights...) -> (Y, hN_l*, cN_l*)`
          — one step with per-layer initial states (numeric or symbolic), wrapped
          by `step_full`.

        Gates follow PyTorch's order **[i, f, g, o]**.

        Parameters
        ----------
        n_inputs : int
        hidden_size : int                # H (cell size)
        proj_size : int                  # P; if 0 → no projection, H_out = H
        num_layers : int, default 1
        bias : bool, default True

        Shapes (key ones)
        -----------------
        Single step:
        - U_step: (1, n_inputs) → Y: (1, H_out)
        - h0 per layer: (H_out, 1) | c0 per layer: (H, 1)

        Notes
        -----
        - Projection (if P>0): h_t = W_hr · (o ⊙ tanh(c_t)); else h_t = o ⊙ tanh(c_t).
        - Recurrent weights `W_hh` use the exposed hidden size `H_out` (P if >0 else H).
        """
        self._uid = next(_CasadiLSTMCore._instance_counter)
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.num_layers = num_layers
        self.bias = bias

        # Exposed dimension of h_t (PyTorch: proj_size=0 => h has dimension H)
        self.h_out = proj_size if proj_size > 0 else hidden_size
        self.h_real = self.h_out

        self.U_step = cs.MX.sym("U_step", 1, self.n_inputs)

        # Parameter symbols (mirroring PyTorch LSTM weight layout)
        self.W_ih = []
        self.W_hh = []
        self.b_ih = []
        self.b_hh = []
        self.W_hr = []  # if P==0, identity is used in _lstm_layer_step

        for l in range(self.num_layers):
            in_dim = self.n_inputs if l == 0 else self.h_out
            self.W_ih.append(cs.MX.sym(f"W_ih_l{l}", 4 * self.hidden_size, in_dim))
            self.W_hh.append(cs.MX.sym(f"W_hh_l{l}", 4 * self.hidden_size, self.h_real))
            if self.bias:
                self.b_ih.append(cs.MX.sym(f"b_ih_l{l}", 4 * self.hidden_size))
                self.b_hh.append(cs.MX.sym(f"b_hh_l{l}", 4 * self.hidden_size))
            else:
                self.b_ih.append(cs.DM.zeros(4 * self.hidden_size))
                self.b_hh.append(cs.DM.zeros(4 * self.hidden_size))

            # projection (if P>0 we will have a symbol; otherwise, virtual identity)
            if self.proj_size > 0:
                self.W_hr.append(
                    cs.MX.sym(f"W_hr_l{l}", self.proj_size, self.hidden_size)
                )
            else:
                self.W_hr.append(None)  # signals "use HxH identity" in the step

        self.W_ih_val = [None] * self.num_layers
        self.W_hh_val = [None] * self.num_layers
        self.b_ih_val = [None] * self.num_layers
        self.b_hh_val = [None] * self.num_layers
        self.W_hr_val = [None] * self.num_layers

        self._build_step_function_full()

    def _build_step_function_full(self):
        """Builds a single-step CasADi Function with per-layer h0/c0 inputs.

        The function takes one (h0, c0) per layer and returns all updated
        per-layer (hN, cN), so that h/c can be persisted across MPC
        iterations for arbitrary `num_layers`.
        """
        h0_list = [
            cs.MX.sym(f"h0_l{l}_full", self.h_out, 1) for l in range(self.num_layers)
        ]
        c0_list = [
            cs.MX.sym(f"c0_l{l}_full", self.hidden_size, 1)
            for l in range(self.num_layers)
        ]
        h_list, c_list = self._lstm_step(self.U_step, h0_list, c0_list)
        Y = h_list[-1].T

        W_ih_args = [w for w in self.W_ih]
        W_hh_args = [w for w in self.W_hh]
        b_ih_args = [b for b in self.b_ih]
        b_hh_args = [b for b in self.b_hh]
        W_hr_args = [
            w if w is not None else cs.DM.eye(self.hidden_size) for w in self.W_hr
        ]

        self.lstm_step_func_full = cs.Function(
            f"openlstm_casadi_step_full_{self._uid}",
            [
                self.U_step,
                *h0_list,
                *c0_list,
                *W_ih_args,
                *W_hh_args,
                *b_ih_args,
                *b_hh_args,
                *W_hr_args,
            ],
            [Y, *h_list, *c_list],
            ["U"]
            + [f"h0_l{l}" for l in range(self.num_layers)]
            + [f"c0_l{l}" for l in range(self.num_layers)]
            + [f"W_ih_l{l}" for l in range(self.num_layers)]
            + [f"W_hh_l{l}" for l in range(self.num_layers)]
            + [f"b_ih_l{l}" for l in range(self.num_layers)]
            + [f"b_hh_l{l}" for l in range(self.num_layers)]
            + [f"W_hr_l{l}" for l in range(self.num_layers)],
            ["Y"]
            + [f"hN_l{l}" for l in range(self.num_layers)]
            + [f"cN_l{l}" for l in range(self.num_layers)],
        )

    def step_full(self, u_t, h_prev_list, c_prev_list):
        """Run one LSTM step with per-layer initial h/c (numeric or symbolic).

        Parameters
        ----------
        u_t : np.ndarray or cs.MX/DM
            Control input at current timestep, shape (1, n_inputs) or (n_inputs,).
        h_prev_list : list of (h_out, 1)
            Previous hidden state per layer.
        c_prev_list : list of (hidden_size, 1)
            Previous cell state per layer.

        Returns
        -------
        y : output, shape (1, h_out)
        h_next_list : list of (h_out, 1)
        c_next_list : list of (hidden_size, 1)
        """
        if isinstance(u_t, np.ndarray) and u_t.ndim == 1:
            u_t = u_t[None, :]
        elif isinstance(u_t, cs.MX):
            u_t = cs.reshape(u_t, 1, self.n_inputs)

        # Pass raw numeric weights so that purely numeric inputs (numpy or DM)
        # yield numeric DM outputs; symbolic MX inputs still produce MX outputs
        # with the weights baked in as DM constants.
        outs = self.lstm_step_func_full(
            u_t,
            *h_prev_list,
            *c_prev_list,
            *self.W_ih_val,
            *self.W_hh_val,
            *self.b_ih_val,
            *self.b_hh_val,
            *self.W_hr_val,
        )
        y = outs[0]
        h_next_list = list(outs[1 : 1 + self.num_layers])
        c_next_list = list(outs[1 + self.num_layers : 1 + 2 * self.num_layers])
        return y, h_next_list, c_next_list

    @staticmethod
    def _sigmoid(x):
        """CasADi-compatible sigmoid activation: 1/(1 + exp(-x))."""
        return 1 / (1 + cs.exp(-x))

    def load_state_dict(self, state_dict):
        """
        Loads parameters from a PyTorch state_dict.

        The state_dict must contain the keys:
         - 'weight_ih_li'
         - 'weight_hh_li'
         - 'bias_ih_li'
         - 'bias_hh_li'
         - 'weight_hr_li'

        Where i is the index of each network layer.

        Accepts both PyTorch tensors and NumPy arrays.
        """

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.array(x)

        # Supports prefix 'model.'. Tries with and without prefix.
        def get(k):
            if k in state_dict:
                return state_dict[k]
            k2 = k.replace("model.", "")
            if k2 in state_dict:
                return state_dict[k2]
            return None

        for l in range(self.num_layers):
            W_ih_k = f"model.weight_ih_l{l}"
            W_hh_k = f"model.weight_hh_l{l}"
            b_ih_k = f"model.bias_ih_l{l}"
            b_hh_k = f"model.bias_hh_l{l}"
            W_hr_k = f"model.weight_hr_l{l}"  # only exists if proj_size>0

            W_ih_v = get(W_ih_k)
            W_hh_v = get(W_hh_k)
            if W_ih_v is None or W_hh_v is None:
                raise KeyError(f"Missing core weights for layer {l}")

            self.W_ih_val[l] = to_numpy(W_ih_v)
            self.W_hh_val[l] = to_numpy(W_hh_v)

            if self.bias:
                b_ih_v = get(b_ih_k)
                b_hh_v = get(b_hh_k)
                if b_ih_v is None or b_hh_v is None:
                    raise KeyError(f"Missing bias for layer {l} while bias=True")
                self.b_ih_val[l] = to_numpy(b_ih_v)
                self.b_hh_val[l] = to_numpy(b_hh_v)
            else:
                self.b_ih_val[l] = np.zeros((4 * self.hidden_size,), dtype=float)
                self.b_hh_val[l] = np.zeros((4 * self.hidden_size,), dtype=float)

            if self.proj_size > 0:
                W_hr_v = get(W_hr_k)
                if W_hr_v is None:
                    raise KeyError(
                        f"Missing projection weight for layer {l} with proj_size>0"
                    )
                self.W_hr_val[l] = to_numpy(W_hr_v)
            else:
                # virtual HxH identity for the no-projection case
                self.W_hr_val[l] = np.eye(self.hidden_size, dtype=float)

    def _lstm_step(self, u_t, h_prev_layers, c_prev_layers):
        """Executes one LSTM step across all layers.

        Parameters
        ----------
        u_t : cs.MX
            Control input at current timestep, shape (1, n_inputs).
        h_prev_layers : list of cs.MX
            Previous hidden states per layer, each shape (H_out, 1).
        c_prev_layers : list of cs.MX
            Previous cell states per layer, each shape (H, 1).

        Returns
        -------
        tuple of (h_layers, c_layers)
            Updated hidden and cell states for all layers.
        """
        h_layers = []
        c_layers = []
        layer_input = u_t

        for l in range(self.num_layers):
            h_l, c_l = self._lstm_layer_step(
                layer_input, h_prev_layers[l], c_prev_layers[l], l
            )
            h_layers.append(h_l)
            c_layers.append(c_l)
            # output of this layer is input to the next
            layer_input = h_l.T  # (1, H_out)

        return h_layers, c_layers

    def _lstm_layer_step(self, layer_input, h_prev, c_prev, l):
        """Computes LSTM gates and updates for a single layer.

        Parameters
        ----------
        layer_input : cs.MX
            Input to this layer, shape (1, in_dim). For layer 0 this is the
            control u_t; for layers > 0 it is the previous layer's hidden state.
        h_prev : cs.MX
            Previous hidden state, shape (H_out, 1).
        c_prev : cs.MX
            Previous cell state, shape (H, 1).
        l : int
            Layer index.

        Returns
        -------
        tuple of (h, c)
            Updated hidden state (H_out, 1) and cell state (H, 1).
        """
        gates = (
            self.W_ih[l] @ layer_input.T
            + self.b_ih[l]
            + self.W_hh[l] @ h_prev
            + self.b_hh[l]
        )
        i = self._sigmoid(gates[0 : self.hidden_size, :])
        f = self._sigmoid(gates[self.hidden_size : 2 * self.hidden_size, :])
        g = cs.tanh(gates[2 * self.hidden_size : 3 * self.hidden_size, :])
        o = self._sigmoid(gates[3 * self.hidden_size : 4 * self.hidden_size, :])

        c = f * c_prev + i * g
        h_raw = o * cs.tanh(c)

        # projection (if P>0 use W_hr; otherwise, HxH identity)
        if self.proj_size > 0:
            h = self.W_hr[l] @ h_raw  # (P,1)
        else:
            h = h_raw  # (H,1)

        return h, c


class CasadiLSTM:
    """
    CasADi LSTM for stateful neural MPC: builds a symbolic single-step LSTM,
    loads PyTorch weights, and rolls the prediction horizon forward from
    externally maintained hidden/cell states with I/O adaptation.

    Usage
    -----
    >>> model = CasadiLSTM(
    ...     n_context=10, n_inputs=2, hidden_size=128,
    ...     horizon=20, proj_size=4,
    ... )
    >>> model.load_state_dict(torch.load("model.pt"))
    >>> mpc.set_neural_dynamics(model=model, n_warmup=1)

    Stateful contract
    -----------------
    The LSTM never estimates its own initial state inside the symbolic graph.
    Instead, per-layer hidden/cell states are maintained numerically between
    MPC solves via the teacher-forced helpers `estimate_numeric` (full context
    window) and `step_numeric` (single step), and passed to `forward(u, h0=,
    c0=)` as the starting point of the symbolic prediction rollout. The
    prediction rollout consumes the controls `u` only; past outputs `y` enter
    exclusively through the warmup helpers (teacher forcing).

    Parameters
    ----------
    n_context : int
        Number of measured context samples used for the numeric warmup.
    n_inputs : int
        Number of control inputs (u).
    hidden_size : int
        LSTM cell size H.
    horizon : int
        Number of steps to predict after context.
    num_layers : int, default 1
        Number of LSTM layers.
    bias : bool, default True
        Whether to use bias terms.
    proj_size : int, default 0
        Projection size P. If 0, output size equals hidden_size.
    n_disturbances : int, default 0
        Number of measured-disturbance (feedforward) channels `d` the model was
        trained on. When >0 the LSTM core consumes ``n_inputs + n_disturbances``
        features per step, laid out as ``[u, d]`` (controls first, disturbances
        immediately after). The disturbance enters the network like the controls
        but is an NLP parameter, not a decision variable.

    Attributes
    ----------
    output_size : int
        Output dimension (proj_size if >0, else hidden_size).
    sequence_length : int
        Total sequence length (horizon).
    """

    def __init__(
        self,
        n_context: int,
        n_inputs: int,
        hidden_size: int,
        horizon: int,
        num_layers: int = 1,
        bias: bool = True,
        proj_size: int = 0,
        n_disturbances: int = 0,
    ):
        self.n_context = n_context
        self.horizon = horizon
        self.n_inputs = n_inputs
        self.n_disturbances = n_disturbances
        # Full LSTM core input width: controls followed by disturbances ([u, d]).
        self.n_core_inputs = n_inputs + n_disturbances
        self.sequence_length = horizon
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.h_out = proj_size if proj_size > 0 else hidden_size
        self.output_size = self.h_out

        self._core = _CasadiLSTMCore(
            n_inputs=self.n_core_inputs,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_layers=num_layers,
            bias=bias,
        )

    def load_state_dict(self, state_dict):
        """Load PyTorch-style weights into the LSTM.

        Accepts state_dicts with or without 'model.' prefix on keys.

        Parameters
        ----------
        state_dict : dict
            PyTorch state_dict containing LSTM weights/biases.
        """
        self._core.load_state_dict(state_dict)

    def forward(self, u, *, h0, c0, d=None):
        """Roll the LSTM over the prediction horizon from external h0/c0.

        Builds the symbolic graph by chaining single-step calls to
        `_core.step_full` for `t = 0 .. sequence_length-1`. The rollout consumes
        the control sequence (and the measured-disturbance sequence `d`, when the
        model has disturbance channels); past outputs `y` enter exclusively
        through the warmup helpers (`estimate_numeric`/`step_numeric`), which seed
        `h0`/`c0`. Each per-step core input is laid out as ``[u_t, d_t]``.

        Parameters
        ----------
        u : cs.MX, cs.DM, or torch.Tensor
            Control sequence with shape (n_inputs, sequence_length).
        h0, c0 : list of cs.MX
            Per-layer initial hidden/cell states, maintained between solves
            by `estimate_numeric`/`step_numeric`.
        d : cs.MX, cs.DM, or torch.Tensor, optional
            Measured-disturbance sequence with shape (n_disturbances,
            sequence_length). Required when `n_disturbances > 0`; ignored when 0.

        Returns
        -------
        cs.MX
            Normalized output, shape (output_size, sequence_length).
        """
        u_cols = u.T  # (T, n_inputs); the rollout is driven by controls + h0/c0
        if self.n_disturbances > 0:
            if d is None:
                raise ValueError(
                    "forward() requires `d` when n_disturbances > 0 "
                    f"(model has n_disturbances={self.n_disturbances})."
                )
            d_cols = d.T  # (T, n_disturbances)

        outputs = []
        h_list = list(h0)
        c_list = list(c0)
        for t in range(0, self.sequence_length):
            u_t = cs.reshape(u_cols[t, :], 1, self.n_inputs)
            if self.n_disturbances > 0:
                d_t = cs.reshape(d_cols[t, :], 1, self.n_disturbances)
                inp_t = cs.horzcat(u_t, d_t)  # (1, n_core_inputs), layout [u, d]
            else:
                inp_t = u_t
            y_t, h_list, c_list = self._core.step_full(inp_t, h_list, c_list)
            outputs.append(y_t)  # (1, h_out)

        y_pred = cs.horzcat(*outputs)  # (1, horizon * h_out)
        return self._normalize_output(y_pred)

    def __call__(self, u, *, h0, c0, d=None):
        return self.forward(u, h0=h0, c0=c0, d=d)

    def estimate_numeric(self, u_ctx, y_ctx, d_ctx=None, h_seed=None, c_seed=None):
        """Teacher-forced numeric rollout over the context window.

        Executes purely numerically using `_core.step_full`. The LSTM's
        layer-0 hidden state is overwritten with the measured `y` at each
        step (teacher forcing, mirroring how the network is trained); the
        cell state and higher-layer hidden states are propagated. This warmup
        (together with `step_numeric`) is the **only** place past outputs `y`
        are consumed — the symbolic prediction rollout in `forward` uses the
        controls (and disturbances) alone. Each per-step core input is laid out
        as ``[u_t, d_t]``.

        Parameters
        ----------
        u_ctx : array_like, shape (n_steps, n_inputs)
        y_ctx : array_like, shape (n_steps, h_out)
        d_ctx : array_like, shape (n_steps, n_disturbances), optional
            Measured-disturbance context. Required when `n_disturbances > 0`.
        h_seed, c_seed : list of np.ndarray, optional
            Per-layer initial states. Default to zeros.

        Returns
        -------
        h_new, c_new : list of np.ndarray
            Per-layer updated hidden/cell states, each (h_out,1) / (hidden_size,1).
        """
        if self.n_disturbances > 0 and d_ctx is None:
            raise ValueError(
                "estimate_numeric() requires `d_ctx` when n_disturbances > 0 "
                f"(model has n_disturbances={self.n_disturbances})."
            )
        if h_seed is None:
            h_list = [np.zeros((self.h_out, 1)) for _ in range(self.num_layers)]
        else:
            h_list = [np.asarray(h, dtype=float).reshape(self.h_out, 1) for h in h_seed]
        if c_seed is None:
            c_list = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
        else:
            c_list = [
                np.asarray(c, dtype=float).reshape(self.hidden_size, 1) for c in c_seed
            ]

        u_arr = np.asarray(u_ctx, dtype=float)
        y_arr = np.asarray(y_ctx, dtype=float)
        d_arr = np.asarray(d_ctx, dtype=float) if self.n_disturbances > 0 else None
        n = u_arr.shape[0]
        for i in range(n):
            u_t = u_arr[i].reshape(1, self.n_inputs)
            if self.n_disturbances > 0:
                d_t = d_arr[i].reshape(1, self.n_disturbances)
                inp_t = np.hstack([u_t, d_t])  # (1, n_core_inputs), layout [u, d]
            else:
                inp_t = u_t
            y_t = y_arr[i].reshape(self.h_out, 1)
            # Teacher forcing on layer 0
            h_list = [y_t] + h_list[1:]
            _, h_list, c_list = self._core.step_full(inp_t, h_list, c_list)

        h_np = [np.asarray(h).reshape(self.h_out, 1) for h in h_list]
        c_np = [np.asarray(c).reshape(self.hidden_size, 1) for c in c_list]
        return h_np, c_np

    def step_numeric(self, u_step, y_step, h_prev, c_prev, d_step=None):
        """Advance the LSTM one numeric step using teacher forcing.

        Used between MPC solves to advance the persisted (h, c) buffers
        with the latest measured `(u, y)` (and disturbance `d`). Teacher forcing
        replaces the layer-0 hidden state with `y_step` so the measurement
        directly corrects the LSTM's memory. The per-step core input is laid out
        as ``[u_step, d_step]``.

        Parameters
        ----------
        u_step : array_like, shape (n_inputs,)
        y_step : array_like, shape (h_out,)
        h_prev, c_prev : list of np.ndarray
            Per-layer previous states.
        d_step : array_like, shape (n_disturbances,), optional
            Measured disturbance at this step. Required when `n_disturbances > 0`.

        Returns
        -------
        h_new, c_new : list of np.ndarray
        """
        if self.n_disturbances > 0 and d_step is None:
            raise ValueError(
                "step_numeric() requires `d_step` when n_disturbances > 0 "
                f"(model has n_disturbances={self.n_disturbances})."
            )
        h_list = [np.asarray(h, dtype=float).reshape(self.h_out, 1) for h in h_prev]
        c_list = [np.asarray(c, dtype=float).reshape(self.hidden_size, 1) for c in c_prev]
        u_t = np.asarray(u_step, dtype=float).reshape(1, self.n_inputs)
        if self.n_disturbances > 0:
            d_t = np.asarray(d_step, dtype=float).reshape(1, self.n_disturbances)
            inp_t = np.hstack([u_t, d_t])  # (1, n_core_inputs), layout [u, d]
        else:
            inp_t = u_t
        y_t = np.asarray(y_step, dtype=float).reshape(self.h_out, 1)
        h_list = [y_t] + h_list[1:]
        _, h_new, c_new = self._core.step_full(inp_t, h_list, c_list)
        h_np = [np.asarray(h).reshape(self.h_out, 1) for h in h_new]
        c_np = [np.asarray(c).reshape(self.hidden_size, 1) for c in c_new]
        return h_np, c_np

    def get_model(self):
        """Get the underlying core LSTM model.

        Returns
        -------
        _CasadiLSTMCore
            The inner LSTM model instance.
        """
        return self._core

    def _normalize_output(self, y):
        """Normalize output shape to (output_size, sequence_length)."""
        if not isinstance(y, (cs.MX, cs.SX, cs.DM)):
            raise TypeError("LSTM output should be CasADi (MX/SX/DM).")
        T = self.sequence_length
        ny = self.output_size
        if y.size1() == 1 and ny == 1:
            return y  # (1, T)
        if y.size1() == ny:
            return y  # (ny, T)
        if y.size2() == ny and y.size1() == T:
            return y.T
        return cs.reshape(y, ny, -1)
