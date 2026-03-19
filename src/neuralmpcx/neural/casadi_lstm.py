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
        prediction_horizon,
        num_layers=1,
        bias=True,
    ):
        """
        Multi-layer LSTM with optional projection (LSTMP) built in CasADi.

        It mirrors PyTorch's LSTM(+projection) layout so you can copy trained weights
        via `load_state_dict`. The class builds two CasADi functions:

        - `openlstm_casadi(X, h0, c0, weights...) -> (Y, hN, cN)`  — full rollout over `T`.
        - `openlstm_casadi_step(X_step, h_prev, c_prev, weights...) -> (Y, hN, cN)` — one step.

        Gates follow PyTorch's order **[i, f, g, o]**.

        Parameters
        ----------
        n_inputs : int
        hidden_size : int                # H (cell size)
        proj_size : int                  # P; if 0 → no projection, H_out = H
        prediction_horizon : int         # T (unroll length)
        num_layers : int, default 1
        bias : bool, default True

        Shapes (key ones)
        -----------------
        Full sequence:
        - X: (T, n_inputs), h0: (H_out,), c0: (H,)
        - Y: (1, T*H_out)  | hN: (H_out, 1) | cN: (H, 1)
        Single step:
        - X_step: (1, n_inputs) → Y: (1, H_out)

        Notes
        -----
        - Projection (if P>0): h_t = W_hr · (o ⊙ tanh(c_t)); else h_t = o ⊙ tanh(c_t).
        - Recurrent weights `W_hh` use the exposed hidden size `H_out` (P if >0 else H).
        """
        self._uid = next(_CasadiLSTMCore._instance_counter)
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.horizon = prediction_horizon
        self.num_layers = num_layers
        self.bias = bias

        # Exposed dimension of h_t (PyTorch: proj_size=0 => h has dimension H)
        self.h_out = proj_size if proj_size > 0 else hidden_size
        self.h_real = self.h_out

        self.X = cs.MX.sym("X", self.horizon, self.n_inputs)
        self.h0 = cs.MX.sym("h0", self.h_out)
        self.c0 = cs.MX.sym("c0", self.hidden_size)

        self.X_step = cs.MX.sym("X_step", 1, self.n_inputs)
        self.h_0_step = cs.MX.sym("h_prev", self.h_out, 1)
        self.c0_step = cs.MX.sym("c_prev", self.hidden_size, 1)

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

        self._build_function()
        self._build_step_function()

    def _build_function(self):
        """Builds the CasADi Function that performs the complete forward pass."""
        # initial states per layer
        h_list0 = [cs.reshape(self.h0, self.h_out, 1) for _ in range(self.num_layers)]
        c_list0 = [
            cs.reshape(self.c0, self.hidden_size, 1) for _ in range(self.num_layers)
        ]

        h_list = h_list0
        c_list = c_list0
        outputs = []

        # loop through the sequence
        for t in range(self.horizon):
            x_t = cs.reshape(self.X[t, :], 1, self.n_inputs)
            h_list, c_list = self._lstm_step(x_t, h_list, c_list)
            outputs.append(h_list[-1])  # h: batch x proj_size

        # concatenate all outputs over time

        Y = cs.horzcat(*[o.T for o in outputs])
        self.Y = Y

        # flatten weights/bias lists for function signature
        W_ih_args = [w for w in self.W_ih]
        W_hh_args = [w for w in self.W_hh]
        b_ih_args = [b for b in self.b_ih]
        b_hh_args = [b for b in self.b_hh]
        W_hr_args = [
            w if w is not None else cs.DM.eye(self.hidden_size) for w in self.W_hr
        ]

        # define the CasADi function (uid suffix avoids name collisions on kernel re-run)
        self.lstm_func = cs.Function(
            f"openlstm_casadi_{self._uid}",
            [
                self.X,
                self.h0,
                self.c0,
                *W_ih_args,
                *W_hh_args,
                *b_ih_args,
                *b_hh_args,
                *W_hr_args,
            ],
            [Y, h_list[-1], c_list[-1]],
            ["X", "h0", "c0"]
            + [f"W_ih_l{l}" for l in range(self.num_layers)]
            + [f"W_hh_l{l}" for l in range(self.num_layers)]
            + [f"b_ih_l{l}" for l in range(self.num_layers)]
            + [f"b_hh_l{l}" for l in range(self.num_layers)]
            + [f"W_hr_l{l}" for l in range(self.num_layers)],
            ["Y", "hN", "cN"],
        )

    def _build_step_function(self):
        """Builds CasADi Function for single LSTM step forward pass."""
        h_list = [self.h_0_step] + [
            cs.MX.sym(f"h_prev_l{l}", self.h_out, 1) for l in range(1, self.num_layers)
        ]
        c_list = [self.c0_step] + [
            cs.MX.sym(f"c_prev_l{l}", self.hidden_size, 1)
            for l in range(1, self.num_layers)
        ]
        # reuse _lstm_step to create symbolic output
        h_list, c_list = self._lstm_step(self.X_step, h_list, c_list)
        # CasADi function for a single step (still requires weights as input)
        Y = h_list[-1].T

        W_ih_args = [w for w in self.W_ih]
        W_hh_args = [w for w in self.W_hh]
        b_ih_args = [b for b in self.b_ih]
        b_hh_args = [b for b in self.b_hh]
        W_hr_args = [
            w if w is not None else cs.DM.eye(self.hidden_size) for w in self.W_hr
        ]

        self.lstm_step_func = cs.Function(
            f"openlstm_casadi_step_{self._uid}",
            [
                self.X_step,
                self.h_0_step,
                self.c0_step,
                *W_ih_args,
                *W_hh_args,
                *b_ih_args,
                *b_hh_args,
                *W_hr_args,
            ],
            [Y, h_list[-1], c_list[-1]],
            ["X", "h0", "c0"]
            + [f"W_ih_l{l}" for l in range(self.num_layers)]
            + [f"W_hh_l{l}" for l in range(self.num_layers)]
            + [f"b_ih_l{l}" for l in range(self.num_layers)]
            + [f"b_hh_l{l}" for l in range(self.num_layers)]
            + [f"W_hr_l{l}" for l in range(self.num_layers)],
            ["Y", "hN", "cN"],
        )

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

    def _lstm_step(self, x_t, h_prev_layers, c_prev_layers):
        """Executes one LSTM step across all layers.

        Parameters
        ----------
        x_t : cs.MX
            Input at current timestep, shape (1, n_inputs).
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
        layer_input = x_t

        for l in range(self.num_layers):
            h_l, c_l = self._lstm_layer_step(
                layer_input, h_prev_layers[l], c_prev_layers[l], l
            )
            h_layers.append(h_l)
            c_layers.append(c_l)
            # output of this layer is input to the next
            layer_input = h_l.T  # (1, H_out)

        return h_layers, c_layers

    def _lstm_layer_step(self, x_t, h_prev, c_prev, l):
        """Computes LSTM gates and updates for a single layer.

        Parameters
        ----------
        x_t : cs.MX
            Layer input, shape (1, in_dim).
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
            self.W_ih[l] @ x_t.T + self.b_ih[l] + self.W_hh[l] @ h_prev + self.b_hh[l]
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

    def forward(self, X_val, h0_val, c0_val):
        """
        Executes the LSTM forward pass.

        Args:
            X_val (np.ndarray): shape (sequence_len, n_inputs)
            h0_val (np.ndarray): shape (proj_size)
            c0_val (np.ndarray): shape (hidden_size)

        Returns:
            Y (np.ndarray): concatenated output, shape (seq_len * proj_size)
            hN (np.ndarray): final h state, shape (proj_size)
            cN (np.ndarray): final c state, shape (hidden_size)
        """
        vals = [X_val, h0_val, c0_val]

        # pack parameters per layer
        for l in range(self.num_layers):
            vals.append(self.W_ih_val[l])
        for l in range(self.num_layers):
            vals.append(self.W_hh_val[l])
        for l in range(self.num_layers):
            vals.append(self.b_ih_val[l])
        for l in range(self.num_layers):
            vals.append(self.b_hh_val[l])
        for l in range(self.num_layers):
            vals.append(self.W_hr_val[l])

        return self.lstm_func(*vals)

    def forward_one_step(self, x_t, h_0_val, c_0_val):
        """
        Executes a single LSTM step.
        x_t: (1, n_inputs)  -- can be np.ndarray 1D, 2D or MX
        h_0_val: (h_out, 1) -- MX recommended
        c_0_val: (hidden_size, 1) -- can come as np, we will convert to MX
        """
        # reshape NumPy → (1,n_inputs)
        if isinstance(x_t, np.ndarray) and x_t.ndim == 1:
            x_t = x_t[None, :]
        # symbolic reshape → (1,n_inputs)
        elif isinstance(x_t, cs.MX):
            x_t = cs.reshape(x_t, 1, self.n_inputs)

        # weights/bias per layer: expand lists and convert to MX
        W_ih_args = [cs.MX(self.W_ih_val[l]) for l in range(self.num_layers)]
        W_hh_args = [cs.MX(self.W_hh_val[l]) for l in range(self.num_layers)]
        b_ih_args = [cs.MX(self.b_ih_val[l]) for l in range(self.num_layers)]
        b_hh_args = [cs.MX(self.b_hh_val[l]) for l in range(self.num_layers)]
        W_hr_args = [cs.MX(self.W_hr_val[l]) for l in range(self.num_layers)]

        return self.lstm_step_func(
            x_t,  # (1,n_inputs)
            h_0_val,  # (proj_size,1)
            c_0_val,  # (hidden_size,1)
            *W_ih_args,
            *W_hh_args,
            *b_ih_args,
            *b_hh_args,
            *W_hr_args,
        )

    def compute_sensitivities(self, X_val, h0_val, c0_val):
        """
        Computes the sensitivity dY/dX for ALL inputs.

        Args:
            X_val (np.ndarray): shape (horizon, n_inputs)  [or (1, horizon, n_inputs) if batch_first True]
            h0_val (np.ndarray): shape (h_out,)  # h_out = proj_size>0 ? proj_size : hidden_size
            c0_val (np.ndarray): shape (hidden_size,)

        Returns:
            sens (np.ndarray): dY/dX stacked by feature, shape (horizon * h_out, horizon, n_inputs)
                            where sens[:, :, j] = dY/dX[:, column_of_feature_j_over_time]
        """
        # Build of jac_func (identical to the one used in _build_function), only once
        if not hasattr(self, "jac_func"):
            jac_sym = cs.jacobian(self.Y, self.X)

            W_ih_args = [w for w in self.W_ih]
            W_hh_args = [w for w in self.W_hh]
            b_ih_args = [b for b in self.b_ih]
            b_hh_args = [b for b in self.b_hh]
            # if proj_size==0, W_hr contains None → use HxH identity
            W_hr_args = [
                w if w is not None else cs.DM.eye(self.hidden_size) for w in self.W_hr
            ]

            self.jac_func = cs.Function(
                "jac_lstm",
                [
                    self.X,
                    self.h0,
                    self.c0,
                    *W_ih_args,
                    *W_hh_args,
                    *b_ih_args,
                    *b_hh_args,
                    *W_hr_args,
                ],
                [jac_sym],
                ["X", "h0", "c0"]
                + [f"W_ih_l{l}" for l in range(self.num_layers)]
                + [f"W_hh_l{l}" for l in range(self.num_layers)]
                + [f"b_ih_l{l}" for l in range(self.num_layers)]
                + [f"b_hh_l{l}" for l in range(self.num_layers)]
                + [f"W_hr_l{l}" for l in range(self.num_layers)],
                ["dY_dX"],
            )

        # h0_val must have dimension h_out (covers proj_size==0 or >0)
        vals = [X_val, h0_val, c0_val]
        vals += [self.W_ih_val[l] for l in range(self.num_layers)]
        vals += [self.W_hh_val[l] for l in range(self.num_layers)]
        vals += [self.b_ih_val[l] for l in range(self.num_layers)]
        vals += [self.b_hh_val[l] for l in range(self.num_layers)]
        vals += [
            self.W_hr_val[l] for l in range(self.num_layers)
        ]  # HxH identity if proj_size==0

        jac_dm = self.jac_func(*vals)[0]  # DM/MX
        jac_np = np.array(jac_dm)  # (horizon*h_out, horizon*n_inputs)

        # CasADi indices for matrix X (column order):
        # column (t, j) -> idx = t + j * horizon
        # We build the columns of each feature j over time t=0..T-1
        T = self.horizon
        J = self.n_inputs

        sens_list = []
        for j in range(J):
            idx_j = [t + j * T for t in range(T)]
            sens_j = jac_np[:, idx_j]  # (horizon*h_out, horizon)
            sens_list.append(sens_j)

        # Stack along the features axis: (horizon*h_out, horizon, n_inputs)
        sens = np.stack(sens_list, axis=2)

        return sens


class CasadiLSTM:
    """
    CasADi LSTM for neural MPC: builds symbolic LSTM functions, loads PyTorch
    weights, and runs estimate-then-predict forward passes with I/O adaptation.

    Usage
    -----
    >>> model = CasadiLSTM(
    ...     n_context=10, n_inputs=2, hidden_size=128,
    ...     horizon=20, proj_size=4, input_order="y_then_u",
    ... )
    >>> model.load_state_dict(torch.load("model.pt"))
    >>> mpc.set_neural_dynamics(model=model, input_order="y_then_u", output_bias=b)

    Modes
    -----
    - Estimator (default): use the first `n_context` samples to estimate internal
      states (`hn`, `cn`), then predict the next `horizon` steps from inputs.
      `forward()` returns output with shape `(output_size, sequence_length)`.
    - Pure forward: set `is_estimator=False` to call the underlying model directly
      with stored `hn`, `cn`.

    Parameters
    ----------
    n_context : int
        Number of measured context samples for state estimation.
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
    is_estimator : bool, default True
        Whether to use context-window state estimation.
    input_order : str, default "y_then_u"
        How features are stacked in the input: "y_then_u" or "u_then_y".
        Must match the `input_order` passed to `set_neural_dynamics`.

    Attributes
    ----------
    output_size : int
        Output dimension (proj_size if >0, else hidden_size).
    sequence_length : int
        Total sequence length (n_context + horizon).
    hn, cn : array_like or None
        Last hidden/cell states (set by `estimate_state`).
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
        is_estimator: bool = True,
        input_order: str = "y_then_u",
    ):
        self.n_context = n_context
        self.horizon = horizon
        self.n_inputs = n_inputs
        self.sequence_length = n_context + horizon
        self.is_estimator = is_estimator
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.h_out = proj_size if proj_size > 0 else hidden_size
        self.output_size = self.h_out
        self.input_order = input_order

        self._core = _CasadiLSTMCore(
            n_inputs=n_inputs,
            hidden_size=hidden_size,
            proj_size=proj_size,
            prediction_horizon=horizon,
            num_layers=num_layers,
            bias=bias,
        )
        self.hn = None
        self.cn = None

    def load_state_dict(self, state_dict):
        """Load PyTorch-style weights into the LSTM.

        Accepts state_dicts with or without 'model.' prefix on keys.

        Parameters
        ----------
        state_dict : dict
            PyTorch state_dict containing LSTM weights/biases.
        """
        self._core.load_state_dict(state_dict)

    def forward(self, inp):
        """Forward pass with I/O adaptation for MPC integration.

        Accepts stacked features from `set_neural_dynamics`, transposes and
        reorders them, runs estimate-then-predict, and returns normalized output.

        Parameters
        ----------
        inp : cs.MX, cs.DM, or torch.Tensor
            Stacked features with shape (output_size + n_inputs, sequence_length).
            Column ordering depends on `input_order`.

        Returns
        -------
        cs.MX or cs.DM
            Model output with shape (output_size, sequence_length).
        """
        # Transpose + reorder: (n_features, T) → (T, n_features) in [u, y] order
        feats_t = self._reorder(inp)

        if self.is_estimator:
            # feats_t columns are [u_0..u_{n_inputs-1}, y_0..y_{h_out-1}]
            u_cols = feats_t[:, : self.n_inputs]
            y_cols = feats_t[:, self.n_inputs :]

            y_est = self.estimate_state(u_cols, y_cols, self.n_context)
            y_pred = self.predict_state(u_cols, self.n_context)
            y_sim = cs.horzcat(y_est, y_pred)
        else:
            self.hn = np.zeros(self.h_out)
            self.cn = np.zeros(self.hidden_size)

            u_cols = feats_t[:, : self.n_inputs]
            y_cols = feats_t[:, self.n_inputs :]

            y_est = np.zeros((1, self.n_context * self.h_out))
            y_pred = self.predict_state(u_cols, self.n_context)
            y_sim = cs.horzcat(y_est, y_pred)

        return self._normalize_output(y_sim)

    def __call__(self, inp):
        return self.forward(inp)

    def estimate_state(self, u_train, y_train, nstep):
        """Estimate RNN hidden states using measured outputs over context window.

        Uses measured outputs (y_train) as pseudo-inputs to warm up the RNN's
        internal hidden and cell states, mimicking teacher forcing during training.

        Parameters
        ----------
        u_train : array_like or cs.MX
            Input sequence, shape (sequence_length, n_inputs).
        y_train : array_like or cs.MX
            Measured output sequence, shape (sequence_length, h_out).
        nstep : int
            Number of context steps to use for state estimation.

        Returns
        -------
        cs.MX
            Estimated outputs over context window, shape (1, nstep * h_out).
        """
        y_est = []
        # Initialize hidden and cell states
        hn = np.zeros(self.h_out)
        cn = np.zeros(self.hidden_size)
        for i in range(nstep):
            # Feed in the known output to estimate state
            out, hn, cn = self._core.forward_one_step(
                cs.reshape(u_train[i, :], 1, self.n_inputs),
                cs.reshape(y_train[i, :], self.h_out, 1),
                cn,
            )
            y_est.append(out)

        y_sim = cs.horzcat(*y_est)
        self.hn, self.cn = hn, cn  # Store hidden and cell states for prediction
        return y_sim

    def predict_state(self, u_train, nstep):
        """Predict outputs using stored hidden states from estimation phase.

        Parameters
        ----------
        u_train : array_like or cs.MX
            Full input sequence, shape (sequence_length, n_inputs).
        nstep : int
            Context length; predictions start from u_train[nstep:, :].

        Returns
        -------
        cs.MX
            Predicted outputs over prediction horizon, shape (1, horizon * h_out).
        """
        y_sim, _, _ = self._core.forward(u_train[nstep:, :], self.hn, self.cn)
        return y_sim

    def get_model(self):
        """Get the underlying core LSTM model.

        Returns
        -------
        _CasadiLSTMCore
            The inner LSTM model instance.
        """
        return self._core

    def compute_sensitivities(self, X_val):
        """Compute Jacobian of outputs w.r.t. inputs over prediction horizon.

        Parameters
        ----------
        X_val : np.ndarray
            Full input features (stacked as per `input_order`),
            shape (output_size + n_inputs, sequence_length).

        Returns
        -------
        np.ndarray
            Jacobian dY/dX with shape (horizon * h_out, horizon, n_inputs).
        """
        X_t = self._reorder(X_val)
        return self._core.compute_sensitivities(
            X_t[self.n_context :, : self.n_inputs], self.hn, self.cn
        )

    def _reorder(self, mat):
        """Transpose and reorder input from (n_features, T) to (T, n_features).

        If input_order is "y_then_u", rotates columns from [y, u] to [u, y].
        """
        if isinstance(mat, (cs.MX, cs.DM)):
            mat_t = mat.T  # (T, n_features)
            if self.input_order == "y_then_u":
                shift = self.output_size
                return cs.horzcat(mat_t[:, shift:], mat_t[:, :shift])
            return mat_t  # already [u, y]
        if isinstance(mat, torch.Tensor):
            mat_t = mat.transpose(0, 1)  # (T, n_features)
            if self.input_order == "y_then_u":
                shift = self.output_size
                return torch.cat((mat_t[:, shift:], mat_t[:, :shift]), dim=1)
            return mat_t  # already [u, y]
        raise TypeError("Input should be CasADi MX/DM or torch.Tensor")

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
