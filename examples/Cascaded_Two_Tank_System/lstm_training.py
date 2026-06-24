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

"""Train an LSTM plant model for Neural MPC on the cascaded two-tank system.

This is a didactic, end-to-end recipe for identifying a recurrent neural
network (LSTM) that can later be used as the *prediction model* inside the
Neural MPC controller in ``neural_mpc_cts.py``. The script teaches the full
workflow:

1. load the cascaded two-tank benchmark dataset (Ts = 4 s);
2. cut the long input/output sequence into short, overlapping training windows;
3. train an LSTM with a **context-window hidden-state warmup** -- the key idea
   that makes recurrent dynamics usable inside a constrained MPC: instead of
   starting every rollout from a zeroed hidden state, the first ``n_context``
   samples are teacher-forced with the measured output to *estimate* the initial
   hidden/cell state ``(h0, c0)``, and the prediction loss is only applied
   *after* that warmup;
4. evaluate the identified model on held-out data and visualise the fit.

The same ``(h0, c0)`` estimation is reproduced numerically at deployment by
``CasadiLSTM`` (see ``neural_mpc_cts.py``), so a model trained here drops
directly into the controller.

Usage
-----
Run from the ``examples/Cascaded_Two_Tank_System/`` directory::

    python lstm_training.py

The script expects ``data/dataBenchmark.csv`` (shipped with the repository).
With ``RUN_TRAINING = True`` it trains a fresh model and saves it to
``models/<REPRO_MODEL_NAME>.pt`` (the shipped ``models/<MODEL_NAME>.pt`` that the
controller consumes is never overwritten). With ``RUN_TRAINING = False`` it skips
training and only evaluates/plots the shipped model.

References
----------
.. [1] Forgione, M., & Piga, D. (2021). "dynoNet: A neural network architecture
       for learning dynamical systems"; and Forgione et al. (2023), "On the
       adaptation of recurrent neural networks for system identification",
       which introduce the initial-state estimation used here.
       https://www.sciencedirect.com/science/article/pii/S0005109823002510
.. [2] Adhau, S. et al. (2024). "Reinforcement learning based MPC with neural
       dynamical models." https://doi.org/10.1016/j.ifacol.2024.07.005
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.ticker import AutoMinorLocator
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torchid import metrics
from torch_wrapper import LSTM

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
try:
    _CURRENT_DIR = Path(__file__).resolve().parent  # type: ignore[name-defined]
except NameError:
    _CURRENT_DIR = Path.cwd()
_PROJECT_ROOT = _CURRENT_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# -----------------------------------------------------------------------------
# USER CONFIGURATION FOR TRAINING THE LSTM RNN
# -----------------------------------------------------------------------------
# When False the script does not train; it loads the shipped MODEL_NAME and only
# produces the evaluation plots (fast). When True it trains a fresh model and
# saves it under REPRO_MODEL_NAME, leaving the shipped model untouched.
RUN_TRAINING = True

HIDDEN_SIZE = 128       # LSTM hidden state width
DROPOUT_RATE = 0.0      # dropout probability (0 disables it)
WEIGHT_DECAY = 1e-4     # Adam L2 regularization
N_CONTEXT = 5           # context-window length used to estimate (h0, c0)
SPLIT_RATIO = 0.8       # fraction of the estimation sequence used for training
BATCH_SIZE = 16 * 2     # windows per mini-batch
WINDOW_SIZE = 100       # length of each training window (context + prediction)
PATIENCE = 150 * 2      # early-stopping patience (epochs without val improvement)
NUM_ITER = 10000        # maximum number of epochs
LR = 1e-3 * 2           # Adam learning rate
MODEL_NAME = "cts-lstm-batched-128"            # shipped model consumed by the MPC
REPRO_MODEL_NAME = "cts-lstm-batched-128-repro"  # where a fresh run is saved

Ts = 4  # sampling time [s]
TIME_UNIT = "s"  # unit shown on time axes

# Windowing / regularization inspired by Forgione & Piga (2021)
STRIDE = 2          # step between consecutive training windows
Q_REG = 0.1         # state-consistency regularization weight (warmup region)
Q_MSE = 1.0         # prediction MSE weight (region after the context window)
Q_YBOX = 0.0        # soft box-constraint penalty weight on the predictions
LR_SCHEDULER = True  # enable ReduceLROnPlateau
LR_PATIENCE = 50    # scheduler patience (epochs before reducing LR)
LR_FACTOR = 0.5     # LR reduction factor

# -----------------------------------------------------------------------------
# NORMALIZATION (disabled for the two-tank system: signals are already O(1-10))
# -----------------------------------------------------------------------------
NORMALIZE_DATA = False

U_NORM_PARAMS = {"U": {"min": 0.0, "max": 10.0}}
Y_NORM_PARAMS = {"H_2": {"min": 0.0, "max": 10.0}}

U_KEYS = list(U_NORM_PARAMS.keys())
Y_KEYS = list(Y_NORM_PARAMS.keys())

# Engineering units used for axis labels (one entry per output / input channel).
Y_UNITS = ["m"]
U_UNITS = ["V"]

# -----------------------------------------------------------------------------
# PLOT STYLE (colorblind-safe palette, consistent with the MPC example scripts)
# -----------------------------------------------------------------------------
COLOR_REAL = "#000000"  # measured / ground-truth signal
COLOR_PRED = "#0072B2"  # model simulation
COLOR_CTX = "#E69F00"   # context-window (teacher-forced warmup) region
COLOR_VAL = "#56B4E9"   # validation split / secondary signal


# -----------------------------------------------------------------------------
# DATA NORMALIZATION HELPERS
# -----------------------------------------------------------------------------
def normalize_data(data_np, params, keys):
    """Min-max scale each column to [0, 1] using ``params``."""
    data_norm = np.copy(data_np)
    for i, key in enumerate(keys):
        lo, hi = params[key]["min"], params[key]["max"]
        data_norm[:, i] = (data_np[:, i] - lo) / (hi - lo)
    return data_norm


def denormalize_data(data_norm_np, params, keys):
    """Invert :func:`normalize_data`, returning engineering units."""
    data_eng = np.copy(data_norm_np)
    for i, key in enumerate(keys):
        lo, hi = params[key]["min"], params[key]["max"]
        data_eng[:, i] = data_norm_np[:, i] * (hi - lo) + lo
    return data_eng


# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
def loss_box_y(y_hat, n_context, lo=0.0, hi=1.0, p=2):
    """Soft box-constraint penalty on the predicted outputs after the warmup."""
    under = torch.relu(lo - y_hat[:, n_context:, :])
    over = torch.relu(y_hat[:, n_context:, :] - hi)
    return (under.pow(p) + over.pow(p)).mean()


def compute_loss(y_sim, y_true, n_context, num_outputs, y_box_lo, y_box_hi):
    """Composite training loss, split into context-warmup and prediction parts.

    The sequence is divided at ``n_context``:

    * ``[n_context:]`` is the free-running **prediction** region: the
      mean-squared error that matters for control (``L_mse``).
    * ``[:n_context]`` is the teacher-forced **warmup** region whose output is
      penalised by a state-consistency term (``L_reg``), the analogue of the
      Forgione & Piga regularization that keeps the estimated ``(h0, c0)``
      consistent with the measured dynamics.

    ``L_ybox`` is an optional soft penalty keeping predictions inside physical
    bounds. Returns ``(L_mse, L_reg, L_ybox, total)``.
    """
    per_step = nn.functional.mse_loss(y_sim, y_true, reduction="none")
    per_step = per_step.view(-1, y_sim.shape[1], num_outputs)
    n_batch = per_step.shape[0]
    seq_len = per_step.shape[1]

    L_mse = per_step[:, n_context:, :].sum() / (
        n_batch * (seq_len - n_context) * num_outputs
    )
    L_reg = (
        per_step[:, :n_context, :].sum() / (n_batch * n_context * num_outputs)
        if Q_REG > 0
        else 0.0
    )
    L_ybox = loss_box_y(y_sim, n_context=n_context, lo=y_box_lo, hi=y_box_hi)
    total = Q_MSE * L_mse + Q_REG * L_reg + Q_YBOX * L_ybox
    return L_mse, L_reg, L_ybox, total


# -----------------------------------------------------------------------------
# DATA LOADING AND WINDOWING
# -----------------------------------------------------------------------------
def load_engineering_data():
    """Load the two-tank benchmark CSV into estimation/test arrays.

    The benchmark file stores the estimation experiment in ``uEst``/``yEst`` and
    a separate validation experiment in ``uVal``/``yVal``. Returns four float32
    arrays of shape ``(T, 1)``: ``(u_est, y_est, u_test, y_test)``.
    """
    csv_path = _PROJECT_ROOT / "data" / "dataBenchmark.csv"

    def _col(name):
        return pd.read_csv(
            csv_path, header=0, thousands=",", decimal=".", na_values="NAN",
            usecols=[name],
        ).to_numpy()

    u_est, u_test = _col("uEst"), _col("uVal")
    y_est, y_test = _col("yEst"), _col("yVal")

    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "u_train.npy", u_est)
    np.save(data_dir / "u_test.npy", u_test)
    np.save(data_dir / "y_train.npy", y_est)
    np.save(data_dir / "y_test.npy", y_test)

    return (
        u_est.astype(np.float32),
        y_est.astype(np.float32),
        u_test.astype(np.float32),
        y_test.astype(np.float32),
    )


def make_windowed_loaders(u_eng, y_eng):
    """Split the estimation sequence and cut it into strided training windows.

    The first ``SPLIT_RATIO`` of the sequence feeds the training windows and the
    remainder the validation windows; each window has length ``WINDOW_SIZE`` and
    consecutive windows are ``STRIDE`` samples apart. Returns the two
    :class:`~torch.utils.data.DataLoader` objects and the split index (for
    plotting).
    """
    u = normalize_data(u_eng, U_NORM_PARAMS, U_KEYS) if NORMALIZE_DATA else u_eng
    y = normalize_data(y_eng, Y_NORM_PARAMS, Y_KEYS) if NORMALIZE_DATA else y_eng

    u_seq = torch.tensor(u)
    y_seq = torch.tensor(y)
    num_features = u_seq.shape[1]
    num_outputs = y_seq.shape[1]

    split_idx = int(u_seq.shape[0] * SPLIT_RATIO)
    splits = {
        "train": (u_seq[:split_idx], y_seq[:split_idx]),
        "val": (u_seq[split_idx:], y_seq[split_idx:]),
    }

    loaders = {}
    for name, (u_part, y_part) in splits.items():
        xs, ys = [], []
        for i in range(0, u_part.shape[0] - WINDOW_SIZE + 1, STRIDE):
            xs.append(u_part[i: i + WINDOW_SIZE, :])
            ys.append(y_part[i: i + WINDOW_SIZE, :])
        X = torch.stack(xs).view(-1, WINDOW_SIZE, num_features)
        Y = torch.stack(ys).view(-1, WINDOW_SIZE, num_outputs)
        loaders[name] = DataLoader(
            TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )

    return loaders["train"], loaders["val"], split_idx, num_features, num_outputs


# -----------------------------------------------------------------------------
# MODEL / TRAIN / EVALUATE
# -----------------------------------------------------------------------------
def build_lstm(n_inputs, n_outputs, *, batch_size, sequence_length, device):
    """Instantiate the context-estimating LSTM wrapper on ``device``."""
    return LSTM(
        N_CONTEXT, n_inputs, hidden_size=HIDDEN_SIZE, batch_size=batch_size,
        sequence_length=sequence_length, n_outputs=n_outputs,
        dropout_rate=DROPOUT_RATE,
    ).to(device)


def _epoch_loss(model, loader, num_features, num_outputs, y_box_lo, y_box_hi,
                device, optimizer=None):
    """Run one epoch; trains when ``optimizer`` is given, else evaluates.

    On the training pass it also gathers the lightweight health diagnostics
    borrowed from Karpathy's "makemore part 3" lecture and returns them as a
    dict; the evaluation pass returns ``None`` for the diagnostics. Returns
    ``(avg_loss, diag)``.

    * ``ud_ratio`` -- update-to-data ratio, ``log10(|ΔW| / |W|)`` averaged over
      parameters. makemore targets ~1e-3 (i.e. -3). We use the SGD-equivalent
      step ``LR * grad`` as the proxy ``ΔW``; with Adam the realised step is
      not exactly ``LR * grad``, but the ~1e-3 scale still flags an init/LR that
      moves the weights too little or too much.
    * ``grad_norm`` -- global pre-clip gradient norm (the value
      ``clip_grad_norm_`` returns); compare against the ``max_norm`` threshold.
    * ``sat_pct`` -- LSTM cell-state saturation, the fraction of ``|tanh(c)|``
      above 0.97. A high value signals a vanishing-gradient regime in the
      recurrence. (``nn.LSTM`` hides the true gate activations, so the exposed
      cell state ``model.cn`` is used as the closest available proxy.)
    """
    training = optimizer is not None
    model.train(training)
    total = 0.0
    seq_len = model.sequence_length
    ud_sum = grad_norm_sum = sat_sum = 0.0
    n_steps = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for u_batch, y_batch in loader:
            u_batch = u_batch.to(device).view(-1, num_features)
            y_batch = y_batch.to(device).view(-1, num_outputs)
            inp = torch.cat((u_batch, y_batch), -1)
            y_sim = model(inp)
            y_true = y_batch.view(-1, seq_len, num_outputs)
            _, _, _, loss = compute_loss(
                y_sim, y_true, N_CONTEXT, num_outputs, y_box_lo, y_box_hi
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )
                with torch.no_grad():
                    ratios = [
                        (LR * p.grad).std().div(p.data.std() + 1e-12).log10().item()
                        for p in model.parameters()
                        if p.grad is not None and p.numel() > 1
                    ]
                    ud_sum += float(np.mean(ratios)) if ratios else 0.0
                    grad_norm_sum += float(grad_norm)
                    if model.cn is not None:
                        sat_sum += (
                            (model.cn.detach().tanh().abs() > 0.97).float().mean()
                            .item() * 100.0
                        )
                    n_steps += 1
                optimizer.step()
            total += loss.item()
    avg = total / len(loader)
    if not training:
        return avg, None
    diag = {
        "ud_ratio": ud_sum / max(n_steps, 1),
        "grad_norm": grad_norm_sum / max(n_steps, 1),
        "sat_pct": sat_sum / max(n_steps, 1),
    }
    return avg, diag


def train_lstm(model, train_loader, val_loader, num_features, num_outputs, device):
    """Train with Adam + ReduceLROnPlateau and early stopping.

    Returns a history dict with per-epoch train/val losses, the learning-rate
    trace, the best validation loss and the best model state.
    """
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
        )
        if LR_SCHEDULER
        else None
    )

    if NORMALIZE_DATA:
        y_box_lo, y_box_hi = 0.0, 1.0
    else:
        y_box_lo = torch.tensor(
            [Y_NORM_PARAMS[k]["min"] for k in Y_KEYS], dtype=torch.float32
        ).to(device)
        y_box_hi = torch.tensor(
            [Y_NORM_PARAMS[k]["max"] for k in Y_KEYS], dtype=torch.float32
        ).to(device)

    train_loss, val_loss, lr_history = [], [], []
    ud_history, grad_norm_history, sat_history = [], [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    start = time.time()
    with tqdm(total=NUM_ITER, desc="Training", unit="epoch") as pbar:
        for itr in range(NUM_ITER):
            tr, diag = _epoch_loss(model, train_loader, num_features, num_outputs,
                                   y_box_lo, y_box_hi, device, optimizer=optimizer)
            va, _ = _epoch_loss(model, val_loader, num_features, num_outputs,
                                y_box_lo, y_box_hi, device, optimizer=None)
            train_loss.append(tr)
            val_loss.append(va)
            lr_history.append(optimizer.param_groups[0]["lr"])
            ud_history.append(diag["ud_ratio"])
            grad_norm_history.append(diag["grad_norm"])
            sat_history.append(diag["sat_pct"])

            if scheduler is not None:
                scheduler.step(va)

            pbar.update(1)
            pbar.set_postfix({
                "train": f"{tr:.5f}", "val": f"{va:.5f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "u:d": f"{diag['ud_ratio']:.2f}",
                "patience": patience_counter,
            })

            if va < best_val_loss:
                best_val_loss = va
                # Deep-copy: state_dict() returns references to the live tensors,
                # which the optimizer keeps mutating in place. Snapshotting freezes
                # the best-validation weights so early stopping restores them.
                #best_state = copy.deepcopy(model.state_dict())
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    tqdm.write(f"Early stopping at epoch {itr + 1}.")
                    break

    print(f"\nTraining time: {time.time() - start:.2f}s")
    return {
        "train_loss": train_loss, "val_loss": val_loss, "lr_history": lr_history,
        "ud_history": ud_history, "grad_norm_history": grad_norm_history,
        "sat_history": sat_history,
        "best_val_loss": best_val_loss, "best_state": best_state,
    }


def rollout_full_sequence(model_path, u_eng, y_eng, num_outputs, device):
    """Simulate the whole sequence with the saved model and score the fit.

    The model is warmed up on the first ``N_CONTEXT`` samples (teacher forcing)
    and then runs free. R^2 / RMSE are computed on the prediction region
    ``[N_CONTEXT:]`` in the model's own units (normalized when applicable, as in
    training). Returns ``(y_sim_eng, r2, rmse)`` with ``y_sim_eng`` in
    engineering units for plotting.
    """
    u = normalize_data(u_eng, U_NORM_PARAMS, U_KEYS) if NORMALIZE_DATA else u_eng
    y = normalize_data(y_eng, Y_NORM_PARAMS, Y_KEYS) if NORMALIZE_DATA else y_eng

    seq_len = u.shape[0]
    n_inputs = u.shape[1]
    model = build_lstm(n_inputs, num_outputs, batch_size=1,
                       sequence_length=seq_len, device=device)
    model.eval()
    model.load_state_dict(torch.load(str(model_path), map_location=device))

    u_flat = torch.tensor(u).view(seq_len, -1).to(device)
    y_flat = torch.tensor(y).view(seq_len, -1).to(device)
    inp = torch.cat((u_flat, y_flat), -1)
    with torch.no_grad():
        y_sim = model(inp)

    y_sim_np = y_sim.detach().cpu().numpy().reshape(-1, num_outputs)
    y_true_np = y_flat.detach().cpu().numpy().reshape(-1, num_outputs)
    r2 = metrics.r_squared(y_true_np[N_CONTEXT:], y_sim_np[N_CONTEXT:])
    rmse = metrics.error_rmse(y_true_np[N_CONTEXT:], y_sim_np[N_CONTEXT:])

    y_sim_eng = (
        denormalize_data(y_sim_np, Y_NORM_PARAMS, Y_KEYS)
        if NORMALIZE_DATA
        else y_sim_np
    )
    return y_sim_eng, r2, rmse


# -----------------------------------------------------------------------------
# DIDACTIC PLOTS
# -----------------------------------------------------------------------------
def _style_axes(ax):
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.grid(True, which="major", alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def plot_dataset(u_est, y_est, u_test, y_test, split_idx):
    """Estimation vs. test data, with the train/validation split shaded."""
    t_est = np.arange(u_est.shape[0]) * Ts
    t_test = np.arange(u_test.shape[0]) * Ts
    n_out, n_in = y_est.shape[1], u_est.shape[1]
    n_rows = n_out + n_in

    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 2.0 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)
    fig.suptitle("Identification dataset", fontsize=11, fontweight="bold")

    for col, (t, u, y, title) in enumerate([
        (t_est, u_est, y_est, "Estimation"),
        (t_test, u_test, y_test, "Test"),
    ]):
        for i in range(n_out):
            ax = axes[i, col]
            ax.plot(t, y[:, i], color=COLOR_REAL, linewidth=1.0, label="measured")
            ax.set_ylabel(f"{Y_KEYS[i]} [{Y_UNITS[i]}]")
            ax.set_title(f"{title} — output", fontsize=9)
            if col == 0:
                ax.axvspan(t[split_idx], t[-1], color=COLOR_VAL, alpha=0.12,
                           label="val split")
            _style_axes(ax)
        for j in range(n_in):
            ax = axes[n_out + j, col]
            ax.plot(t, u[:, j], color=COLOR_PRED, linewidth=1.0, label="input")
            ax.set_ylabel(f"{U_KEYS[j]} [{U_UNITS[j]}]")
            ax.set_title(f"{title} — input", fontsize=9)
            if col == 0:
                ax.axvspan(t[split_idx], t[-1], color=COLOR_VAL, alpha=0.12)
            _style_axes(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel(f"time [{TIME_UNIT}]")
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    return fig


def plot_context_window(y_true, u, n_context, zoom_len, y_sim=None,
                        title="Context-window warmup"):
    """Zoom on the start of a sequence to explain the warmup vs. prediction split.

    Shades ``[0, n_context)`` (teacher-forced estimation of ``h0, c0``) and the
    free-running prediction region after it. When ``y_sim`` is given, the model
    simulation is overlaid so the reader sees the fit inside each region.
    """
    k = np.arange(zoom_len)
    fig, (ax_y, ax_u) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.suptitle(title, fontsize=11, fontweight="bold")

    ax_y.plot(k, y_true[:zoom_len, 0], color=COLOR_REAL, linewidth=1.5,
              label="measured")
    if y_sim is not None:
        ax_y.plot(k, y_sim[:zoom_len, 0], color=COLOR_PRED, linestyle=":",
                  linewidth=1.5, label="LSTM simulation")
    ax_y.axvspan(0, n_context - 1, color=COLOR_CTX, alpha=0.18,
                 label="context warmup (estimates $h_0, c_0$)")
    ax_y.axvline(n_context - 1, color=COLOR_CTX, linestyle="--", linewidth=1.0)
    ax_y.set_ylabel(f"{Y_KEYS[0]} [{Y_UNITS[0]}]")
    _style_axes(ax_y)
    ax_y.text(
        (n_context - 1) / 2, ax_y.get_ylim()[1], "teacher\nforced",
        ha="center", va="top", fontsize=7, color="#7a5c00",
    )
    ax_y.text(
        n_context + (zoom_len - n_context) / 2, ax_y.get_ylim()[1],
        "free-running prediction (loss region)",
        ha="center", va="top", fontsize=7, color=COLOR_PRED,
    )

    ax_u.step(k, u[:zoom_len, 0], where="post", color=COLOR_PRED, label="input")
    ax_u.axvspan(0, n_context - 1, color=COLOR_CTX, alpha=0.18)
    ax_u.axvline(n_context - 1, color=COLOR_CTX, linestyle="--", linewidth=1.0)
    ax_u.set_ylabel(f"{U_KEYS[0]} [{U_UNITS[0]}]")
    ax_u.set_xlabel("time step")
    _style_axes(ax_u)

    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    return fig


def plot_loss_curve(history):
    """Train/validation loss (log scale) with best epoch and LR-drop markers."""
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    lr_history = history["lr_history"]
    best_epoch = int(np.argmin(val_loss))

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Training loss", fontsize=11, fontweight="bold")
    epochs = np.arange(len(train_loss))
    ax.semilogy(epochs, train_loss, color=COLOR_PRED, linewidth=1.2, label="train")
    ax.semilogy(epochs, val_loss, color=COLOR_REAL, linewidth=1.2, label="val")
    ax.plot(best_epoch, val_loss[best_epoch], marker="o", color=COLOR_CTX,
            markersize=7, label=f"best (epoch {best_epoch})")

    drops = [i for i in range(1, len(lr_history)) if lr_history[i] < lr_history[i - 1]]
    for d in drops:
        ax.axvline(d, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    if drops:
        ax.plot([], [], color="grey", linestyle="--", linewidth=0.7, label="LR drop")

    ax.set_xlabel("epoch")
    ax.set_ylabel("composite loss")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.grid(True, which="major", alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())  # y-axis is log: no minor ticks
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    return fig


def plot_diagnostics(history):
    """Training-health diagnostics, after Karpathy's "makemore part 3" lecture.

    Three stacked panels vs. epoch:

    * **update-to-data ratio** (``log10``) with the ~1e-3 (= -3) health line --
      below it the weights barely move (init/LR too small), well above it the
      steps are large and noisy;
    * **pre-clip gradient norm** (log scale) with the ``max_norm`` clip
      threshold marked -- shows how often clipping actually bites;
    * **cell-state saturation** (% of ``|tanh(c)| > 0.97``) -- a rising trace
      warns of a vanishing-gradient regime in the recurrence.
    """
    ud = history["ud_history"]
    gn = history["grad_norm_history"]
    sat = history["sat_history"]
    epochs = np.arange(len(ud))

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig.suptitle("Training diagnostics", fontsize=11, fontweight="bold")

    axes[0].plot(epochs, ud, color=COLOR_PRED, linewidth=1.2,
                 label="update:data ratio")
    axes[0].axhline(-3, color=COLOR_REAL, linestyle="--", linewidth=0.9,
                    label="healthy $\\approx$ 1e-3")
    axes[0].set_ylabel("log10 update:data")
    _style_axes(axes[0])

    axes[1].semilogy(epochs, gn, color=COLOR_PRED, linewidth=1.2,
                     label="grad norm (pre-clip)")
    axes[1].axhline(5.0, color=COLOR_CTX, linestyle="--", linewidth=0.9,
                    label="clip max_norm")
    axes[1].set_ylabel("grad norm")
    axes[1].legend(loc="best", fontsize=7, framealpha=0.9)
    axes[1].grid(True, which="major", alpha=0.3)
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())  # y-axis is log

    axes[2].plot(epochs, sat, color=COLOR_PRED, linewidth=1.2,
                 label="$|\\tanh(c)| > 0.97$")
    axes[2].set_ylabel("cell saturation [%]")
    axes[2].set_xlabel("epoch")
    _style_axes(axes[2])

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    return fig


def plot_fit(y_true, y_sim, u, r2, rmse, title):
    """Ground truth vs. simulation per output (+ inputs), annotated with R^2/RMSE."""
    n_out, n_in = y_true.shape[1], u.shape[1]
    n_rows = n_out + n_in
    r2 = np.atleast_1d(r2)
    rmse = np.atleast_1d(rmse)

    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(9, 1.9 * n_rows))
    axes = np.atleast_1d(axes)
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for i in range(n_out):
        ax = axes[i]
        ax.plot(y_true[:, i], color=COLOR_REAL, linewidth=1.5, label="ground truth")
        ax.plot(y_sim[:, i], color=COLOR_PRED, linestyle=":", linewidth=1.5,
                label="LSTM simulation")
        ax.axvline(N_CONTEXT - 1, color=COLOR_CTX, linestyle="--", linewidth=1.0,
                   label="context boundary")
        ax.set_ylabel(f"{Y_KEYS[i]} [{Y_UNITS[i]}]")
        ax.text(
            0.99, 0.04, f"$R^2$ = {r2[i]:.4f}   RMSE = {rmse[i]:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "ec": "grey", "alpha": 0.8},
        )
        _style_axes(ax)
    for j in range(n_in):
        ax = axes[n_out + j]
        ax.step(np.arange(u.shape[0]), u[:, j], where="post", color=COLOR_PRED,
                label="input")
        ax.axvline(N_CONTEXT - 1, color=COLOR_CTX, linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"{U_KEYS[j]} [{U_UNITS[j]}]")
        _style_axes(ax)

    axes[-1].set_xlabel("time step")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    return fig


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # Seeds are fixed before any RNG is consumed so the run is reproducible.
    # The only RNG consumers are the LSTM weight init (in build_lstm) and the
    # shuffled DataLoaders (during training); their order must not change.
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    u_est, y_est, u_test, y_test = load_engineering_data()

    train_loader, val_loader, split_idx, num_features, num_outputs = (
        make_windowed_loaders(u_est, y_est)
    )

    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    history = None
    if RUN_TRAINING:
        model = build_lstm(num_features, num_outputs, batch_size=BATCH_SIZE,
                           sequence_length=WINDOW_SIZE, device=device)
        history = train_lstm(model, train_loader, val_loader, num_features,
                             num_outputs, device)
        if history["best_state"] is not None:
            model.load_state_dict(history["best_state"])
            print(f"Best model restored (val loss: {history['best_val_loss']:.6f})")
        eval_path = models_dir / f"{REPRO_MODEL_NAME}.pt"
        torch.save(model.state_dict(), eval_path)
        print(f"Model saved to: {eval_path}")
    else:
        eval_path = models_dir / f"{MODEL_NAME}.pt"
        assert eval_path.exists(), f"Shipped model not found at '{eval_path}'"
        print(f"Loaded shipped model for evaluation: {eval_path}")

    # --- Evaluation on estimation and test sequences ---
    y_sim_train, r2_tr, rmse_tr = rollout_full_sequence(
        eval_path, u_est, y_est, num_outputs, device
    )
    y_sim_test, r2_te, rmse_te = rollout_full_sequence(
        eval_path, u_test, y_test, num_outputs, device
    )
    print(f"Train R^2: {np.atleast_1d(r2_tr)}  RMSE: {np.atleast_1d(rmse_tr)}")
    print(f"Test  R^2: {np.atleast_1d(r2_te)}  RMSE: {np.atleast_1d(rmse_te)}")

    # --- Figures ---
    plot_dataset(u_est, y_est, u_test, y_test, split_idx)
    if history is not None:
        plot_loss_curve(history)
        plot_diagnostics(history)
    plot_context_window(y_est, u_est, N_CONTEXT, zoom_len=WINDOW_SIZE,
                        y_sim=y_sim_train,
                        title="Context-window warmup (estimation data)")
    plot_fit(y_est, y_sim_train, u_est, r2_tr, rmse_tr, title="Estimation fit")
    plot_fit(y_test, y_sim_test, u_test, r2_te, rmse_te, title="Test fit")
    plt.show()


if __name__ == "__main__":
    main()
