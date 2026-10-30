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

"""Compare context and zero initialization on random held-out LSTM windows.

Run from any directory. Edit PANELS to select/reorder outputs and CASES to
change the sampled windows. Data generation follows the training notebooks;
the figure uses the shared benchmark style and mean +/- standard-deviation bands.
Choose --color-mode color or grayscale for the publication palette.
No training is performed. All model outputs remain available in the saved NPZs.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bench import plotstyle as ps


HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent

# Panels appear in this order. Other CSTR outputs: C_A and T_R.
PANELS = [("cstr", "C_B"), ("cstr", "T_K"), ("cts", "H_2")]
RANDOM_SEED = 0
FORECAST_STEPS = 30  # displayed steps; use None to display the full forecast
COLOR_MODE = "color"  # "color" or "grayscale"; also selectable with --color-mode
CASES = {
    "cstr": {
        "directory": "CSTR",
        "module": "lstm_training_cstr",
        "window_size": 40,
        "n_windows": 1000,
    },
    "cts": {
        "directory": "Cascaded_Two_Tank_System",
        "module": "lstm_training",
        "window_size": 35,
        "n_windows": 900,
    },
}
# Context lengths, normalization and checkpoints come from the training modules:
# CSTR: N_CONTEXT=10, normalized cstr-lstm-batched-16;
# CTS: N_CONTEXT=5, raw cts-lstm-batched-128.

SYMBOLS = {
    "C_A": r"$C_A$", "C_B": r"$C_B$", "T_R": r"$T_R$",
    "T_K": r"$T_K$", "H_2": r"$h_2$",
}


def load_training_module(case):
    """Reuse the notebooks' configuration and model-building helpers."""
    # Both examples use local torch_wrapper/torchid imports, as in the notebooks.
    for cfg in CASES.values():
        directory = str(EXAMPLES / cfg["directory"])
        if directory not in sys.path:
            sys.path.insert(0, directory)
    return importlib.import_module(CASES[case]["module"])


def load_test_data(lt, case):
    """Read the same held-out CSV as the notebook, preserving channel order."""
    # The training loader also writes .npy files; only the held-out CSV is needed.
    data_dir = lt._PROJECT_ROOT / "data"
    if case == "cstr":
        frame = pd.read_csv(data_dir / "cstr_testing_data.csv", na_values="NAN")
        u_columns, y_columns = lt.U_KEYS, lt.Y_KEYS
    else:
        frame = pd.read_csv(data_dir / "dataBenchmark.csv", na_values="NAN")
        u_columns, y_columns = ["uVal"], ["yVal"]
    return (
        frame.loc[:, u_columns].to_numpy(dtype=np.float32),
        frame.loc[:, y_columns].to_numpy(dtype=np.float32),
    )


def generate_errors(lt, case, *, window_size, n_windows, seed=RANDOM_SEED):
    """Run paired forecasts; return absolute errors in engineering units.

    Each sampled window estimates its own state from its measured context.
    The zero-state rollout starts at the same forecast step, with the same inputs.
    Arrays have shape (sampled windows, forecast steps, all model outputs).
    """
    u_test, y_test = load_test_data(lt, case)
    n_context = lt.N_CONTEXT
    n_possible = len(u_test) - window_size + 1
    if not n_context < window_size <= len(u_test):
        raise ValueError(f"{case}: require N_CONTEXT < window_size <= test length")
    if not 1 <= n_windows <= n_possible:
        raise ValueError(f"{case}: n_windows must be between 1 and {n_possible}")

    starts = np.sort(np.random.default_rng(seed).choice(
        n_possible, n_windows, replace=False,
    ))
    indices = starts[:, None] + np.arange(window_size)
    u = (lt.normalize_data(u_test, lt.U_NORM_PARAMS, lt.U_KEYS)
         if lt.NORMALIZE_DATA else u_test)
    y = (lt.normalize_data(y_test, lt.Y_NORM_PARAMS, lt.Y_KEYS)
         if lt.NORMALIZE_DATA else y_test)
    u_windows = torch.as_tensor(u[indices], dtype=torch.float32)
    y_windows = torch.as_tensor(y[indices], dtype=torch.float32)
    n_inputs, n_outputs = u_test.shape[1], y_test.shape[1]

    model = lt.build_lstm(n_inputs, n_outputs, batch_size=n_windows,
                          sequence_length=window_size, device=torch.device("cpu"))
    checkpoint = lt._PROJECT_ROOT / "models" / f"{lt.MODEL_NAME}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        context = model(torch.cat((u_windows, y_windows), dim=-1))
        hidden = u_windows.new_zeros(1, n_windows, n_outputs)
        cell = u_windows.new_zeros(1, n_windows, lt.HIDDEN_SIZE)
        zero, _ = model.get_model()(u_windows[:, n_context:, :], (hidden, cell))

    def engineering_units(prediction):
        values = prediction.cpu().numpy()
        if lt.NORMALIZE_DATA:
            values = lt.denormalize_data(
                values.reshape(-1, n_outputs), lt.Y_NORM_PARAMS, lt.Y_KEYS,
            ).reshape(values.shape)
        return values

    truth = y_test[indices[:, n_context:]]
    return {
        "error_context": np.abs(truth - engineering_units(context[:, n_context:, :])),
        "error_zero": np.abs(truth - engineering_units(zero)),
        "window_starts": starts,
        "output_keys": np.array(lt.Y_KEYS),
        "output_units": np.array(lt.Y_UNITS),
        "n_context": np.array(n_context),
        "window_size": np.array(window_size),
        "random_seed": np.array(seed),
        "checkpoint": np.array(str(checkpoint.relative_to(EXAMPLES))),
    }


def plot_errors(datasets, panels=PANELS, forecast_steps=FORECAST_STEPS,
                color_mode=COLOR_MODE):
    """Plot mean +/- std bands, with pooled mean/std statistics in each legend."""
    if color_mode not in ("color", "grayscale"):
        raise ValueError("color_mode must be 'color' or 'grayscale'")
    with plt.rc_context(ps.RCPARAMS):
        fig, axes = plt.subplots(len(panels), 1, squeeze=False,
                                 figsize=(6.5, 2.7 * len(panels)),
                                 constrained_layout=True)
        for i, (case, key) in enumerate(panels):
            ax = axes[i, 0]
            data = datasets[case]
            output_index = list(data["output_keys"]).index(key)
            available_steps = data["error_context"].shape[1]
            steps = (available_steps if forecast_steps is None
                     else min(forecast_steps, available_steps))
            horizon = np.arange(1, steps + 1)
            unit = str(data["output_units"][output_index])
            for series_index, (name, label) in enumerate((
                ("error_context", "Context initialized"),
                ("error_zero", "Zero initialized"),
            )):
                style = ps.series_style(series_index)
                if color_mode == "grayscale":
                    # Match the monochrome timing figures; dashes/markers carry
                    # the distinction even when both mean curves use black ink.
                    style["color"] = ps.CATEGORICAL[0]
                errors = data[name][:, :steps, output_index]
                mean_error = errors.mean(axis=0)
                std_error = errors.std(axis=0)
                ax.fill_between(horizon, np.maximum(0, mean_error - std_error),
                                mean_error + std_error, color=style["color"],
                                alpha=0.2, linewidth=0)
                # Pool windows and displayed forecast steps, as in the notebooks.
                ax.plot(horizon, mean_error, **style,
                        markevery=max(1, steps // 6),
                        label=(f"{label}: μ={errors.mean():.3g}, "
                               f"σ={errors.std():.3g} {unit}"))
            unit = unit.replace("°C", r"$^\circ$C")
            ax.set(xlim=(1, max(2, steps)), ylim=(0, None),
                   title=f"({chr(97 + i)}) {case.upper()}: {SYMBOLS.get(key, key)}",
                   ylabel=f"MAE [{unit}]", xlabel="Forecast step")
            ps.finish(ax, loc="upper right")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", metavar="CASE:OUTPUT",
                        default=[f"{case}:{key}" for case, key in PANELS],
                        help="ordered panels, e.g. cstr:C_A cstr:T_K cts:H_2")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--color-mode", choices=("color", "grayscale"),
                        default=COLOR_MODE,
                        help="shared categorical palette or black curves with "
                             "distinct markers and line styles")
    parser.add_argument("--data-dir", type=Path,
                        default=HERE / "results" / "initialization_error")
    args = parser.parse_args()

    panels, modules = [], {}
    for panel in args.panels:
        case, separator, key = panel.partition(":")
        if not separator or case not in CASES:
            parser.error(f"Invalid panel {panel!r}; use cstr:OUTPUT or cts:OUTPUT")
        if case not in modules:
            modules[case] = load_training_module(case)
        if key not in modules[case].Y_KEYS:
            parser.error(f"{case}: choose an output from {modules[case].Y_KEYS}")
        panels.append((case, key))

    torch.set_num_threads(1)
    datasets = {}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for case, lt in modules.items():
        cfg = CASES[case]
        data = generate_errors(lt, case, window_size=cfg["window_size"],
                               n_windows=cfg["n_windows"])
        datasets[case] = data
        path = args.data_dir / f"{case}.npz"
        np.savez_compressed(path, **data)
        print(f"Saved {path} ({data['error_context'].shape}, "
              f"context={int(data['n_context'])})")

    fig = plot_errors(datasets, panels, color_mode=args.color_mode)
    name = "context_vs_zero_prediction"
    if args.color_mode == "grayscale":
        name += "_grayscale"
    with plt.rc_context(ps.RCPARAMS):
        for path in ps.save(fig, name, args.output_dir):
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
