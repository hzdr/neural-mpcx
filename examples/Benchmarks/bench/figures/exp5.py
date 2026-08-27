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

"""Experiment 5 figure: nominal closed-loop performance.

Reproduces the closed-loop panel of the analysis notebooks, 3x2 for the CSTR
(four states plus both inputs) and 3x1 for the two-tank (h_2, h_1, u), from the
nominal configuration the standalone example scripts run. Plant in black,
one-step prediction dotted blue, setpoint dashed green, constraints grey.

This run is the reference the study normalizes against, so its absolute IAE goes
in the caption: elsewhere ``IAE_norm = 1.0`` means as good as this figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .. import config as _config
from .. import plotstyle as ps
from .. import store as _store

import matplotlib.pyplot as plt


EXPERIMENT = "exp5"

#: Panel layout per benchmark: (nrows, ncols, [(row, col, state_index)]).
LAYOUTS = {
    # C_B occupies the first panel because it is the controlled variable.
    "cstr": (3, 2, [(0, 0, 1), (0, 1, 0), (1, 0, 2), (1, 1, 3)]),
    "cts": (3, 1, [(0, 0, 1), (1, 0, 0)]),
}
FIGSIZES = {"cstr": (7.0, 8.25), "cts": (6.0, 7.0)}


def _performance(frame, bench, outdir, root) -> List[Path]:
    if not _store.has_timeseries(EXPERIMENT, bench.key, root=root):
        return []
    neural = frame[frame["controller"] == "neural"]
    if neural.empty:
        return []
    row = neural.iloc[0]
    ts = _store.load_timeseries(EXPERIMENT, bench.key, [row["run_id"]], root=root)
    if ts.empty:
        return []
    ts = ts.sort_values("step")
    t = ts["time_s"].to_numpy()

    nrows, ncols, panels = LAYOUTS[bench.key]
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZES[bench.key],
                             sharex=True, squeeze=False)

    for r, c, s_i in panels:
        ax = axes[r][c]
        key = bench.state_keys[s_i]
        ax.plot(t, ts[key], color=ps.COLOR_PLANT, linestyle="-", linewidth=1.5,
                label="plant")
        pred_col = f"{key}_pred"
        if pred_col in ts:
            ax.plot(t, ts[pred_col], color=ps.COLOR_PRED, linestyle=":",
                    linewidth=1.2, label="pred.")
        if s_i == bench.track_index:
            ax.plot(t, ts["sp"], color=ps.COLOR_SP, linestyle="--",
                    linewidth=1.0, label="setpoint")
        ax.axhline(bench.x_lower[s_i], color="grey", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.axhline(bench.x_upper[s_i], color="grey", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.set_title(bench.state_labels[s_i], fontsize=9)
        ax.set_ylabel(f"{bench.state_labels[s_i]} [{bench.state_units[s_i]}]")
        ps.finish(ax, legend=True, loc="best", fontsize=7, framealpha=0.9)

    # Inputs occupy the bottom row.
    for k, akey in enumerate(bench.action_keys):
        ax = axes[nrows - 1][k if ncols > 1 else 0]
        ax.step(t, ts[akey], where="post", color=ps.COLOR_PRED,
                label=bench.action_labels[k])
        ax.set_title(f"{bench.action_labels[k]} (input)", fontsize=9)
        ax.set_ylabel(f"{bench.action_labels[k]} [{bench.action_units[k]}]")
        ax.set_xlabel("time [s]")
        ps.finish(ax, legend=True, loc="best", fontsize=7, framealpha=0.9)

    # Switch off any leftover panel so it does not render as an empty frame.
    used = {(r, c) for r, c, _ in panels}
    used |= {(nrows - 1, k if ncols > 1 else 0)
             for k in range(len(bench.action_keys))}
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axes[r][c].axis("off")

    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("time [s]")

    fig.suptitle(f"{bench.label}: nominal closed-loop performance",
                 fontsize=11, fontweight="bold")
    ps.caption(
        fig,
        f"Absolute IAE = {row['iae_tracked']:.4g} "
        f"{bench.state_units[bench.track_index]}·s over "
        f"{int(row['n_steps'])} steps; this is the reference every "
        f"IAE_norm in the study is divided by. "
        f"Mean solve {row['solve_ms_mean']:.1f} ms, P99 {row['solve_ms_p99']:.1f} ms, "
        f"worst {row['solve_ms_max']:.1f} ms against a "
        f"{bench.sample_time_s:g} s period. "
        f"Failed solves: {int(row['n_failed_solves'])}.",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    return ps.save(fig, f"exp5_closed_loop_{bench.key}", outdir)


def build(metrics: pd.DataFrame, outdir: Path,
          root: Path | None = None) -> List[Path]:
    """Build the Experiment 5 figure for each benchmark."""
    data = metrics[metrics["experiment"] == EXPERIMENT]
    if data.empty:
        return []
    written: List[Path] = []
    for bench_key, frame in data.groupby("benchmark"):
        written += _performance(frame, _config.BENCHMARKS[bench_key], outdir, root)
    return written
