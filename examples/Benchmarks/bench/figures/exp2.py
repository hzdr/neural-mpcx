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

"""Experiment 2 figure: robustness over a region of initial states.

A neural MPC comes with no recursive-feasibility guarantee, so the region is
established by sampling it. ``exp2_fan`` draws all 50 Latin-hypercube
trajectories, colored by how far each initial state sits from nominal, with a
median, a 5–95 % band and the inadmissible side of each constraint shaded. The
success rate over the same runs is in ``T2_initial_condition_summary``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .. import config as _config
from .. import plotstyle as ps
from .. import store as _store

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


EXPERIMENT = "exp2"


def _x0_array(frame: pd.DataFrame) -> np.ndarray:
    """Stack the stored ``x0`` lists into an array."""
    return np.array([np.asarray(v, dtype=float) for v in frame["x0"]])


#: Box half-widths are declared in ``configs/`` as round percentages, so
#: :func:`_box_percent` snaps to this grid.
BOX_PERCENT_STEP = 5.0


def _box_percent(x0: np.ndarray, nominal: np.ndarray) -> float:
    """Half-width of the sampled box, as a percentage of the nominal state.

    Read off the sampled points instead of the config, so the figure cannot
    quote a range the runs do not cover.

    A Latin hypercube draws inside the outermost stratum, so its extreme point
    sits a little short of the box edge and the raw maximum understates the
    declared half-width by a margin that depends on both the sample size and the
    number of states. Snapping to the nearest :data:`BOX_PERCENT_STEP` absorbs
    that margin. It assumes the declared half-width is a multiple of that step,
    which every ``lhs_x0`` case in ``configs/`` and ``configs_smoke/`` is; a case
    declaring, say, 12 % would be reported as 10 %.
    """
    nz = nominal != 0.0
    if not nz.any() or x0.shape[0] < 2:
        return float("nan")
    raw = float(np.max(np.abs(x0[:, nz] / nominal[nz] - 1.0)))
    step = BOX_PERCENT_STEP
    return step * round(100.0 * raw / step)


def draw_fan(ax, ax_u, lhs, bench, ts) -> dict | None:
    """Draw the initial-condition fan onto a supplied state and input axis."""
    tracked = bench.state_keys[bench.track_index]
    x0 = _x0_array(lhs)
    nominal = np.asarray(bench.x_nominal, dtype=float)
    dist = np.linalg.norm(x0 - nominal[None, :], axis=1)
    norm = Normalize(vmin=dist.min(), vmax=dist.max())
    cmap = ps.trace_cmap()

    curves = []
    for run_id, d in zip(lhs["run_id"], dist):
        run = ts[ts.run_id == run_id].sort_values("step")
        if run.empty:
            continue
        ax.plot(run["time_s"], run[tracked], color=cmap(norm(d)),
                linewidth=0.6, alpha=0.5, zorder=2)
        ax_u.plot(run["time_s"], run[bench.action_keys[0]], color=cmap(norm(d)),
                  linewidth=0.5, alpha=0.45, zorder=2)
        curves.append(run[tracked].to_numpy())

    if not curves:
        return None

    n = min(len(c) for c in curves)
    stack = np.vstack([c[:n] for c in curves])
    first = ts[ts.run_id == lhs["run_id"].iloc[0]].sort_values("step")
    t = first["time_s"].to_numpy()[:n]
    lo, med, hi = np.nanpercentile(stack, [5, 50, 95], axis=0)
    ax.fill_between(t, lo, hi, color=ps.COLOR_PRED, alpha=0.22, linewidth=0,
                    zorder=3, label="5–95 %")
    ax.plot(t, med, color=ps.COLOR_PLANT, linewidth=1.8, zorder=4,
            label="median")
    ax.plot(t, first["sp"].to_numpy()[:n], color=ps.COLOR_SP,
            linestyle="--", linewidth=1.2, zorder=5, label="setpoint")

    ax.set_ylabel(f"{bench.state_labels[bench.track_index]} "
                  f"[{bench.state_units[bench.track_index]}]")
    ps.shade_inadmissible(ax, bench.x_lower[bench.track_index],
                          bench.x_upper[bench.track_index])
    ax_u.set_ylabel(f"{bench.action_labels[0]} [{bench.action_units[0]}]")
    ax_u.set_xlabel("time [s]")
    ps.finish(ax, legend=False)
    ps.finish(ax_u, legend=False)

    return {
        "mappable": plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        "cbar_label": r"$\|x_0 - x_{\mathrm{nom}}\|$",
        "box_percent": _box_percent(x0, nominal),
        "n_total": len(lhs),
        "n_completed": int(lhs["completed"].sum()),
    }


def _caption_text(info) -> str:
    """The initial-condition fan's caption, shared by both figures."""
    return (
        f"Latin-hypercube starts inside a ±{info['box_percent']:.0f} % box "
        f"around the nominal state, each run to completion. This sample is the "
        f"study's empirical substitute for a recursive-feasibility guarantee. "
        f"Shading marks the inadmissible side of each state constraint. "
        f"N_total = {info['n_total']}, N_completed = {info['n_completed']}."
    )


def _fan(frame, bench, outdir, root) -> List[Path]:
    lhs = frame[frame["case"] == "lhs50"]
    if lhs.empty or not _store.has_timeseries(EXPERIMENT, bench.key, root=root):
        return []

    ts = _store.load_timeseries(EXPERIMENT, bench.key, lhs["run_id"], root=root)
    fig, (ax, ax_u) = plt.subplots(
        2, 1, figsize=(6.0, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    info = draw_fan(ax, ax_u, lhs, bench, ts)
    if info is None:
        plt.close(fig)
        return []

    ax.set_title(f"{bench.label}: {len(lhs)} initial conditions, "
                 f"±{info['box_percent']:.0f} % around nominal", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(info["mappable"], ax=[ax, ax_u], label=info["cbar_label"],
                 fraction=0.04, pad=0.02)

    ps.caption(fig, _caption_text(info))
    return ps.save(fig, f"exp2_fan_{bench.key}", outdir)


def build(metrics: pd.DataFrame, outdir: Path,
          root: Path | None = None) -> List[Path]:
    """Build every Experiment 2 figure that the store supports."""
    data = metrics[metrics["experiment"] == EXPERIMENT]
    if data.empty:
        return []
    written: List[Path] = []
    for bench_key, frame in data.groupby("benchmark"):
        bench = _config.BENCHMARKS[bench_key]
        written += _fan(frame, bench, outdir, root)
    return written
