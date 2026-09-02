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

"""Experiment 1 figures: robustness to measurement noise.

Two figures per benchmark: the trajectories banded by noise level, and the
per-run distribution of offset and tracking cost against noise level.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .. import config as _config
from .. import metrics as _metrics
from .. import plotstyle as ps
from .. import store as _store

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


EXPERIMENT = "exp1"

#: Replicates drawn as individual traces on the two-tank fan. Its runs are 1050
#: steps against the CSTR's 60, so all 20 at all six noise levels would put
#: ~126k vector points in one figure. The bands and medians use the full
#: population; the thinning applies to the drawn traces alone, and the indices
#: are fixed so the figure stays reproducible.
TRACE_REPLICATES = (0, 4, 8, 12, 16)


def draw_fan(ax, ax_u, frame, bench, ts) -> dict | None:
    """Draw the noise fan onto a supplied state axis and input axis.

    Returns the colorbar mappable and the counts the caption needs, or None
    when the store held no trajectory for any run.
    """
    sigmas = sorted(frame["noise_sigma_pct"].unique())
    tracked = bench.state_keys[bench.track_index]
    thin = bench.key == "cts"
    colors = ps.ordinal(len(sigmas))
    pooled = []

    for i, sigma in enumerate(sigmas):
        runs = frame[frame["noise_sigma_pct"] == sigma]
        color = colors[i]

        curves, inputs, times = [], [], []
        for run_id, replicate in zip(runs["run_id"], runs["replicate"]):
            run = ts[ts.run_id == run_id].sort_values("step")
            if run.empty:
                continue
            curves.append(run[tracked].to_numpy())
            inputs.append(run[bench.action_keys[0]].to_numpy())
            times.append(run["time_s"].to_numpy())
            if thin and int(replicate) not in TRACE_REPLICATES:
                continue
            ax.plot(run["time_s"], run[tracked], color=color, linewidth=0.5,
                    alpha=0.25, zorder=2)
            ax_u.plot(run["time_s"], run[bench.action_keys[0]], color=color,
                      linewidth=0.5, alpha=0.2, zorder=2)
        if not curves:
            continue

        n = min(len(c) for c in curves)
        t = times[0][:n]
        stack = np.vstack([c[:n] for c in curves])
        pooled.append(stack)
        lo, med, hi = np.nanpercentile(stack, [5, 50, 95], axis=0)
        ax.fill_between(t, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=3,
                        label="5–95 %" if i == 0 else None)
        # The colorbar carries the level now, so the medians share one style.
        ax.plot(t, med, color=color, linewidth=1.4, zorder=4)
        u_med = np.nanmedian(np.vstack([u[:n] for u in inputs]), axis=0)
        ax_u.plot(t, u_med, color=color, linewidth=1.2, zorder=4)

    if not pooled:
        return None

    first = ts[ts.run_id == frame["run_id"].iloc[0]].sort_values("step")
    n = min(s.shape[1] for s in pooled)
    t = first["time_s"].to_numpy()[:n]
    ax.plot(t, np.nanmedian(np.vstack([s[:, :n] for s in pooled]), axis=0),
            color=ps.COLOR_PLANT, linewidth=1.8, zorder=6,
            label="median")
    ax.plot(first["time_s"], first["sp"], color=ps.COLOR_SP, linestyle="--",
            linewidth=1.2, zorder=5, label="setpoint")

    ax.set_ylabel(f"{bench.state_labels[bench.track_index]} "
                  f"[{bench.state_units[bench.track_index]}]")
    ps.shade_inadmissible(ax, bench.x_lower[bench.track_index],
                          bench.x_upper[bench.track_index])
    ax_u.set_ylabel(f"{bench.action_labels[0]} [{bench.action_units[0]}]")
    ax_u.set_xlabel("time [s]")
    ps.finish(ax, legend=False)
    ps.finish(ax_u, legend=False)

    sm = plt.cm.ScalarMappable(
        norm=BoundaryNorm(np.arange(len(sigmas) + 1), len(sigmas)),
        cmap=ListedColormap(colors),
    )
    return {
        "mappable": sm,
        "cbar_label": r"$\sigma$ [% of range]",
        "cbar_ticks": np.arange(len(sigmas)) + 0.5,
        "cbar_ticklabels": [f"{s:g}" for s in sigmas],
        "n_total": len(frame),
        "n_completed": int(frame["completed"].sum()),
        "n_reps": frame["replicate"].nunique(),
        "thinned": thin,
    }


def _caption_text(info) -> str:
    """The noise fan's caption, shared by the standalone and combined figures."""
    thinning = (
        f" Traces shown for replicates "
        f"{', '.join(str(r) for r in TRACE_REPLICATES)}; bands use all "
        f"{info['n_reps']}."
        if info["thinned"] else ""
    )
    return (
        f"One band and one colored median per noise level, over {info['n_reps']} "
        f"seeded realizations each, common random numbers throughout; the "
        f"black line is the median over every level. Shading marks the "
        f"inadmissible side of the constraint.{thinning} "
        f"{info['n_completed']} of {info['n_total']} runs completed."
    )


def _fan(frame, bench, ts, outdir) -> List[Path]:
    """Trajectories banded by noise level, with the input below."""
    fig, (ax, ax_u) = plt.subplots(
        2, 1, figsize=(6.0, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    info = draw_fan(ax, ax_u, frame, bench, ts)
    if info is None:
        plt.close(fig)
        return []

    ax.set_title(f"{bench.label}: closed loop under measurement noise",
                 fontsize=10)
    ax.legend(loc="best", fontsize=7.5)
    bar = fig.colorbar(info["mappable"], ax=[ax, ax_u],
                       label=info["cbar_label"], fraction=0.04, pad=0.02)
    bar.set_ticks(info["cbar_ticks"], labels=info["cbar_ticklabels"])

    ps.caption(fig, _caption_text(info))
    return ps.save(fig, f"exp1_fan_{bench.key}", outdir)


def _offset_dots(frame, bench, outdir) -> List[Path]:
    """One dot per run: steady-state offset and tracking cost against noise."""
    sigmas = sorted(frame["noise_sigma_pct"].unique())
    if not sigmas:
        return []
    rng = np.random.default_rng(0)  # jitter is cosmetic; no value depends on it
    ink = ps.CATEGORICAL[0]

    fig, (ax_off, ax_iae) = plt.subplots(
        1, 2, figsize=(7.6, max(3.0, 0.55 * len(sigmas) + 1.4)), sharey=True,
    )

    n_dropped = 0
    n_no_offset = 0
    dropped_rows: List[int] = []
    for row, sigma in enumerate(sigmas):
        cell = frame[frame["noise_sigma_pct"] == sigma]
        offsets = cell["ss_offset"].to_numpy(float)
        iae = cell["iae_norm"].to_numpy(float)

        # The offset is signed, so median_iqr's "rank non-finite worst" rule
        # does not apply: it would report a diverged run as a large positive
        # offset and invent a direction. The bar is the median of the runs that
        # produced a finite offset, and the caption carries the count of those
        # that did not.
        finite_off = np.isfinite(offsets)
        n_no_offset += int((~finite_off).sum())
        if finite_off.any():
            y = row + rng.uniform(-0.22, 0.22, size=int(finite_off.sum()))
            ax_off.scatter(offsets[finite_off], y, s=13, alpha=0.7, color=ink,
                           edgecolor="none", zorder=3)
            ax_off.vlines(np.median(offsets[finite_off]), row - 0.34, row + 0.34,
                          color="black", linewidth=1.8, zorder=5)

        # A log axis has no room for a non-finite or non-positive value. Those
        # runs go to the right edge and into the caption instead of vanishing.
        plottable = np.isfinite(iae) & (iae > 0.0)
        if (~plottable).any():
            n_dropped += int((~plottable).sum())
            dropped_rows.append(row)
        if plottable.any():
            y = row + rng.uniform(-0.22, 0.22, size=int(plottable.sum()))
            ax_iae.scatter(iae[plottable], y, s=13, alpha=0.7, color=ink,
                           edgecolor="none", zorder=3)
            # IAE is positive and larger is worse, so the same rule the tables
            # use applies: non-finite runs rank worst and move the median.
            iae_median = _metrics.median_iqr(iae)["median"]
            if np.isfinite(iae_median) and iae_median > 0.0:
                ax_iae.vlines(iae_median, row - 0.34, row + 0.34,
                              color="black", linewidth=1.8, zorder=5)

    ax_off.axvline(0.0, color=ps.COLOR_THRESHOLD, linewidth=1.4, zorder=2)
    ax_off.set_xlabel(f"steady-state offset "
                      f"[{bench.state_units[bench.track_index]}]")
    ax_off.set_ylabel(r"measurement noise $\sigma$ [% of range]")

    ps.log_axis_plain(ax_iae)
    ax_iae.axvline(1.0, color=ps.COLOR_THRESHOLD, linestyle="--", linewidth=1.0,
                   zorder=2)
    if dropped_rows:
        edge = ax_iae.get_xlim()[1]
        ps.annotate_failures(ax_iae, [edge] * len(dropped_rows), dropped_rows,
                             [True] * len(dropped_rows), label="non-finite IAE")
        ax_iae.set_xlim(right=edge)
        ax_iae.legend(loc="best", fontsize=7.5)
    ax_iae.set_xlabel(r"IAE$_{\mathrm{norm}}$ [–]")

    ax_off.set_yticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
    ax_off.set_ylim(-0.7, len(sigmas) - 0.3)
    for ax in (ax_off, ax_iae):
        # Minor ticks mean nothing on evenly spaced categorical rows.
        ps.finish(ax, minor=False, legend=False)

    fig.suptitle(f"{bench.label}: offset and tracking cost per noise level",
                 fontsize=10)
    undrawn = []
    if n_no_offset:
        undrawn.append(f"{n_no_offset} run(s) left no finite offset")
    if n_dropped:
        undrawn.append(f"{n_dropped} had no finite positive IAE and cannot sit "
                       f"on a log axis")
    missing = (f" {' and '.join(undrawn)}; both counts are inside the "
               f"{len(frame)} total." if undrawn else "")
    ps.caption(
        fig,
        f"One dot per run, {len(frame)} in total, jittered within its noise "
        f"level. Left of the zero line the controller settles below the "
        f"setpoint, right of it above. The offset bar is the median over the "
        f"runs with a finite offset; the IAE bar ranks non-finite runs worst, "
        f"as the tables do.{missing}",
    )
    fig.tight_layout()
    return ps.save(fig, f"exp1_offset_dots_{bench.key}", outdir)


def build(metrics: pd.DataFrame, outdir: Path,
          root: Path | None = None) -> List[Path]:
    """Build every Experiment 1 figure that the store supports."""
    data = metrics[metrics["experiment"] == EXPERIMENT]
    if data.empty:
        return []

    written: List[Path] = []
    for bench_key, frame in data.groupby("benchmark"):
        bench = _config.BENCHMARKS[bench_key]
        written += _offset_dots(frame, bench, outdir)
        if _store.has_timeseries(EXPERIMENT, bench_key, root=root):
            ts = _store.load_timeseries(EXPERIMENT, bench_key, frame["run_id"],
                                        root=root)
            written += _fan(frame, bench, ts, outdir)
    return written
