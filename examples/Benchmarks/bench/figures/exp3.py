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

"""Experiment 3 figures: plant-model mismatch and unmeasured disturbances.

Mismatch and disturbances answer two different questions, and the figures keep
them apart. Mismatch measures the envelope: how far the plant may drift from the
model before the controller stops being acceptable. A disturbance step measures
the transient: how the loop rejects it and what offset it leaves behind.

``exp3_fan``
    Every trajectory of the narrow mismatch sample, colored by how far its plant
    sits from the model, with a median and a 5–95 % band.
``exp3_disturbance``
    Response to the unmeasured step, onset shaded, one line per magnitude, input
    below, residual offset marked with a labeled bracket.
``exp3_offset_dots``
    Signed steady-state offset across every mismatch case, diverging about zero.
``exp3_envelope_iae``, ``exp3_envelope_offset``
    The wide mismatch sample as a filled field over the two plant parameters,
    once for normalized IAE and once for signed offset, with the sample drawn on
    top.
``exp3_neural_vs_nmpc``
    Paired dot plot under identical mismatch. Both controllers hold a nominal
    model, so the comparison measures whether the LSTM degrades worse than the
    model it replaced. CSTR only: the two-tank has no physics-MPC counterpart.
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


EXPERIMENT = "exp3"

#: The plant parameters each benchmark perturbs, in display order.
MISMATCH_PARAMS = {
    "cstr": ("alpha", "beta"),
    "cts": ("mismatch_factor", "gain_mismatch"),
}
PARAM_LABELS = {
    "alpha": r"$\alpha$ (side-reaction activation)",
    "beta": r"$\beta$ (main-reaction rate)",
    "mismatch_factor": r"outflow factor (valves)",
    "gain_mismatch": r"pump gain",
}
DIST_LABELS = {
    "CA0": r"feed concentration $C_{A0}$",
    "Tin": r"inlet temperature $T_{in}$",
    "leak2": r"tank-2 leak ($k_3$)",
}


def _mismatch_distance(frame, bench) -> np.ndarray:
    """Euclidean distance of each sampled plant from the nominal one.

    Both mismatch parameters are multiplicative factors on a nominal of 1.0, so
    they are already on a common scale and the plain norm is meaningful.
    """
    p0, p1 = MISMATCH_PARAMS[bench.key]
    offsets = np.column_stack(
        [frame[p0].to_numpy(float), frame[p1].to_numpy(float)]
    ) - 1.0
    return np.linalg.norm(offsets, axis=1)


def draw_fan(ax, ax_u, lhs, bench, ts) -> dict | None:
    """Draw the mismatch fan onto a supplied state axis and input axis."""
    tracked = bench.state_keys[bench.track_index]
    dist = _mismatch_distance(lhs, bench)
    norm = Normalize(vmin=float(dist.min()), vmax=float(dist.max()))
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
    ax.plot(t, med, color=ps.COLOR_PLANT, linewidth=1.8, zorder=4, label="median")
    ax.plot(t, first["sp"].to_numpy()[:n], color=ps.COLOR_SP, linestyle="--",
            linewidth=1.2, zorder=5, label="setpoint")

    ax.set_ylabel(f"{bench.state_labels[bench.track_index]} "
                  f"[{bench.state_units[bench.track_index]}]")
    ps.shade_inadmissible(ax, bench.x_lower[bench.track_index],
                          bench.x_upper[bench.track_index])
    ax_u.set_ylabel(f"{bench.action_labels[0]} [{bench.action_units[0]}]")
    ax_u.set_xlabel("time [s]")
    ps.finish(ax, legend=False)
    ps.finish(ax_u, legend=False)

    p0, p1 = MISMATCH_PARAMS[bench.key]
    return {
        "mappable": plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        "cbar_label": r"$\|(p_0, p_1) - (1, 1)\|$",
        "params": (p0, p1),
        "n_total": len(lhs),
        "n_completed": int(lhs["completed"].sum()),
    }


def _fan_caption_text(info) -> str:
    """The mismatch fan's caption, shared by the standalone and combined figures."""
    p0, p1 = info["params"]
    return (
        f"Color is how far the plant has drifted from the model "
        f"($p_0$ = {p0}, $p_1$ = {p1}). The controller holds the nominal model "
        f"throughout and receives no measurement of the drift. "
        f"N_total = {info['n_total']}, N_completed = {info['n_completed']}."
    )


def _fan(frame, bench, outdir, root) -> List[Path]:
    """Every mismatch trajectory at once, colored by distance from nominal."""
    lhs = frame[frame["case"].str.startswith("mismatch_lhs")]
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

    ax.set_title(f"{bench.label}: {len(lhs)} mismatched plants", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(info["mappable"], ax=[ax, ax_u], label=info["cbar_label"],
                 fraction=0.04, pad=0.02)

    ps.caption(fig, _fan_caption_text(info))
    return ps.save(fig, f"exp3_fan_{bench.key}", outdir)


def _disturbance(frame, bench, outdir, root) -> List[Path]:
    dist = frame[frame["dist_kind"] != "none"]
    if dist.empty or not _store.has_timeseries(EXPERIMENT, bench.key, root=root):
        return []

    written: List[Path] = []
    tracked = bench.state_keys[bench.track_index]

    for kind, group in dist.groupby("dist_kind"):
        mags = sorted(group["dist_magnitude"].unique())
        ts = _store.load_timeseries(EXPERIMENT, bench.key, group["run_id"], root=root)
        onset_step = int(group["dist_onset"].iloc[0])
        onset_t = onset_step * bench.sample_time_s

        fig, (ax, ax_u) = plt.subplots(
            2, 1, figsize=(6.0, 5.2), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        for i, mag in enumerate(mags):
            sel = group[group["dist_magnitude"] == mag]
            run = ts[ts.run_id == sel.iloc[0]["run_id"]].sort_values("step")
            if run.empty:
                continue
            style = ps.series_style(i, len(mags), ordered=True)
            style.pop("marker", None)
            ax.plot(run["time_s"], run[tracked], label=f"{mag:+.0%}",
                    linewidth=1.4, **{k: v for k, v in style.items()
                                      if k in ("color", "linestyle")})
            ax_u.plot(run["time_s"], run[bench.action_keys[0]], linewidth=1.0,
                      **{k: v for k, v in style.items()
                         if k in ("color", "linestyle")})
            t_end = run["time_s"].to_numpy()[-1]
            sp_end = run["sp"].to_numpy()[-1]
            y_end = run[tracked].to_numpy()[-1]
            if i == len(mags) - 1 and np.isfinite(y_end):
                # Bracket the residual offset of the largest disturbance.
                ax.annotate(
                    "", xy=(t_end * 0.98, sp_end), xytext=(t_end * 0.98, y_end),
                    arrowprops={"arrowstyle": "<->", "color": ps.COLOR_EST,
                                "lw": 1.2},
                )
                ax.text(t_end * 0.965, (sp_end + y_end) / 2,
                        f"offset {y_end - sp_end:+.3g}", fontsize=7.5,
                        color=ps.COLOR_EST, ha="right", va="center")

        run0 = ts[ts.run_id == group.iloc[0]["run_id"]].sort_values("step")
        ax.plot(run0["time_s"], run0["sp"], color=ps.COLOR_SP, linestyle="--",
                linewidth=1.0, label="setpoint", zorder=1)
        for a in (ax, ax_u):
            a.axvspan(onset_t, a.get_xlim()[1], color="0.9", alpha=0.5,
                      zorder=0, lw=0)
            a.axvline(onset_t, color=ps.COLOR_THRESHOLD, linestyle=":",
                      linewidth=1.2, zorder=1)
        ax.set_ylabel(f"{bench.state_labels[bench.track_index]} "
                      f"[{bench.state_units[bench.track_index]}]")
        ax.set_title(f"{bench.label}: unmeasured step in "
                     f"{DIST_LABELS.get(kind, kind)}", fontsize=10)
        ps.finish(ax, legend=True)
        ax_u.set_ylabel(f"{bench.action_labels[0]} [{bench.action_units[0]}]")
        ax_u.set_xlabel("time [s]")
        ps.finish(ax_u, legend=False)
        ps.caption(fig, "The disturbance is active over the shaded span. It is "
                        "unmeasured: no controller input carries it.")
        fig.tight_layout()
        written += ps.save(fig, f"exp3_disturbance_{kind}_{bench.key}", outdir)
    return written


def _offset_dots(frame, bench, outdir) -> List[Path]:
    cases = frame[frame["case"].str.startswith("mismatch_lhs")]
    if cases.empty:
        return []
    params = MISMATCH_PARAMS[bench.key]
    offsets = cases["ss_offset"].to_numpy()
    order = np.argsort(offsets)
    offsets = offsets[order]
    labels = [
        f"{cases.iloc[k][params[0]]:.3f} / {cases.iloc[k][params[1]]:.3f}"
        for k in order
    ]

    fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.16 * len(offsets))))
    lim = np.nanmax(np.abs(offsets)) if np.isfinite(offsets).any() else 1.0
    cmap = plt.get_cmap(ps.DIVERGING)
    colors = cmap((offsets / (2 * lim)) + 0.5)
    y = np.arange(len(offsets))
    ax.hlines(y, 0, offsets, color="0.8", linewidth=0.8, zorder=1)
    ax.scatter(offsets, y, c=colors, s=26, edgecolor="black", linewidth=0.4,
               zorder=3)
    ps.annotate_failures(ax, offsets, y,
                         ~cases.iloc[order]["completed"].to_numpy())
    ax.axvline(0.0, color=ps.COLOR_THRESHOLD, linewidth=1.4, zorder=2)
    ax.set_yticks(y[:: max(1, len(y) // 20)], labels[:: max(1, len(y) // 20)],
                  fontsize=6.5)
    ax.set_xlabel(f"steady-state offset [{bench.state_units[bench.track_index]}]")
    ax.set_ylabel(f"{params[0]} / {params[1]}", fontsize=8)
    ax.set_title(f"{bench.label}: residual offset under mismatch", fontsize=10)
    ps.finish(ax, legend=False)
    ps.caption(fig, "Above zero the controller settles above the setpoint, "
                    "below zero under it. One row per sampled mismatch case, "
                    "sorted by offset.")
    fig.tight_layout()
    return ps.save(fig, f"exp3_offset_dots_{bench.key}", outdir)


def _envelope_frame(frame) -> pd.DataFrame:
    """The wide mismatch sample, the design behind both parameter-space maps."""
    return frame[frame["case"].str.startswith("envelope")]


def _usable(frame, values: np.ndarray) -> np.ndarray:
    """Which sampled plants produced a number worth plotting.

    A failed solve or a non-finite metric means no result. The ``completed``
    flag is the wrong test here: it is also False for a run that solved every
    step but never settled, and that run carries a finite IAE and is the outcome
    the acceptability contour has to be drawn against. Interpolating over the
    settled runs alone would map the field only where the controller already
    works.
    """
    solved = frame["n_failed_solves"].to_numpy(float) == 0.0
    return solved & np.isfinite(values)


def _parameter_axes(ax, bench) -> None:
    """Axis labels, nominal marker and grid treatment shared by both maps."""
    p0, p1 = MISMATCH_PARAMS[bench.key]
    ax.scatter([1.0], [1.0], marker="*", s=140, color="white",
               edgecolor="black", linewidth=0.8, zorder=7, label="nominal plant")
    ax.set_xlabel(PARAM_LABELS.get(p0, p0))
    ax.set_ylabel(PARAM_LABELS.get(p1, p1))
    ax.grid(False)


def draw_envelope(ax, frame, bench, metric: str, **cbar_kw) -> dict | None:
    """Draw one mismatch-parameter map, with its scale, onto a supplied axis.

    The Latin-hypercube design is scattered, not gridded, so the field comes
    from a Delaunay triangulation of the sample.

    Runs whose solver failed, or whose metric is non-finite, are excluded from
    the triangulation and marked with a cross: a missing result bleeding into
    its neighbors' colors would misreport the runs around it. Runs that solved
    without settling are kept. See :func:`_usable`.
    """
    grid = _envelope_frame(frame)
    if grid.empty:
        return None
    p0, p1 = MISMATCH_PARAMS[bench.key]
    xs = grid[p0].to_numpy(float)
    ys = grid[p1].to_numpy(float)

    if metric == "iae":
        values = grid["iae_norm"].to_numpy(float)
        label = r"IAE$_{\mathrm{norm}}$ [–]"
        title = f"{bench.label}: mismatch envelope"
    elif metric == "offset":
        values = grid["ss_offset"].to_numpy(float)
        label = (f"steady-state offset "
                 f"[{bench.state_units[bench.track_index]}]")
        title = f"{bench.label}: residual offset under mismatch"
    else:
        raise ValueError(f"unknown metric {metric!r}")

    ok = _usable(grid, values)
    if int(ok.sum()) < 4:
        return None

    fig = ax.get_figure()
    x, y, z = xs[ok], ys[ok], values[ok]
    try:
        if metric == "iae":
            levels = np.linspace(float(np.min(z)),
                                 float(np.nanpercentile(z, 98)), 12)
            cf = ax.tricontourf(x, y, z, levels=levels, cmap=ps.SEQUENTIAL,
                                extend="max")
            fig.colorbar(cf, ax=ax, label=label, **cbar_kw)
            # The acceptability contour: twice the nominal cost.
            if float(np.min(z)) <= 2.0 <= float(np.max(z)):
                cs = ax.tricontour(x, y, z, levels=[2.0], colors="white",
                                   linewidths=1.6)
                ax.clabel(cs, fmt={2.0: "2x nominal"}, fontsize=7)
        else:
            # Symmetric about zero at the 98th percentile, so one diverged plant
            # cannot flatten the rest of the map to white.
            lim = float(np.nanpercentile(np.abs(z), 98)) or float(np.max(np.abs(z)))
            levels = np.linspace(-lim, lim, 13) if lim > 0 else None
            cf = ax.tricontourf(x, y, z, levels=levels, cmap=ps.DIVERGING,
                                extend="both")
            fig.colorbar(cf, ax=ax, label=label, **cbar_kw)
            if float(np.min(z)) <= 0.0 <= float(np.max(z)):
                ax.tricontour(x, y, z, levels=[0.0], colors="0.3",
                              linewidths=1.2)
    except Exception:  # noqa: BLE001 - a near-degenerate design has no triangulation
        return None

    ax.scatter(xs, ys, s=8, facecolor="none", edgecolor="black", linewidth=0.4,
               zorder=5, label="sampled plant")
    ps.annotate_failures(ax, xs, ys, ~ok, label="no result")
    _parameter_axes(ax, bench)

    return {
        "cbar_label": label,
        "title": title,
        "metric": metric,
        "params": (p0, p1),
        "ranges": (xs.min(), xs.max(), ys.min(), ys.max()),
        "n_used": int(ok.sum()),
        "n_total": len(grid),
        "n_failed": int((~ok).sum()),
        "n_unsettled": int((~grid["completed"].to_numpy(bool) & ok).sum()),
    }


def _envelope_caption_text(info) -> str:
    """One envelope map's caption, shared by the standalone and combined figures."""
    p0, p1 = info["params"]
    x_lo, x_hi, y_lo, y_hi = info["ranges"]
    extra = (
        "The map bounds how far the plant may drift before it needs retraining."
        if info["metric"] == "iae" else
        "Red settles above the setpoint, blue under it; the drawn contour is "
        "zero. The scale is clipped at the 98th percentile of |offset|."
    )
    notes = []
    if info["n_failed"]:
        notes.append(f"{info['n_failed']} crossed run(s) gave no usable result "
                     f"and are excluded")
    if info["n_unsettled"]:
        notes.append(f"{info['n_unsettled']} solved without settling and are kept")
    excluded = f" {'; '.join(notes)}." if notes else ""
    return (
        f"Interpolated from {info['n_used']} of {info['n_total']} sampled "
        f"plants, {p0} over {x_lo:.2f}–{x_hi:.2f} and {p1} over "
        f"{y_lo:.2f}–{y_hi:.2f}; dots are the sample.{excluded} {extra}"
    )


def _envelope_map(frame, bench, outdir, metric: str) -> List[Path]:
    """Filled map of one metric over the two mismatch parameters."""
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    info = draw_envelope(ax, frame, bench, metric)
    if info is None:
        plt.close(fig)
        return []

    ax.set_title(info["title"], fontsize=10)
    ax.legend(loc="best", fontsize=7.5)
    ps.caption(fig, _envelope_caption_text(info))
    fig.tight_layout()
    return ps.save(fig, f"exp3_envelope_{metric}_{bench.key}", outdir)


def _neural_vs_nmpc(frame, bench, outdir) -> List[Path]:
    cases = frame[frame["case"] == "neural_vs_nmpc"]
    if cases.empty or cases["controller"].nunique() < 2:
        return []
    p0, p1 = MISMATCH_PARAMS[bench.key]
    pivot = cases.pivot_table(index=[p0, p1], columns="controller",
                              values="iae_norm", aggfunc="median")
    pivot = pivot.dropna()
    if pivot.empty or "neural" not in pivot or "nmpc" not in pivot:
        return []

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    y = np.arange(len(pivot))
    ax.hlines(y, pivot["nmpc"], pivot["neural"], color="0.75", linewidth=1.0,
              zorder=1)
    ax.scatter(pivot["nmpc"], y, s=34, color=ps.CATEGORICAL[1], marker="s",
               edgecolor="black", linewidth=0.5, zorder=3, label="physics NMPC")
    ax.scatter(pivot["neural"], y, s=34, color=ps.CATEGORICAL[2], marker="o",
               edgecolor="black", linewidth=0.5, zorder=3, label="neural MPC")
    ax.set_yticks(y, [f"{a:g} / {b:g}" for a, b in pivot.index], fontsize=7)
    ax.set_ylabel(f"{p0} / {p1}", fontsize=8)
    ax.set_xlabel(r"IAE$_{\mathrm{norm}}$ [–]")
    ax.set_title(f"{bench.label}: degradation under identical mismatch",
                 fontsize=10)
    ax.legend(loc="best", fontsize=7.5)
    ps.finish(ax, legend=False)
    worse = int((pivot["neural"] > pivot["nmpc"]).sum())
    ps.caption(
        fig,
        f"Paired by mismatch case: both controllers hold a nominal model while "
        f"the plant has drifted, so both carry the same model error. The neural "
        f"controller is worse in {worse} of {len(pivot)} cases. The two "
        f"formulations differ in one further respect, which predates this "
        f"suite: the physics NMPC bounds {bench.action_labels[0]} below at "
        f"5 {bench.action_units[0]} (0.05 of its normalized range), the neural "
        f"controller at 0.",
    )
    fig.tight_layout()
    return ps.save(fig, f"exp3_neural_vs_nmpc_{bench.key}", outdir)


def build(metrics: pd.DataFrame, outdir: Path,
          root: Path | None = None) -> List[Path]:
    """Build every Experiment 3 figure that the store supports."""
    data = metrics[metrics["experiment"] == EXPERIMENT]
    if data.empty:
        return []
    written: List[Path] = []
    for bench_key, frame in data.groupby("benchmark"):
        bench = _config.BENCHMARKS[bench_key]
        written += _fan(frame, bench, outdir, root)
        written += _disturbance(frame, bench, outdir, root)
        written += _offset_dots(frame, bench, outdir)
        written += _envelope_map(frame, bench, outdir, "iae")
        written += _envelope_map(frame, bench, outdir, "offset")
        written += _neural_vs_nmpc(frame, bench, outdir)
    return written
