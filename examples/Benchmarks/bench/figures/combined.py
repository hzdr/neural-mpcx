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

"""Multi-experiment figures, for the page budget the single-panel figures burn.

``combined_fan_{cstr,cts}``
    One benchmark's four closed-loop views in a lettered 2x2: the noise sweep,
    the initial-condition sample, the mismatch sample and the nominal run. Each
    panel keeps its own actuator axis underneath, so the figure carries eight
    axes.
``combined_envelope``
    Both benchmarks' mismatch maps in one lettered 2x2: the two-tank on top, the
    CSTR below, tracking cost on the left and signed offset on the right.

Every panel is painted by the ``draw_*`` function of the experiment that owns
it, so a combined figure and its standalone counterpart cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .. import config as _config
from .. import plotstyle as ps
from .. import store as _store
from . import exp1, exp2, exp3, exp5

import matplotlib.pyplot as plt

LETTERS = ("a", "b", "c", "d")

#: Each fan panel's heading. They name what varies, not the experiment id.
FAN_TITLES = (
    "measurement noise",
    "initial condition",
    "plant mismatch",
    "nominal reference",
)


def _cell_axes(fig, gridspec, row, col):
    """A state axis over an actuator axis, sharing one time axis."""
    inner = gridspec[row, col].subgridspec(2, 1, height_ratios=(2.0, 1.0),
                                           hspace=0.06)
    ax = fig.add_subplot(inner[0])
    ax_u = fig.add_subplot(inner[1], sharex=ax)
    ax.tick_params(labelbottom=False)
    return ax, ax_u


def _attach_colorbar(fig, info, ax, ax_u) -> None:
    """Hang one panel's scale beside its own pair of axes."""
    if info is None or info.get("mappable") is None:
        return
    bar = fig.colorbar(info["mappable"], ax=[ax, ax_u], label=info["cbar_label"],
                       fraction=0.05, pad=0.02)
    if "cbar_ticks" in info:
        bar.set_ticks(info["cbar_ticks"], labels=info["cbar_ticklabels"])
    bar.ax.tick_params(labelsize=8)
    bar.set_label(info["cbar_label"], fontsize=9)


#: Where the shared key and the caption sit, in figure fractions below the axes.
LEGEND_Y = -0.028
CAPTION_Y = -0.062


def _shared_legend(fig, axes) -> None:
    """One key for the whole figure, from the roles the panels share.

    A box per panel would repeat the same three entries and sit on top of the
    traces it describes.

    The anchor is an explicit negative offset rather than ``outside lower
    center``: the layout engine reserves one band for outside artists, and the
    suptitle already holds it, so an outside legend lands on the title.
    """
    seen, handles, labels = set(), [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, LEGEND_Y), ncol=len(handles),
                   fontsize=9, frameon=False)


def _fan_grid(metrics, bench, outdir, root) -> List[Path]:
    """One benchmark's four closed-loop views, lettered (a) to (d)."""
    panels = []
    for experiment, case_filter, drawer in (
        ("exp1", None, exp1.draw_fan),
        ("exp2", "lhs50", exp2.draw_fan),
        ("exp3", "mismatch_lhs", exp3.draw_fan),
        ("exp5", None, None),
    ):
        frame = metrics[(metrics["experiment"] == experiment)
                        & (metrics["benchmark"] == bench.key)]
        if case_filter:
            frame = frame[frame["case"].str.startswith(case_filter)]
        if frame.empty or not _store.has_timeseries(experiment, bench.key,
                                                    root=root):
            return []
        panels.append((experiment, frame, drawer))

    fig = plt.figure(figsize=(11.0, 9.2), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)
    outer = fig.add_gridspec(2, 2)

    infos, state_axes = [], []
    for i, (experiment, frame, drawer) in enumerate(panels):
        ax, ax_u = _cell_axes(fig, outer, i // 2, i % 2)
        ts = _store.load_timeseries(experiment, bench.key, frame["run_id"],
                                    root=root)
        if drawer is None:
            row = frame[frame["controller"] == "neural"].iloc[0]
            info = exp5.draw_nominal(
                ax, ax_u, row, bench, ts[ts.run_id == row["run_id"]]
                .sort_values("step"))
        else:
            info = drawer(ax, ax_u, frame, bench, ts)
        if info is None:
            plt.close(fig)
            return []
        ax.set_title(FAN_TITLES[i], fontsize=10)
        ps.panel_label(ax, LETTERS[i])
        _attach_colorbar(fig, info, ax, ax_u)
        infos.append(info)
        state_axes.append(ax)

    # The title goes on before the legend: constrained layout stacks outside
    # artists in the order they are added, and a legend added first lands under
    # the title instead of below it.
    fig.suptitle(f"{bench.label}: closed loop across the four experiments",
                 fontsize=12, fontweight="bold")
    # The three fans share one key. Panel (d) is a single run with two traces of
    # its own, so it keeps a small box rather than adding them to the shared one.
    _shared_legend(fig, state_axes[:3])
    state_axes[3].legend(loc="lower right", fontsize=8)

    counts = ", ".join(
        f"({LETTERS[i]}) {info['n_completed']} of {info['n_total']}"
        for i, info in enumerate(infos[:3])
    )
    ps.caption(
        fig,
        f"Median and 5–95 % band over every run of each panel, with the colored "
        f"traces the runs themselves; (a) bands and medians are per noise level "
        f"and the black line spans all six. Shading marks the inadmissible side "
        f"of each constraint. Runs completed: {counts}. (d) is the nominal run, "
        f"absolute IAE = {infos[3]['iae']:.4g} {infos[3]['unit']}·s over "
        f"{infos[3]['n_steps']} steps, the reference every normalized IAE in the "
        f"study divides by.",
        y=CAPTION_Y,
    )
    return ps.save(fig, f"combined_fan_{bench.key}", outdir)


def _envelope_grid(metrics, outdir) -> List[Path]:
    """Both benchmarks' mismatch maps, two-tank on top, lettered (a) to (d)."""
    data = metrics[metrics["experiment"] == "exp3"]
    if data.empty:
        return []
    # The two-tank leads, so the row order matches the benchmark order the text
    # introduces the maps in.
    order = [k for k in ("cts", "cstr") if k in set(data["benchmark"])]
    if len(order) < 2:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.05)

    infos = []
    for r, bench_key in enumerate(order):
        bench = _config.BENCHMARKS[bench_key]
        frame = data[data["benchmark"] == bench_key]
        for c, metric in enumerate(("iae", "offset")):
            ax = axes[r][c]
            info = exp3.draw_envelope(ax, frame, bench, metric,
                                      fraction=0.05, pad=0.02)
            if info is None:
                plt.close(fig)
                return []
            ax.set_title(info["title"], fontsize=10)
            ps.panel_label(ax, LETTERS[2 * r + c])
            infos.append(info)

    axes[0][0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Mismatch envelope over the plant parameters",
                 fontsize=12, fontweight="bold")

    # Both columns share a method and both rows share a sample, so the caption
    # states each once and then only what separates the rows.
    rows = []
    for r, bench_key in enumerate(order):
        info = infos[2 * r]
        p0, p1 = info["params"]
        x_lo, x_hi, y_lo, y_hi = info["ranges"]
        rows.append(
            f"({LETTERS[2 * r]}, {LETTERS[2 * r + 1]}) "
            f"{_config.BENCHMARKS[bench_key].label}, {p0} over "
            f"{x_lo:.2f}–{x_hi:.2f} and {p1} over {y_lo:.2f}–{y_hi:.2f}, "
            f"{info['n_used']} of {info['n_total']} plants usable"
        )
    ps.caption(
        fig,
        f"Delaunay-interpolated from each benchmark's mismatch sample; dots are "
        f"the sample and the star is the nominal plant. Left column: tracking "
        f"cost, the white contour at twice nominal bounding how far the plant "
        f"may drift before it needs retraining. Right column: signed "
        f"steady-state offset, red above the setpoint and blue below, clipped at "
        f"the 98th percentile of |offset|. {'. '.join(rows)}.",
    )
    return ps.save(fig, "combined_envelope", outdir)


def build(metrics: pd.DataFrame, outdir: Path,
          root: Path | None = None) -> List[Path]:
    """Build every combined figure the store supports."""
    written: List[Path] = []
    for bench_key in sorted(set(metrics["benchmark"])):
        written += _fan_grid(metrics, _config.BENCHMARKS[bench_key], outdir, root)
    written += _envelope_grid(metrics, outdir)
    return written
