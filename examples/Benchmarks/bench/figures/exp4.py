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

"""Experiment 4 figure: real-time feasibility.

``exp4_rtf_wcet`` reproduces the analysis-notebook figure: worst-case real-time
factor against horizon, one line per LSTM hidden size, log-2 horizon axis, log
RTF axis, threshold at 1.0. The CSTR panel carries the physics NMPC under the
``hidden_size = -1`` sentinel, the convention the notebook uses.

The JSON table flags these runs ``timing_critical``, so they execute serially
after the rest of the sweep and their latencies measure the controller alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .. import config as _config
from .. import plotstyle as ps

import matplotlib.pyplot as plt


EXPERIMENT = "exp4"

NMPC_SENTINEL = -1


def _rtf(frame, bench, outdir) -> List[Path]:
    """Worst-case solve time over the control period, per hidden size."""
    horizons = sorted(frame["horizon"].unique())
    sizes = sorted(frame["hidden_size"].unique())
    if not horizons or not sizes:
        return []

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.axhline(y=1, color=ps.COLOR_THRESHOLD, linestyle="--", linewidth=1.5,
               label="Threshold", zorder=1)

    for i, hs in enumerate(sizes):
        sub = frame[frame.hidden_size == hs].sort_values("horizon")
        if sub.empty:
            continue
        name = "NMPC" if hs == NMPC_SENTINEL else rf"$H_s$ = {hs}"
        style = ps.series_style(i)
        # Marker and dash carry the series here, as in the notebook, so the
        # panel survives grayscale printing.
        style["color"] = "#000000"
        ax.plot(sub["horizon"], sub["rtf_wcet"], label=name, zorder=2,
                **style)

    ax.set_xlabel(r"Horizon length, $N_p$ [–]")
    ax.set_ylabel(r"RTF$_{\mathrm{WCET}}$ [–]")
    ps.log2_axis(ax, horizons)
    ax.set_yscale("log")
    ax.legend(loc="upper left", ncol=2, title="Hidden size", fontsize=8,
              title_fontsize=8)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    ps.caption(
        fig,
        f"{bench.label}; control period {bench.sample_time_s:g} s. Measured with "
        f"one solve at a time on an otherwise idle machine. Below the threshold "
        f"the controller finishes within its period.",
    )
    fig.tight_layout()
    return ps.save(fig, f"exp4_rtf_wcet_{bench.key}", outdir)


def build(metrics: pd.DataFrame, outdir: Path,
          root: Path | None = None) -> List[Path]:
    """Build every Experiment 4 figure that the store supports."""
    data = metrics[metrics["experiment"] == EXPERIMENT]
    if data.empty:
        return []
    written: List[Path] = []
    for bench_key, frame in data.groupby("benchmark"):
        bench = _config.BENCHMARKS[bench_key]
        written += _rtf(frame, bench, outdir)
    return written
