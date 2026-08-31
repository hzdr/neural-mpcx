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

"""Publication style, shared by every figure in the suite.

The ``rcParams`` block is the style cell of
``examples/Analysis/results_analysis.ipynb``, so the figures here match the ones
already in the paper.

Every palette below is color-vision-deficiency safe and survives grayscale
printing, which the notebooks achieve by encoding each series with a marker and
a linestyle on top of its color.

``CATEGORICAL``
    Okabe-Ito, for unordered series.
``ordinal(n)``
    A single-hue light-to-dark ramp for ordered factors such as noise level and
    disturbance magnitude. A categorical palette would discard the ordering.
``SEQUENTIAL`` (``cividis``)
    Perceptually uniform, for heatmaps and filled contours. A rainbow map would
    invent boundaries the data does not have.
``DIVERGING`` (``RdBu_r``)
    For signed quantities such as steady-state offset, centered on zero with the
    zero line drawn.
``trace_cmap()``
    The ordinal ramp truncated for trajectory fans, where a continuous value
    colors one thin line per run. ``SEQUENTIAL`` fails there: its pale end
    disappears at a fan's linewidth.
``COLOR_FAIL``
    Failed runs get ``#D55E00`` and an X marker, so color alone never carries
    the distinction. See :func:`annotate_failures`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator


#: The analysis notebooks' style cell.
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.top": True,
    "ytick.right": True,
}

#: Okabe-Ito, the palette the notebooks use.
CATEGORICAL: List[str] = [
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#0072B2",
]
MARKERS: List[str] = ["o", "s", "^", "D", "v", "p", "X"]
LINESTYLES: List = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]

#: Role colors, matching the example scripts' trajectory plots.
COLOR_PLANT = "#000000"
COLOR_PRED = "#0072B2"
COLOR_SP = "#009E73"
COLOR_EST = "#D55E00"
COLOR_MEAS = "#56B4E9"
COLOR_THRESHOLD = "#666666"
COLOR_FAIL = "#D55E00"

SEQUENTIAL = "cividis"
DIVERGING = "RdBu_r"

#: Light-to-dark blue ramp for ordered factors. It starts above white to keep
#: the lightest series legible on a white page.
ORDINAL_CMAP = LinearSegmentedColormap.from_list(
    "nmpcx_ordinal", ["#BFD3E6", "#6BAED6", "#2171B5", "#08306B"]
)
_ORDINAL_CMAP = ORDINAL_CMAP

#: The slice of ``ORDINAL_CMAP`` that :func:`trace_cmap` maps a fan onto. Fan
#: traces are thin and semi-transparent, so the light end of the ramp washes out
#: against the page. Starting at 0.25 keeps the lightest trajectory a solid
#: mid-blue.
ORDINAL_TRACE_RANGE = (0.25, 1.0)

MARKER_KW = {"markerfacecolor": "white", "markeredgewidth": 1.2,
             "markersize": 7}


def apply() -> None:
    """Install the publication rcParams. Idempotent; call once per session."""
    mpl.rcParams.update(RCPARAMS)


def ordinal(n: int) -> List:
    """``n`` colors on a light-to-dark ramp, for an ordered factor."""
    if n <= 1:
        return [_ORDINAL_CMAP(0.85)]
    return [_ORDINAL_CMAP(v) for v in np.linspace(0.12, 1.0, n)]


def trace_cmap() -> LinearSegmentedColormap:
    """``ORDINAL_CMAP`` truncated to :data:`ORDINAL_TRACE_RANGE`.

    Draw the fan lines and their colorbar through this, so the bar covers the
    range the traces span.
    """
    lo, hi = ORDINAL_TRACE_RANGE
    return LinearSegmentedColormap.from_list(
        "nmpcx_trace", ORDINAL_CMAP(np.linspace(lo, hi, 256))
    )


def series_style(index: int, n_total: int = 0, ordered: bool = False) -> dict:
    """Color, marker and linestyle for series ``index``.

    All three encode the series, so the figure reads in grayscale and for a
    color-blind reader.
    """
    if ordered and n_total:
        color = ordinal(n_total)[index]
    else:
        color = CATEGORICAL[index % len(CATEGORICAL)]
    return {
        "color": color,
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        **MARKER_KW,
    }


def finish(ax, minor: bool = True, legend: bool = True, **legend_kw) -> None:
    """Apply the standard grid, minor ticks and legend treatment to an axis."""
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    if minor:
        if ax.get_xscale() == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if ax.get_yscale() == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator())
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(**{"loc": "best", "fontsize": 8, **legend_kw})


def log2_axis(ax, values: Sequence[int], which: str = "x") -> None:
    """Log-2 axis with explicit integer ticks, as the notebooks use for hidden
    size and horizon."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log", base=2)
    (ax.set_xticks if which == "x" else ax.set_yticks)(list(values))
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)(
        [str(v) for v in values]
    )
    axis.set_minor_formatter(mpl.ticker.NullFormatter())


def log_axis_plain(ax, which: str = "x") -> None:
    """Log scale labeled with ordinary numbers, not powers of ten.

    A ratio like ``IAE_norm`` sits inside one decade, where the default locator
    fails at both ends: left alone it labels the minor ticks and collides them
    (``8.75 x 10^-1``), and silencing the minors leaves a 0.9-2.5 axis with a
    lone "1.00" between its limits. Majors go on sparse sub-decade multiples,
    printed plainly, and only the unlabeled minors are silenced.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_locator(mpl.ticker.LogLocator(
        base=10.0, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0), numticks=12))
    # %g per tick. ScalarFormatter picks one precision for the whole axis, so
    # over more than a decade it rounds 1.5 and 2.0 to the same "2" and the axis
    # grows a duplicate label.
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(mpl.ticker.LogLocator(
        base=10.0, subs="all", numticks=100))
    axis.set_minor_formatter(mpl.ticker.NullFormatter())


def panel_label(ax, letter: str) -> None:
    """Letter one panel of a combined figure, inside its top-left corner.

    The box sits over the data, so it carries an opaque background: a fan puts
    a hundred traces under this corner.
    """
    ax.text(0.015, 0.975, f"({letter})", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75,
                  "pad": 1.6})


def save(fig, name: str, outdir: Path, close: bool = True) -> List[Path]:
    """Write a figure as both PDF and SVG, the notebooks' convention."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "svg"):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path, format=ext)
        written.append(path)
    if close:
        plt.close(fig)
    return written


def shade_inadmissible(ax, lower: float | None, upper: float | None,
                       color: str = "0.85", alpha: float = 0.45) -> None:
    """Shade the region outside a constraint and draw the bound.

    The shading marks which side of the line violates the constraint.
    """
    x0, x1 = ax.get_xlim()
    if lower is not None:
        ax.axhspan(ax.get_ylim()[0], lower, color=color, alpha=alpha, zorder=0, lw=0)
        ax.axhline(lower, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    if upper is not None:
        ax.axhspan(upper, ax.get_ylim()[1], color=color, alpha=alpha, zorder=0, lw=0)
        ax.axhline(upper, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlim(x0, x1)


def annotate_failures(ax, x, y, mask: Iterable[bool], label: str = "failed") -> None:
    """Overlay failed runs with a distinct color and a distinct marker."""
    mask = np.asarray(list(mask), dtype=bool)
    if not mask.any():
        return
    ax.scatter(
        np.asarray(x)[mask], np.asarray(y)[mask],
        marker="X", s=42, facecolor=COLOR_FAIL, edgecolor="black",
        linewidth=0.6, zorder=6, label=label,
    )


#: Average glyph advance of the caption face, in inches per character at
#: :data:`CAPTION_FONTSIZE`. Used to wrap a caption to the figure's own width.
CAPTION_FONTSIZE = 7
_CAPTION_CHAR_IN = 0.48 * CAPTION_FONTSIZE / 72.0


def caption(fig, text: str, y: float = -0.02) -> None:
    """Attach a small footnote carrying absolute values and N counts.

    The text is wrapped to the figure's width. Left unwrapped it sets as one
    long line, and ``savefig.bbox = "tight"`` then widens the saved bounding box
    to fit it, so a figure with a long caption is saved several times wider than
    its axes and shrinks when a document scales it to a column.

    Lower ``y`` clears an artist already sitting under the axes, such as a
    figure-level legend.
    """
    width_in = fig.get_size_inches()[0]
    columns = max(40, int(width_in / _CAPTION_CHAR_IN))
    wrapped = textwrap.fill(text, width=columns)
    fig.text(0.005, y, wrapped, fontsize=CAPTION_FONTSIZE, va="top",
             ha="left", color="0.25")


__all__ = [
    "apply", "RCPARAMS", "CATEGORICAL", "MARKERS", "LINESTYLES", "MARKER_KW",
    "ordinal", "trace_cmap", "series_style", "finish", "log2_axis",
    "log_axis_plain", "panel_label", "save",
    "shade_inadmissible", "annotate_failures", "caption", "CAPTION_FONTSIZE",
    "SEQUENTIAL", "DIVERGING", "ORDINAL_CMAP", "ORDINAL_TRACE_RANGE",
    "COLOR_PLANT", "COLOR_PRED", "COLOR_SP",
    "COLOR_EST", "COLOR_MEAS", "COLOR_THRESHOLD", "COLOR_FAIL",
]
