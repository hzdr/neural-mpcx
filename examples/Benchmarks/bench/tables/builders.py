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

"""Every table in the study.

One rule governs all of them: no aggregate excludes a failed run. Every table
reports ``N_total`` beside ``N_completed``, and every median covers all runs
with non-finite values ranked worst.

Each builder writes a CSV and a LaTeX fragment, and returns the paths it wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import config as _config
from .. import metrics as _metrics
from .. import store as _store


def _write(frame: pd.DataFrame, name: str, outdir: Path,
           caption: str = "") -> List[Path]:
    """Persist one table as CSV plus a LaTeX fragment."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{name}.csv"
    tex_path = outdir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    tex = frame.to_latex(
        index=False, escape=True, float_format=lambda v: f"{v:.4g}",
        caption=caption or name, label=f"tab:{name}",
    )
    tex_path.write_text(tex)
    return [csv_path, tex_path]


def _nanmedian(values) -> float:
    """Median ignoring NaN, returning NaN when every value is NaN.

    A case without a disturbance has no recovery time, which is a genuine "not
    applicable" and should not raise a warning.
    """
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")


def _summary_row(group: pd.DataFrame, column: str = "iae_norm") -> Dict[str, float]:
    """The standard robustness summary of one group of runs."""
    stats = _metrics.median_iqr(group[column].to_numpy())
    values = group[column].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    n_total = len(group)
    n_completed = int(group["completed"].sum())
    return {
        "N_total": n_total,
        "N_completed": n_completed,
        "success_rate": n_completed / n_total if n_total else np.nan,
        "IAE_norm_median": stats["median"],
        "IAE_norm_q1": stats["q1"],
        "IAE_norm_q3": stats["q3"],
        "IAE_norm_worst": float(finite.max()) if finite.size else np.nan,
        "worst_violation": float(np.nanmax(group["worst_violation"]))
        if len(group) else np.nan,
        "violation_rate": float(np.nanmean(group["violation_rate"]))
        if len(group) else np.nan,
        "failed_solves": int(np.nansum(group["n_failed_solves"].clip(lower=0))),
        "RTF_max": float(np.nanmax(group["rtf_wcet"])) if len(group) else np.nan,
    }


# --------------------------------------------------------------------------
# Master table
# --------------------------------------------------------------------------

#: What each experiment varies, for T0's "what varied" column. exp2's box
#: differs per benchmark, so it names both.
VARIED = {
    "exp1": "measurement noise sigma",
    "exp2": "initial condition (+-15 % CSTR, +-50 % two-tank)",
    "exp3": "plant mismatch and step disturbances",
    "exp4": "hidden size x horizon",
    "exp5": "nothing (nominal reference)",
}


def robustness_at_a_glance(metrics: pd.DataFrame, outdir: Path) -> List[Path]:
    """Every experiment and benchmark, one row each: the summary table.

    Columns: what varied, N_total / N_completed, success rate, median IAE with
    its interquartile range, worst constraint violation, failed solves, and the
    worst real-time factor observed.
    """
    rows = []
    for (exp, bench_key), group in metrics.groupby(["experiment", "benchmark"]):
        summary = _summary_row(group)
        rows.append({
            "experiment": exp,
            "benchmark": _config.BENCHMARKS[bench_key].label,
            "what varied": VARIED.get(exp, ""),
            **summary,
        })
    frame = pd.DataFrame(rows).sort_values(["experiment", "benchmark"])
    frame["IAE_norm [IQR]"] = [
        f"{m:.3g} [{a:.3g}, {b:.3g}]"
        for m, a, b in zip(frame["IAE_norm_median"], frame["IAE_norm_q1"],
                           frame["IAE_norm_q3"])
    ]
    display = frame[[
        "experiment", "benchmark", "what varied", "N_total", "N_completed",
        "success_rate", "IAE_norm [IQR]", "worst_violation", "failed_solves",
        "RTF_max",
    ]]
    return _write(
        display, "T0_robustness_at_a_glance", outdir,
        caption=(
            "Robustness at a glance. Success = no failed solve and the tracked "
            "variable inside the +-5 % setpoint band at the end of the run. "
            "Violations are expressed as a fraction of each variable's operating "
            "range. All aggregates include failed runs."
        ),
    )


# --------------------------------------------------------------------------
# Per-experiment tables
# --------------------------------------------------------------------------

def exp2_summary(metrics: pd.DataFrame, outdir: Path) -> List[Path]:
    """Initial-condition robustness, one row per benchmark."""
    data = metrics[metrics["experiment"] == "exp2"]
    if data.empty:
        return []
    rows = []
    for bench_key, group in data.groupby("benchmark"):
        rows.append({
            "benchmark": _config.BENCHMARKS[bench_key].label,
            **_summary_row(group),
        })
    return _write(
        pd.DataFrame(rows), "T2_initial_condition_summary", outdir,
        caption=(
            "Robustness to the initial condition, over the 50-point Latin "
            "hypercube of each benchmark. The success rate is the empirical "
            "stand-in for the recursive-feasibility guarantee a neural MPC "
            "cannot offer. Every run is counted, including failures."
        ),
    )


def exp2_worst(metrics: pd.DataFrame, outdir: Path, n: int = 5) -> List[Path]:
    """The hardest initial conditions, listed explicitly.

    A median hides which starts were hard, so the worst runs are listed with
    the initial states that produced them.
    """
    data = metrics[metrics["experiment"] == "exp2"]
    if data.empty:
        return []
    rows = []
    for bench_key, frame in data.groupby("benchmark"):
        bench = _config.BENCHMARKS[bench_key]
        ranked = frame.assign(
            _rank=frame["iae_norm"].fillna(np.inf)
        ).sort_values("_rank", ascending=False).head(n)
        for _, r in ranked.iterrows():
            entry = {
                "benchmark": bench.label,
                "design": r["case"],
                "IAE_norm": r["iae_norm"],
                "completed": bool(r["completed"]),
                "failed solves": int(r["n_failed_solves"]),
                "worst violation": r["worst_violation"],
                "settling [s]": r["settling_time_s"],
            }
            x0 = np.asarray(r["x0"], dtype=float) if r["x0"] is not None else None
            for i, key in enumerate(bench.state_keys):
                entry[f"x0 {key}"] = x0[i] if x0 is not None else np.nan
            rows.append(entry)
    return _write(
        pd.DataFrame(rows), "T2_worst_initial_conditions", outdir,
        caption=f"The {n} worst runs per benchmark, with their initial states.",
    )


def exp3_cases(metrics: pd.DataFrame, outdir: Path) -> List[Path]:
    """Mismatch and disturbance cases, summarized per case."""
    data = metrics[metrics["experiment"] == "exp3"]
    if data.empty:
        return []
    rows = []
    for (bench_key, case), group in data.groupby(["benchmark", "case"]):
        summary = _summary_row(group)
        rows.append({
            "benchmark": _config.BENCHMARKS[bench_key].label,
            "case": case,
            "median offset": float(np.nanmedian(group["ss_offset"])),
            "median recovery [steps]": _nanmedian(group["recovery_steps"]),
            **summary,
        })
    return _write(
        pd.DataFrame(rows).sort_values(["benchmark", "case"]),
        "T3_mismatch_and_disturbance_cases", outdir,
        caption=(
            "Mismatch and disturbance cases. Recovery time is measured from the "
            "disturbance onset until the tracked variable re-enters the +-5 % "
            "band and stays; it is infinite when the run ends outside the band."
        ),
    )


def _acceptable_interval(
    grid: np.ndarray, values: np.ndarray, threshold: float
) -> tuple[float, float] | None:
    """Widest run of ``grid`` around nominal whose ``values`` stay acceptable.

    Walks outward from the sample nearest 1.0 and stops at the first point over
    ``threshold`` or at the first NaN. NaN means outside the sample's convex
    hull, so stopping there holds the reported band inside the region the study
    covered.
    """
    centre = int(np.argmin(np.abs(grid - 1.0)))
    if not np.isfinite(values[centre]) or values[centre] > threshold:
        return None
    lo = hi = centre
    while lo > 0 and np.isfinite(values[lo - 1]) and values[lo - 1] <= threshold:
        lo -= 1
    while (hi < grid.size - 1 and np.isfinite(values[hi + 1])
           and values[hi + 1] <= threshold):
        hi += 1
    return float(grid[lo]), float(grid[hi])


def exp3_tolerance(metrics: pd.DataFrame, outdir: Path,
                   threshold: float = 2.0) -> List[Path]:
    """How far each plant parameter may drift.

    For each parameter, the largest perturbation in each direction whose
    normalized IAE stays under ``threshold``, along the slice with the other
    parameter at nominal.

    A Latin hypercube places no samples on that slice, so
    :func:`scipy.interpolate.griddata` interpolates it from every run with a
    finite IAE. It returns NaN outside the convex hull, which this treats as
    unacceptable, holding the band inside the sampled region.
    """
    from scipy.interpolate import griddata

    from ..figures.exp3 import MISMATCH_PARAMS

    data = metrics[metrics["experiment"] == "exp3"]
    data = data[data["case"].str.startswith("envelope")]
    if data.empty:
        return []

    rows = []
    for bench_key, frame in data.groupby("benchmark"):
        bench = _config.BENCHMARKS[bench_key]
        params = MISMATCH_PARAMS[bench_key]
        # The test is a solved run with a finite IAE, not ``completed``: a
        # plant that solved every step without settling carries a meaningful
        # IAE and is the case the threshold has to be crossed against.
        # Interpolating over the acceptable runs alone would bound the reported
        # band by where the data stops instead of by the threshold.
        usable = frame[(frame["n_failed_solves"] == 0)
                       & np.isfinite(frame["iae_norm"])]
        points = usable[list(params)].to_numpy(float)
        iae = usable["iae_norm"].to_numpy(float)

        for param in params:
            span = frame[param].to_numpy(float)
            axis = np.linspace(float(span.min()), float(span.max()), 401)
            interval = None
            if len(points) >= 4:
                query = np.empty((axis.size, len(params)), dtype=float)
                for k, name in enumerate(params):
                    query[:, k] = axis if name == param else 1.0
                sliced = griddata(points, iae, query, method="linear")
                interval = _acceptable_interval(axis, sliced, threshold)

            if interval is None:
                rows.append({"benchmark": bench.label, "parameter": param,
                             "lowest acceptable": np.nan,
                             "highest acceptable": np.nan,
                             "tolerance band": "none"})
                continue
            lo, hi = interval
            rows.append({
                "benchmark": bench.label,
                "parameter": param,
                "lowest acceptable": lo,
                "highest acceptable": hi,
                "tolerance band": f"{(lo - 1) * 100:+.0f}% to {(hi - 1) * 100:+.0f}%",
            })
    return _write(
        pd.DataFrame(rows), "T3_tolerance_summary", outdir,
        caption=(
            f"Largest still-acceptable drift per parameter, along the slice with "
            f"the other parameter at nominal, linearly interpolated from the "
            f"Latin-hypercube mismatch sample. Acceptable means "
            f"IAE_norm <= {threshold:g}. Runs that solved without settling carry "
            f"their IAE into the interpolation, and the band stops at the edge "
            f"of the sampled region without extrapolating past it."
        ),
    )


def exp4_realtime(metrics: pd.DataFrame, outdir: Path) -> List[Path]:
    """Solve-time and real-time-feasibility table, the full distribution."""
    data = metrics[metrics["experiment"] == "exp4"]
    if data.empty:
        return []
    rows = []
    for _, r in data.sort_values(
        ["benchmark", "hidden_size", "horizon"]
    ).iterrows():
        bench = _config.BENCHMARKS[r["benchmark"]]
        rows.append({
            "benchmark": bench.label,
            "controller": "physics NMPC" if r["controller"] == "nmpc" else "neural",
            "hidden size": "-" if r["hidden_size"] == -1 else int(r["hidden_size"]),
            "horizon": int(r["horizon"]),
            "period [s]": bench.sample_time_s,
            "mean [ms]": r["solve_ms_mean"],
            "P50 [ms]": r["solve_ms_p50"],
            "P95 [ms]": r["solve_ms_p95"],
            "P99 [ms]": r["solve_ms_p99"],
            "WCET [ms]": r["solve_ms_max"],
            "RTF_P99": r["rtf_p99"],
            "RTF_WCET": r["rtf_wcet"],
            "real-time": "yes" if r["rtf_wcet"] < 1.0 else "no",
            "IAE_norm": r["iae_norm"],
            "failed solves": int(r["n_failed_solves"]),
        })
    return _write(
        pd.DataFrame(rows), "T4_real_time_feasibility", outdir,
        caption=(
            "Solve-time distribution and real-time factors, measured serially on "
            "an otherwise idle machine so the numbers reflect the controller "
            "alone and not contention between parallel workers."
        ),
    )


def solver_failures(metrics: pd.DataFrame, outdir: Path,
                    root: Path | None = None) -> List[Path]:
    """Where the solver failed, and whether it failed briefly or persistently.

    A run that lost two steps out of a thousand and a run infeasible from step
    nine both read as "failed" in a success rate, though the first is a
    transient the warm start recovered from and the second is a controller that
    stopped working. This table separates them per case.

    Columns
    -------
    ``runs`` / ``runs_affected``
        How many runs of the case had at least one failed solve.
    ``failed_steps`` / ``%_of_steps``
        Total failed solves, and their share of every solve attempted.
    ``severity p50`` / ``p90`` / ``worst``
        Percentiles of the per-run fraction of steps that failed. Near 0 means
        brief transients; near 1 means infeasible for most of the run.
    ``persistent``
        Runs that lost more than half their steps.
    ``first failure``
        Median step of the first failure, from the stored time series. Read
        with the severity, it separates a loop that degraded from one that
        never got going.
    """
    rows = []
    for (exp, bench_key, case), group in metrics.groupby(
        ["experiment", "benchmark", "case"]
    ):
        failed = group["n_failed_solves"].clip(lower=0)
        affected = group[failed > 0]
        if affected.empty:
            continue
        frac = (affected["n_failed_solves"].clip(lower=0)
                / affected["n_solves"].replace(0, np.nan)).to_numpy(dtype=float)
        first = np.nan
        try:
            ts = _store.load_timeseries(exp, bench_key, affected["run_id"], root=root)
            bad = ts[~ts["solve_ok"]]
            if not bad.empty:
                first = float(bad.groupby("run_id")["step"].min().median())
        except FileNotFoundError:
            pass
        rows.append({
            "experiment": exp,
            "benchmark": _config.BENCHMARKS[bench_key].label,
            "case": case,
            "runs": len(group),
            "runs_affected": len(affected),
            "failed_steps": int(failed.sum()),
            "%_of_steps": 100.0 * failed.sum() / max(1, int(group["n_solves"].sum())),
            "severity p50": float(np.nanpercentile(frac, 50)),
            "severity p90": float(np.nanpercentile(frac, 90)),
            "severity worst": float(np.nanmax(frac)),
            "persistent (>50% of steps)": int((frac > 0.5).sum()),
            "first failure [step]": first,
            "median steps": float(group["n_solves"].median()),
        })

    if not rows:
        # State it explicitly: an absent table and a table reporting no
        # failures anywhere would otherwise look the same.
        rows = [{"experiment": "(none)", "benchmark": "-", "case": "-",
                 "runs": int(len(metrics)), "runs_affected": 0,
                 "failed_steps": 0, "%_of_steps": 0.0}]

    return _write(
        pd.DataFrame(rows).sort_values(
            ["experiment", "benchmark", "failed_steps"], ascending=[1, 1, 0]
        ),
        "T6_solver_failures", outdir,
        caption=(
            "Solver failures by case. Severity is the per-run fraction of steps "
            "whose solve failed: near zero means isolated transients, near one "
            "means the problem was infeasible for most of the run. Failed runs "
            "stay in every other aggregate in this study."
        ),
    )


def normalizers_table(outdir: Path, root: Path | None = None) -> List[Path]:
    """The absolute IAE behind every normalized number, for the captions."""
    normalizers = _store.load_normalizers(root)
    if not normalizers:
        return []
    rows = [
        {
            "benchmark": _config.BENCHMARKS[k].label,
            "tracked variable": v["tracked_variable"],
            "nominal IAE": v["iae_tracked"],
            "unit": v["unit"],
            "runs": v["n_runs"],
        }
        for k, v in normalizers.items()
    ]
    return _write(
        pd.DataFrame(rows), "T5_normalizers", outdir,
        caption=(
            "The nominal closed-loop IAE of each benchmark. Every IAE_norm in "
            "the study is divided by these, so 1.0 means as good as the nominal "
            "reported figure."
        ),
    )


def build_all(metrics: pd.DataFrame, outdir: Path,
              root: Path | None = None) -> List[Path]:
    """Build every table the store supports."""
    written: List[Path] = []
    written += robustness_at_a_glance(metrics, outdir)
    written += exp2_summary(metrics, outdir)
    written += exp2_worst(metrics, outdir)
    written += exp3_cases(metrics, outdir)
    written += exp3_tolerance(metrics, outdir)
    written += exp4_realtime(metrics, outdir)
    written += solver_failures(metrics, outdir, root)
    written += normalizers_table(outdir, root)
    return written
