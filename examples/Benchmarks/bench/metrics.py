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

"""Closed-loop performance metrics.

:func:`iae`, :func:`nrmse` and the real-time-feasibility ratios use the
definitions in ``examples/Analysis/results_analysis.ipynb``, so numbers here
compare directly with the figures already in the paper. :func:`settling_steps`
comes from ``mpc_hpo_cts.py``.

Every quantity is computed in physical units (mol/L, degC, m, V), whether or not
the benchmark stores normalized values, so both benchmarks share an axis once
the IAE is divided by its nominal run.

Every function returns a number for every run, failed solves included. Failure
is carried separately by ``n_failed_solves`` and ``completed``, and the table
helpers report ``N_total`` beside ``N_completed``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


#: ``np.trapz`` was removed in NumPy 2.0 and renamed ``np.trapezoid``. Bind the
#: one this install has, so the suite runs on either.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def iae(actual: np.ndarray, setpoint: np.ndarray, sample_time: float) -> float:
    """Integral of absolute error, by trapezoidal integration.

    Parameters
    ----------
    actual, setpoint : np.ndarray
        Equal-length 1-D series in physical units.
    sample_time : float
        Sampling interval [s]. The result therefore has units of
        (physical unit) x s.

    Returns
    -------
    float
        NaN if the series is empty or entirely NaN.
    """
    actual = np.asarray(actual, dtype=float)
    setpoint = np.asarray(setpoint, dtype=float)
    err = np.abs(actual - setpoint)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("nan")
    return float(_trapezoid(err, dx=sample_time))


def nrmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root-mean-square prediction error normalized by ``std(actual)``.

    The definition of ``calculate_nrmse`` in the analysis notebooks. A failed
    solve is recorded as a NaN prediction and excluded pairwise, so a failure
    shrinks the sample size and leaves the statistic intact.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() < 2:
        return float("nan")
    a, p = actual[mask], predicted[mask]
    denom = np.std(a)
    if denom == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - p) ** 2)) / denom)


def total_variation(u: np.ndarray) -> float:
    """Total variation of an input series, ``sum |u_{k+1} - u_k|``.

    The actuator-effort counterpart to the IAE. Noise reaching the LSTM's
    recurrent state appears here as input chatter before it reaches the tracking
    error.

    Parameters
    ----------
    u : np.ndarray
        Shape ``(nsteps,)`` or ``(nsteps, nu)``, in physical units.

    Returns
    -------
    float
        Summed over channels when ``u`` is 2-D.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    d = np.abs(np.diff(u, axis=0))
    if d.size == 0:
        return 0.0
    return float(np.nansum(d))


def _segments(timestamps: Sequence[int], n_steps: int):
    """Yield ``(index, start, end)`` for each active setpoint segment."""
    for i, start in enumerate(timestamps):
        if start >= n_steps:
            break
        end = timestamps[i + 1] if i + 1 < len(timestamps) else n_steps
        yield i, int(start), int(min(end, n_steps))


def settling_steps(
    tracked: np.ndarray,
    setpoints: Sequence[float],
    timestamps: Sequence[int],
    band_frac: float = 0.25,
) -> int:
    """Total settling steps summed over the setpoint segments.

    Per segment: the number of steps from the segment start until the tracked
    variable enters the ±``band_frac`` band around that segment's setpoint and
    *stays* there through the segment end. A segment that never settles
    contributes its full length.

    The definition of ``_compute_settling`` in ``mpc_hpo_cts.py``, so these
    numbers match the ones the controllers were tuned against.

    Parameters
    ----------
    tracked : np.ndarray
        The controlled variable over time, physical units, shape ``(nsteps,)``.
    setpoints : sequence of float
        Setpoint value of the tracked variable per segment, physical units.
    timestamps : sequence of int
        Step index at which each segment begins.
    band_frac : float, optional
        Half-width of the settling band as a fraction of the segment setpoint.

    Returns
    -------
    int
        Steps, not seconds. Multiply by the sample time for a duration.
    """
    tracked = np.asarray(tracked, dtype=float)
    n_steps = tracked.size
    total = 0
    for i, start, end in _segments(timestamps, n_steps):
        sp = float(setpoints[i])
        band = band_frac * abs(sp)
        seg = tracked[start:end]
        viol = ~(np.abs(seg - sp) <= band)  # NaN counts as a violation
        if viol.any():
            total += int(np.max(np.where(viol)[0])) + 1
    return int(total)


def reached_band(
    tracked: np.ndarray,
    setpoints: Sequence[float],
    timestamps: Sequence[int],
    band_frac: float = 0.5,
    all_segments: bool = False,
) -> bool:
    """Did the run end inside the ±``band_frac`` setpoint band?

    One half of the success criterion; the other half is "no failed solve".

    Parameters
    ----------
    all_segments : bool, optional
        False (the default) checks only the final segment. True requires every
        segment to end inside its own band.

    Notes
    -----
    The default checks the final segment because the two-tank nominal run fails
    the strict rule: it settles about 0.6 m below the h_2 = 2 m setpoint, which
    is a steady-state offset and not a divergence. Under the strict rule every
    two-tank success rate would read 0 % and would measure that one known offset
    instead of the factor under test. The store keeps both readings, so the
    offset stays visible.
    """
    tracked = np.asarray(tracked, dtype=float)
    n_steps = tracked.size
    segs = list(_segments(timestamps, n_steps))
    if not segs:
        return False
    checked = segs if all_segments else segs[-1:]
    for i, _start, end in checked:
        sp = float(setpoints[i])
        last = tracked[end - 1]
        if not np.isfinite(last) or abs(last - sp) > band_frac * abs(sp):
            return False
    return True


def steady_state_offset(
    tracked: np.ndarray,
    setpoints: Sequence[float],
    timestamps: Sequence[int],
    tail_frac: float = 0.2,
) -> float:
    """Signed offset left at the end of the final setpoint segment.

    Mean of ``tracked - setpoint`` over the last ``tail_frac`` of that segment.
    The sign is kept because it names the failure: sitting 0.3 m below the
    setpoint is a different fault from sitting 0.3 m above it, and the diverging
    dot plots encode the direction.

    Returns
    -------
    float
        Physical units; NaN if the tail is empty or all-NaN.
    """
    tracked = np.asarray(tracked, dtype=float)
    n_steps = tracked.size
    segs = list(_segments(timestamps, n_steps))
    if not segs:
        return float("nan")
    i, start, end = segs[-1]
    length = end - start
    tail_start = end - max(1, int(round(tail_frac * length)))
    tail = tracked[tail_start:end]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail) - float(setpoints[i]))


def recovery_steps(
    tracked: np.ndarray,
    setpoint_series: np.ndarray,
    onset: int,
    band_frac: float = 0.05,
) -> float:
    """Steps from a disturbance onset until the error re-enters the band and stays.

    Parameters
    ----------
    tracked : np.ndarray
        Controlled variable, physical units.
    setpoint_series : np.ndarray
        Active setpoint at each step, same length as ``tracked``.
    onset : int
        Step at which the disturbance was applied.
    band_frac : float, optional
        Half-width of the recovery band as a fraction of the active setpoint.

    Returns
    -------
    float
        Steps to recover, or ``inf`` when the run ends still outside the band.
        Never NaN, so a run that does not recover stays visible in an aggregate
        instead of dropping out of it.
    """
    tracked = np.asarray(tracked, dtype=float)
    sp = np.asarray(setpoint_series, dtype=float)
    n = min(tracked.size, sp.size)
    onset = int(max(0, min(onset, n - 1)))
    seg_t, seg_sp = tracked[onset:n], sp[onset:n]
    band = band_frac * np.abs(seg_sp)
    viol = ~(np.abs(seg_t - seg_sp) <= band)
    if not viol.any():
        return 0.0
    last_viol = int(np.max(np.where(viol)[0]))
    if last_viol == seg_t.size - 1:
        return float("inf")  # still outside the band when the run ended
    return float(last_viol + 1)


def violations(
    x: np.ndarray, lower: Sequence[float], upper: Sequence[float],
    ranges: Sequence[float],
) -> Dict[str, float]:
    """Constraint-violation summary over a state trajectory.

    Violations are reported as a fraction of each variable's operating range,
    so a concentration excursion and a temperature excursion are comparable and
    both benchmarks share one axis.

    Parameters
    ----------
    x : np.ndarray
        Shape ``(nsteps, nx)``, physical units.
    lower, upper : sequence of float
        Per-state bounds, physical units.
    ranges : sequence of float
        Per-state operating range used to make the violation dimensionless.

    Returns
    -------
    dict
        ``n_violations`` (steps with any violated state), ``violation_rate``
        (that count over the number of steps) and ``worst_violation``
        (largest excursion, as a fraction of range).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    ranges = np.asarray(ranges, dtype=float)

    below = np.maximum(lower[None, :] - x, 0.0)
    above = np.maximum(x - upper[None, :], 0.0)
    excursion = np.maximum(below, above) / ranges[None, :]
    excursion = np.where(np.isfinite(excursion), excursion, 0.0)

    per_step = excursion.max(axis=1)
    n_steps = max(1, per_step.size)
    return {
        "n_violations": float((per_step > 0).sum()),
        "violation_rate": float((per_step > 0).sum()) / n_steps,
        "worst_violation": float(per_step.max()) if per_step.size else 0.0,
    }


def solve_time_stats(exec_ms: np.ndarray, sample_time_s: float) -> Dict[str, float]:
    """Latency distribution and the three real-time-feasibility ratios.

    RTF is solve time over the control period, so ``RTF < 1`` means the
    controller finished in time. P50, P95, P99 and the maximum are all kept,
    since the maximum alone cannot separate a chronic overrun from one outlier.
    """
    e = np.asarray(exec_ms, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        nan = float("nan")
        return {
            "solve_ms_mean": nan, "solve_ms_p50": nan, "solve_ms_p95": nan,
            "solve_ms_p99": nan, "solve_ms_max": nan,
            "rtf_mean": nan, "rtf_p99": nan, "rtf_wcet": nan,
        }
    mean, p50 = float(np.mean(e)), float(np.percentile(e, 50))
    p95, p99 = float(np.percentile(e, 95)), float(np.percentile(e, 99))
    mx = float(np.max(e))
    period_ms = sample_time_s * 1000.0
    return {
        "solve_ms_mean": mean,
        "solve_ms_p50": p50,
        "solve_ms_p95": p95,
        "solve_ms_p99": p99,
        "solve_ms_max": mx,
        "rtf_mean": mean / period_ms,
        "rtf_p99": p99 / period_ms,
        "rtf_wcet": mx / period_ms,
    }


def median_iqr(values: Sequence[float]) -> Dict[str, float]:
    """Median and interquartile range over all values.

    Non-finite values rank above every finite one instead of being discarded, so
    a diverged run pushes the median toward the bad end. This assumes larger is
    worse, which holds for the IAE and the settling time but not for a signed
    quantity such as ``steady_state_offset``.
    """
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        nan = float("nan")
        return {"median": nan, "q1": nan, "q3": nan, "iqr": nan, "n": 0}
    finite_max = np.nanmax(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0
    ranked = np.where(np.isfinite(v), v, finite_max * 10.0 + 1.0)
    q1, med, q3 = np.percentile(ranked, [25, 50, 75])
    return {
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "n": int(v.size),
    }


def expand_setpoint_series(
    setpoints: Sequence[float], timestamps: Sequence[int], n_steps: int
) -> np.ndarray:
    """Expand a piecewise-constant schedule into a per-step series."""
    out = np.empty(n_steps, dtype=float)
    for i, start, end in _segments(timestamps, n_steps):
        out[start:end] = float(setpoints[i])
    return out


__all__: List[str] = [
    "iae",
    "nrmse",
    "total_variation",
    "settling_steps",
    "reached_band",
    "steady_state_offset",
    "recovery_steps",
    "violations",
    "solve_time_stats",
    "median_iqr",
    "expand_setpoint_series",
]
