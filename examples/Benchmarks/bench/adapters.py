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

"""Bridge between a run spec and the example scripts that do the simulating.

Each example module defines its own ``RunConfig`` dataclass and a
``simulate(cfg) -> dict``. This module maps a benchmark-agnostic run spec onto
whichever ``RunConfig`` applies, then reduces the returned trajectories to the
metric row and the time-series frame the store persists.

The example scripts pin the BLAS thread count at import time, which takes effect
only before NumPy and torch load. Workers therefore import them inside the
worker process, and a per-process cache holds that import to once per worker.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import config as _config
from . import metrics as _metrics


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIRS = {
    "cstr": PROJECT_ROOT / "examples" / "CSTR",
    "cts": PROJECT_ROOT / "examples" / "Cascaded_Two_Tank_System",
}

_MODULE_CACHE: Dict[str, Any] = {}


def load_example_module(benchmark: str, controller: str):
    """Import and cache the example module implementing one controller.

    Raises ``ValueError`` when the combination has no implementation. The
    two-tank has no physics-MPC script, so ``("cts", "nmpc")`` is unavailable.
    """
    bench = _config.BENCHMARKS[benchmark]
    if controller == "neural":
        module_name = bench.module
    elif controller == "nmpc":
        if bench.nmpc_module is None:
            raise ValueError(
                f"benchmark {benchmark!r} has no physics-MPC script; "
                "the neural-vs-physics comparison is CSTR-only"
            )
        module_name = bench.nmpc_module
    else:
        raise ValueError(f"unknown controller {controller!r}")

    cache_key = f"{benchmark}:{module_name}"
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    example_dir = str(EXAMPLE_DIRS[benchmark])
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    module = importlib.import_module(module_name)
    _MODULE_CACHE[cache_key] = module
    return module


def build_run_config(spec: Dict[str, Any], module) -> Any:
    """Instantiate the example module's ``RunConfig`` from a run spec.

    Only the fields the target dataclass declares are passed through, so a spec
    carrying two-tank fields does not break the CSTR module or the reverse.
    """
    cfg_cls = module.RunConfig
    accepted = set(getattr(cfg_cls, "__dataclass_fields__", {}))
    kwargs = {k: v for k, v in spec.items() if k in accepted and v is not None}
    # x0 is a list in JSON; the plants reshape it themselves.
    if spec.get("x0") is not None and "x0" in accepted:
        kwargs["x0"] = list(spec["x0"])
    return cfg_cls(**kwargs)


def run_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single run spec and return its raw simulation output."""
    module = load_example_module(spec["benchmark"], spec.get("controller", "neural"))
    cfg = build_run_config(spec, module)
    return module.simulate(cfg, progress=False)


def _tracked_physical(bench: _config.Benchmark, x_phys: np.ndarray) -> np.ndarray:
    """Controlled-variable series in physical units, one value per applied step.

    ``X`` carries the initial state in row 0, so the trajectory aligned with the
    applied inputs is ``X[1:]``.
    """
    return x_phys[1:, bench.track_index]


def _prediction_physical(
    bench: _config.Benchmark, x_pred_raw: Any, n_steps: int
) -> np.ndarray:
    """One-step predictions as a physical ``(n_steps, ncols)`` array.

    The CSTR model predicts all four states and the two-tank model only ``h_2``,
    so the column count tells them apart. The shape comes from the raw array's
    dimensionality, not its size: a short run can give a wide array the same
    element count as a long narrow one.
    """
    arr = np.asarray(x_pred_raw, dtype=float)
    x_pred = arr.reshape(-1, 1) if arr.ndim == 1 else np.atleast_2d(arr)
    x_pred = x_pred[:n_steps]
    if x_pred.shape[1] == len(bench.state_keys):
        return bench.to_physical_states(x_pred)
    if bench.stored_normalized:
        lo = bench.norm_min[bench.track_index]
        hi = bench.norm_max[bench.track_index]
        return x_pred * (hi - lo) + lo
    return x_pred


def summarize(
    spec: Dict[str, Any],
    raw: Dict[str, Any],
    nominal_iae: float | None = None,
) -> Dict[str, Any]:
    """Reduce one run's trajectories to a single row of scalar metrics.

    Parameters
    ----------
    spec : dict
        The run spec, echoed into the row for joining.
    raw : dict
        Whatever ``simulate`` returned.
    nominal_iae : float, optional
        Denominator for the normalized IAE. :mod:`bench.store` fills it in once
        the nominal runs are known.

    Returns
    -------
    dict
        One row for ``metrics.parquet``, populated even for runs with failed
        solves. Failure is recorded in ``n_failed_solves`` and ``completed``;
        the row is always written.
    """
    bench = _config.BENCHMARKS[spec["benchmark"]]
    x_phys = bench.to_physical_states(np.atleast_2d(raw["X"]))
    u_phys = bench.to_physical_actions(raw["U"], _config.ACTION_BOUNDS[bench.key])

    n_steps = int(np.asarray(raw["exec_ms"]).size)
    tracked = _tracked_physical(bench, x_phys)
    sp_series = _metrics.expand_setpoint_series(
        bench.setpoint_values, bench.setpoint_timestamps, n_steps
    )

    row: Dict[str, Any] = {k: v for k, v in spec.items()}
    row["n_steps"] = n_steps

    # --- tracking -----------------------------------------------------------
    # Only the tracked variable has a setpoint. The schedule holds the others
    # at 0 as padding, so an IAE against them would measure the padding.
    tracked_key = bench.state_keys[bench.track_index]
    row[f"iae_{tracked_key}"] = _metrics.iae(tracked, sp_series, bench.sample_time_s)
    row["iae_tracked"] = row[f"iae_{tracked_key}"]

    # --- prediction quality -------------------------------------------------
    pred_phys = _prediction_physical(bench, raw["X_pred"], n_steps)
    if pred_phys.shape[1] == len(bench.state_keys):
        for i, key in enumerate(bench.state_keys):
            row[f"nrmse_pred_{key}"] = _metrics.nrmse(x_phys[1:, i], pred_phys[:, i])
        row["nrmse_pred_tracked"] = row[f"nrmse_pred_{tracked_key}"]
    else:
        # The two-tank model predicts only h_2, so there is one prediction
        # series instead of one per state.
        row["nrmse_pred_tracked"] = _metrics.nrmse(tracked, pred_phys[:, 0])

    # --- actuator effort ----------------------------------------------------
    row["tv_u"] = _metrics.total_variation(u_phys)
    for i, key in enumerate(bench.action_keys):
        row[f"tv_{key}"] = _metrics.total_variation(u_phys[:, i])

    # --- dynamics -----------------------------------------------------------
    row["settling_steps"] = _metrics.settling_steps(
        tracked, bench.setpoint_values, bench.setpoint_timestamps
    )
    row["settling_time_s"] = row["settling_steps"] * bench.sample_time_s
    row["ss_offset"] = _metrics.steady_state_offset(
        tracked, bench.setpoint_values, bench.setpoint_timestamps
    )
    row["recovery_steps"] = (
        _metrics.recovery_steps(tracked, sp_series, int(spec.get("dist_onset", 0)))
        if spec.get("dist_kind", "none") != "none"
        else float("nan")
    )

    # --- constraints --------------------------------------------------------
    row.update(
        _metrics.violations(x_phys[1:], bench.x_lower, bench.x_upper, bench.x_ranges)
    )

    # --- timing -------------------------------------------------------------
    row.update(_metrics.solve_time_stats(raw["exec_ms"], bench.sample_time_s))

    # --- status -------------------------------------------------------------
    row["n_solves"] = int(raw.get("n_solves", n_steps))
    row["n_failed_solves"] = int(raw.get("n_failed_solves", 0))
    row["reached_band"] = bool(
        _metrics.reached_band(tracked, bench.setpoint_values, bench.setpoint_timestamps)
    )
    row["reached_band_all_segments"] = bool(
        _metrics.reached_band(
            tracked, bench.setpoint_values, bench.setpoint_timestamps,
            all_segments=True,
        )
    )
    row["completed"] = bool(row["n_failed_solves"] == 0 and row["reached_band"])

    if nominal_iae:
        row["iae_norm"] = row["iae_tracked"] / nominal_iae

    return row


def timeseries_frame(spec: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Tidy per-step columns for one run, in physical units.

    Holds the columns committed for every run in the study: states, prediction,
    input, setpoint, solve time and the per-step solve status. The trajectory
    figures need nothing more.
    """
    bench = _config.BENCHMARKS[spec["benchmark"]]
    x_phys = bench.to_physical_states(np.atleast_2d(raw["X"]))
    u_phys = bench.to_physical_actions(raw["U"], _config.ACTION_BOUNDS[bench.key])
    n_steps = int(np.asarray(raw["exec_ms"]).size)

    cols: Dict[str, np.ndarray] = {
        "step": np.arange(n_steps, dtype=np.int32),
        "time_s": np.arange(n_steps, dtype=np.float32) * bench.sample_time_s,
    }
    for i, key in enumerate(bench.state_keys):
        cols[key] = x_phys[1 : n_steps + 1, i].astype(np.float32)
    for i, key in enumerate(bench.action_keys):
        cols[key] = u_phys[:n_steps, i].astype(np.float32)

    pred_phys = _prediction_physical(bench, raw["X_pred"], n_steps)
    if pred_phys.shape[1] == len(bench.state_keys):
        for i, key in enumerate(bench.state_keys):
            cols[f"{key}_pred"] = pred_phys[:, i].astype(np.float32)
    else:
        tracked_key = bench.state_keys[bench.track_index]
        cols[f"{tracked_key}_pred"] = pred_phys[:, 0].astype(np.float32)

    cols["sp"] = _metrics.expand_setpoint_series(
        bench.setpoint_values, bench.setpoint_timestamps, n_steps
    ).astype(np.float32)
    cols["exec_ms"] = np.asarray(raw["exec_ms"], dtype=np.float32)
    cols["solve_ok"] = np.asarray(raw["solve_ok"], dtype=bool)
    return cols


def run_group(specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute one group of run specs inside a single worker.

    Returns
    -------
    dict
        ``rows`` (metric rows) and ``series`` (``run_id`` -> per-step columns).
        A run that raises is still reported, with ``error`` set and
        ``completed`` False, so a crash shrinks the completed count and stays in
        the study.
    """
    raws: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    for spec in specs:
        try:
            raws[spec["run_id"]] = run_one(spec)
        except Exception as exc:  # noqa: BLE001 - the failure is recorded below
            errors[spec["run_id"]] = f"{type(exc).__name__}: {exc}"

    rows: List[Dict[str, Any]] = []
    series: Dict[str, Dict[str, np.ndarray]] = {}
    for spec in specs:
        run_id = spec["run_id"]
        if run_id in errors:
            row = dict(spec)
            row.update(
                {"completed": False, "error": errors[run_id],
                 "n_failed_solves": -1, "n_steps": 0}
            )
            rows.append(row)
            continue
        raw = raws[run_id]
        row = summarize(spec, raw)
        row["error"] = ""
        rows.append(row)
        series[run_id] = timeseries_frame(spec, raw)

    return {"rows": rows, "series": series}


__all__ = [
    "load_example_module", "build_run_config", "run_one", "run_group",
    "summarize", "timeseries_frame", "PROJECT_ROOT",
]
