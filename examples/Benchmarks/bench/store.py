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

"""The Parquet result store.

Layout
------
::

    results/
      RUNINFO.json                       provenance: versions, host, seeds, timing
      manifest.parquet                   one row per run: the full specification
      metrics.parquet                    one row per run: every scalar metric
      normalizers.json                   nominal IAE per benchmark (the denominators)
      timeseries/{experiment}_{bench}.parquet
      _shards/                           crash-safe intermediates, removed on consolidate

Runs are keyed by a content-addressed ``run_id``, so re-running the suite
regenerates identical keys and ``--resume`` skips completed work.

Writes are append-only. :func:`consolidate` merges new rows and dedupes on
``run_id``, and it retracts nothing: narrowing a swept range leaves the runs it
replaced in the store, still joined to their case and still drawn. Retraction
goes through :func:`prune`.

Time series are float32, one file per (experiment, benchmark) with a ``run_id``
column. That columnar layout compresses about an order of magnitude better than
one file per run, which keeps the committed store small enough to live in the
repository.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd

from . import config as _config


SHARD_DIR_NAME = "_shards"
TIMESERIES_DIR_NAME = "timeseries"


def results_dir(root: Path | None = None) -> Path:
    return Path(root or _config.RESULTS_DIR)


def _paths(root: Path) -> Dict[str, Path]:
    return {
        "manifest": root / "manifest.parquet",
        "metrics": root / "metrics.parquet",
        "runinfo": root / "RUNINFO.json",
        "normalizers": root / "normalizers.json",
        "timeseries": root / TIMESERIES_DIR_NAME,
        "shards": root / SHARD_DIR_NAME,
    }


def ensure_dirs(root: Path | None = None) -> Dict[str, Path]:
    """Create the store directories and return the canonical paths."""
    root = results_dir(root)
    p = _paths(root)
    root.mkdir(parents=True, exist_ok=True)
    p["timeseries"].mkdir(parents=True, exist_ok=True)
    p["shards"].mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------

def write_shard(
    shard_id: str,
    rows: List[Dict[str, Any]],
    series: Dict[str, Dict[str, np.ndarray]],
    experiment: str,
    benchmark_of: Dict[str, str],
    root: Path | None = None,
) -> None:
    """Persist one batch of completed runs.

    Shards are written as the sweep progresses, so an interrupted run loses at
    most the batch in flight. :func:`consolidate` folds them into the final
    files.
    """
    p = ensure_dirs(root)
    shard_dir = p["shards"] / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_parquet(shard_dir / "metrics.parquet", index=False)

    by_bench: Dict[str, List[pd.DataFrame]] = {}
    for run_id, cols in series.items():
        frame = pd.DataFrame(cols)
        frame.insert(0, "run_id", run_id)
        by_bench.setdefault(benchmark_of[run_id], []).append(frame)
    for bench_key, frames in by_bench.items():
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(
            shard_dir / f"ts_{experiment}_{bench_key}.parquet",
            index=False, compression="zstd",
        )


def consolidate(root: Path | None = None) -> Dict[str, int]:
    """Merge every shard into the canonical store files and delete the shards.

    Returns
    -------
    dict
        Row counts of the consolidated ``metrics`` and ``timeseries`` tables.
    """
    p = ensure_dirs(root)
    shard_dirs = sorted(d for d in p["shards"].iterdir() if d.is_dir())
    if not shard_dirs:
        return {"metrics": 0, "timeseries": 0}

    metric_frames: List[pd.DataFrame] = []
    if p["metrics"].exists():
        metric_frames.append(pd.read_parquet(p["metrics"]))
    for d in shard_dirs:
        f = d / "metrics.parquet"
        if f.exists():
            metric_frames.append(pd.read_parquet(f))

    ts_groups: Dict[str, List[pd.DataFrame]] = {}
    for d in shard_dirs:
        for f in sorted(d.glob("ts_*.parquet")):
            ts_groups.setdefault(f.name[len("ts_"):], []).append(pd.read_parquet(f))

    n_ts = 0
    for name, frames in ts_groups.items():
        target = p["timeseries"] / name
        if target.exists():
            frames.insert(0, pd.read_parquet(target))
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["run_id", "step"], keep="last")
        merged.to_parquet(target, index=False, compression="zstd")
        n_ts += len(merged)

    n_metrics = 0
    if metric_frames:
        merged = pd.concat(metric_frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["run_id"], keep="last")
        merged.to_parquet(p["metrics"], index=False, compression="zstd")
        manifest_cols = [c for c in merged.columns if c in _config.SPEC_FIELDS] + [
            "run_id", "experiment", "case", "benchmark"
        ]
        manifest = merged[list(dict.fromkeys(manifest_cols))]
        manifest.to_parquet(p["manifest"], index=False, compression="zstd")
        n_metrics = len(merged)

    shutil.rmtree(p["shards"])
    p["shards"].mkdir(parents=True, exist_ok=True)
    return {"metrics": n_metrics, "timeseries": n_ts}


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def existing_run_ids(root: Path | None = None) -> Set[str]:
    """Every ``run_id`` already in the store, including unconsolidated shards.

    ``--resume`` consults this, so an interrupted sweep restarted with the same
    configs picks up where it stopped.
    """
    p = _paths(results_dir(root))
    ids: Set[str] = set()
    if p["metrics"].exists():
        ids |= set(pd.read_parquet(p["metrics"], columns=["run_id"])["run_id"])
    if p["shards"].exists():
        for f in p["shards"].glob("*/metrics.parquet"):
            ids |= set(pd.read_parquet(f, columns=["run_id"])["run_id"])
    return ids


def prune(
    keep: Dict[str, Set[str]], root: Path | None = None, dry_run: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Drop stored runs that the current configs no longer define.

    A ``run_id`` digests the whole spec, so narrowing a swept range stops
    referring to the old runs without replacing them. They stay in
    ``metrics.parquet`` and keep being plotted, since the figures select on the
    case name and read whatever the store holds. ``--force`` re-simulates the
    new points beside the old ones, so retraction needs its own command.

    Parameters
    ----------
    keep : dict
        ``{experiment_id: {run_id, ...}}``, the runs the configs now define.
        Only experiments named here are touched, so pruning one leaves the
        others' results alone.
    root : Path, optional
    dry_run : bool, optional
        Report what would be dropped and write nothing.

    Returns
    -------
    dict
        Per experiment: rows dropped from metrics and time series, plus a few of
        the dropped run ids.
    """
    p = _paths(results_dir(root))
    report: Dict[str, Dict[str, Any]] = {}
    if not p["metrics"].exists():
        return report

    metrics = pd.read_parquet(p["metrics"])
    stale_ids: Set[str] = set()
    for experiment, keep_ids in keep.items():
        rows = metrics[metrics["experiment"] == experiment]
        gone = rows[~rows["run_id"].isin(keep_ids)]
        if gone.empty:
            continue
        stale_ids |= set(gone["run_id"])
        report[experiment] = {
            "metrics": len(gone),
            "cases": gone["case"].value_counts().to_dict(),
            "sample": gone.head(5)["run_id"].tolist(),
        }
    if not stale_ids:
        return report

    if dry_run:
        for experiment in keep:
            for f in sorted(p["timeseries"].glob(f"{experiment}_*.parquet")):
                ids = pd.read_parquet(f, columns=["run_id"])["run_id"]
                n = int(ids.isin(stale_ids).sum())
                if n and experiment in report:
                    report[experiment]["timeseries"] = (
                        report[experiment].get("timeseries", 0) + n
                    )
        return report

    kept = metrics[~metrics["run_id"].isin(stale_ids)]
    kept.to_parquet(p["metrics"], index=False, compression="zstd")
    manifest_cols = [c for c in kept.columns if c in _config.SPEC_FIELDS] + [
        "run_id", "experiment", "case", "benchmark"
    ]
    kept[list(dict.fromkeys(manifest_cols))].to_parquet(
        p["manifest"], index=False, compression="zstd"
    )

    for experiment in keep:
        for f in sorted(p["timeseries"].glob(f"{experiment}_*.parquet")):
            frame = pd.read_parquet(f)
            trimmed = frame[~frame["run_id"].isin(stale_ids)]
            if len(trimmed) == len(frame):
                continue
            report.setdefault(experiment, {}).setdefault("timeseries", 0)
            report[experiment]["timeseries"] += len(frame) - len(trimmed)
            trimmed.to_parquet(f, index=False, compression="zstd")

    return report


def load_metrics(root: Path | None = None) -> pd.DataFrame:
    """The full metrics table.

    Every run ever completed is present, failures included. Callers must not
    filter on ``completed`` before aggregating; see the module docstring of
    :mod:`bench.metrics`.
    """
    p = _paths(results_dir(root))
    if not p["metrics"].exists():
        raise FileNotFoundError(
            f"no metrics at {p['metrics']}. Run:\n"
            f"    python examples/Benchmarks/run_experiments.py --all"
        )
    return pd.read_parquet(p["metrics"])


def load_timeseries(
    experiment: str, benchmark: str, run_ids: Iterable[str] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Time series for one (experiment, benchmark), optionally a subset of runs."""
    p = _paths(results_dir(root))
    f = p["timeseries"] / f"{experiment}_{benchmark}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"no time series at {f}")
    frame = pd.read_parquet(f)
    if run_ids is not None:
        wanted = set(run_ids)
        frame = frame[frame["run_id"].isin(wanted)]
    return frame


def has_timeseries(experiment: str, benchmark: str, root: Path | None = None) -> bool:
    p = _paths(results_dir(root))
    return (p["timeseries"] / f"{experiment}_{benchmark}.parquet").exists()


# --------------------------------------------------------------------------
# Normalizers and provenance
# --------------------------------------------------------------------------

def compute_normalizers(root: Path | None = None) -> Dict[str, Dict[str, float]]:
    """Nominal IAE per benchmark, the denominator of every normalized IAE.

    The nominal run is the closed-loop experiment (``exp5``), the configuration
    that produced the paper's own figures. Dividing by it puts both benchmarks
    on one axis where 1.0 means "as good as the reported figure". The absolute
    value is stored alongside because it belongs in the caption.
    """
    metrics = load_metrics(root)
    nominal = metrics[metrics["experiment"] == "exp5"]
    if nominal.empty:
        # Fall back to the unperturbed run of whatever else is present, so the
        # normalization has a defensible reference before exp5 has run.
        nominal = metrics[
            (metrics.get("noise_sigma_pct", 0) == 0)
            & (metrics.get("dist_kind", "none") == "none")
        ]

    out: Dict[str, Dict[str, float]] = {}
    for bench_key, group in nominal.groupby("benchmark"):
        neural = group[group["controller"] == "neural"]
        source = neural if not neural.empty else group
        value = float(np.nanmedian(source["iae_tracked"]))
        bench = _config.BENCHMARKS[bench_key]
        out[bench_key] = {
            "iae_tracked": value,
            "n_runs": int(len(source)),
            "tracked_variable": bench.state_keys[bench.track_index],
            "unit": f"{bench.state_units[bench.track_index]}*s",
        }

    p = ensure_dirs(root)
    p["normalizers"].write_text(json.dumps(out, indent=2))
    return out


def load_normalizers(root: Path | None = None) -> Dict[str, Dict[str, float]]:
    p = _paths(results_dir(root))
    if not p["normalizers"].exists():
        return compute_normalizers(root)
    return json.loads(p["normalizers"].read_text())


def attach_normalized_iae(
    metrics: pd.DataFrame, normalizers: Dict[str, Dict[str, float]] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Add an ``iae_norm`` column: IAE divided by the benchmark's nominal run."""
    normalizers = normalizers or load_normalizers(root)
    denom = metrics["benchmark"].map(
        lambda b: normalizers.get(b, {}).get("iae_tracked", np.nan)
    )
    out = metrics.copy()
    out["iae_norm"] = out["iae_tracked"] / denom
    return out


def attach_normalized_tv(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add ``tv_norm``: input total variation in fractions of actuator range.

    The raw ``tv_u`` sums the per-channel total variations in their own physical
    units, which carries no meaning for the CSTR: feed flow runs to 100 h^-1
    while heat removal runs to 8500 kJ/h, so the sum reports the heat channel
    and little else. Dividing each channel by its own span before summing gives
    the fraction of its travel each actuator used, which is the right measure of
    chatter and is comparable across benchmarks.
    """
    out = metrics.copy()
    values = np.zeros(len(out), dtype=float)
    for bench_key, bench in _config.BENCHMARKS.items():
        mask = (out["benchmark"] == bench_key).to_numpy()
        if not mask.any():
            continue
        total = np.zeros(int(mask.sum()), dtype=float)
        for i, key in enumerate(bench.action_keys):
            column = f"tv_{key}"
            if column not in out:
                continue
            lo, hi = _config.ACTION_BOUNDS[bench_key][i]
            span = abs(hi - lo) or 1.0
            total += out.loc[mask, column].to_numpy(dtype=float) / span
        values[mask] = total
    out["tv_norm"] = values
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_config.RESULTS_DIR.parents[2]),
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def write_runinfo(
    extra: Dict[str, Any] | None = None, root: Path | None = None
) -> Dict[str, Any]:
    """Record provenance, so a result can be traced back to how it was made."""
    from . import seeds as _seeds

    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "casadi", "torch", "matplotlib", "joblib"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "?"))
        except Exception:  # noqa: BLE001
            versions[name] = "not installed"

    info: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "host": platform.node(),
        "platform": platform.platform(),
        "versions": versions,
        "seeding": _seeds.describe(),
    }
    info.update(extra or {})
    p = ensure_dirs(root)
    p["runinfo"].write_text(json.dumps(info, indent=2))
    return info


def load_runinfo(root: Path | None = None) -> Dict[str, Any]:
    p = _paths(results_dir(root))
    return json.loads(p["runinfo"].read_text()) if p["runinfo"].exists() else {}


__all__ = [
    "ensure_dirs", "results_dir", "write_shard", "consolidate",
    "existing_run_ids", "prune",
    "load_metrics", "load_timeseries", "has_timeseries",
    "compute_normalizers", "load_normalizers", "attach_normalized_iae",
    "attach_normalized_tv",
    "write_runinfo", "load_runinfo",
]
