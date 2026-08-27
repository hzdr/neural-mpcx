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

"""Parallel execution of run groups with joblib.

Four constraints shape this module.

Runs go to processes, not threads, because ``neuralmpcx`` reads IPOPT statistics
off the CasADi solver after the call returns, so two threads sharing one ``Nlp``
would read each other's numbers. The loky backend gives every solver its own
interpreter.

Each worker builds its own controller. A built ``Mpc`` cannot cross a process
boundary: ``__getstate__`` drops its CasADi attributes, so it survives pickling
and arrives gutted. Workers receive plain dicts and build locally.

Each worker gets one BLAS thread. The entry point sets the environment in the
parent, loky inherits it, and each worker asserts it again for torch, which
ignores the variable once imported.

Timing runs are serial. An experiment flagged ``timing_critical`` produces the
real-time-feasibility numbers, and twenty workers on twenty cores would measure
contention between them. Those runs execute in a separate pass after everything
else.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from joblib import Parallel, delayed
from tqdm import tqdm

from . import adapters as _adapters
from . import config as _config
from . import store as _store


#: Measured on the reference machine (20 cores, 2026-08): seconds per simulated
#: step at the nominal configuration. It feeds the dry-run estimate, which is
#: there to warn before a multi-hour sweep and is not a precise prediction.
SECONDS_PER_STEP = {"cstr": 0.225, "cts": 0.038}


def _pin_threads() -> None:
    """Force single-threaded numerics inside a worker."""
    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001 - torch may not be installed
        pass


def _worker(specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Top-level worker callable (must be importable for the loky backend)."""
    _pin_threads()
    return _adapters.run_group(specs)


@dataclass
class RunnerOptions:
    """Knobs for one invocation of the suite."""

    n_jobs: int = -1
    timing_jobs: int = 1
    #: Run groups dispatched per batch. None derives it from ``n_jobs``: a
    #: batch smaller than the worker pool leaves workers idle, which on a
    #: 64-core node with a fixed default of 32 would waste half the machine.
    batch_size: int | None = None
    resume: bool = True
    force: bool = False
    limit: int | None = None
    dry_run: bool = False
    root: Any | None = None
    verbose: bool = True


@dataclass
class RunnerReport:
    """What a sweep did."""

    planned_runs: int = 0
    executed_runs: int = 0
    skipped_runs: int = 0
    failed_runs: int = 0
    wall_seconds: float = 0.0
    per_experiment: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def estimate_seconds(groups: Sequence[Sequence[Dict[str, Any]]]) -> float:
    """Rough serial wall-clock estimate for a set of run groups."""
    total = 0.0
    for group in groups:
        for spec in group:
            per_step = SECONDS_PER_STEP.get(spec["benchmark"], 0.1)
            # The solve cost grows with the horizon and, for the neural path,
            # with the hidden size; both enter the NLP roughly linearly.
            horizon_factor = spec.get("horizon", 20) / _config.DEFAULTS[
                spec["benchmark"]
            ]["horizon"]
            hidden = spec.get("hidden_size", 16)
            hidden_factor = (
                1.0 if hidden in (None, -1)
                else hidden / _config.DEFAULTS[spec["benchmark"]]["hidden_size"]
            )
            total += (
                spec.get("num_iter", 100) * per_step
                * max(0.25, horizon_factor) * max(0.25, hidden_factor ** 0.5)
            )
    return total


def _batched(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def effective_batch_size(options: RunnerOptions, n_jobs: int) -> int:
    """How many run groups to dispatch at once.

    A batch is never smaller than the pool, so every worker has something to do,
    and several groups per worker keep it fed while the driver writes a shard.
    The batch is also the crash-safety granularity, since the driver writes a
    shard after each one. The size trades what an interruption costs against how
    often the driver stops to write.
    """
    if options.batch_size is not None:
        return max(1, int(options.batch_size))
    return max(32, 4 * max(1, int(n_jobs)))


def plan(
    experiments: Sequence[_config.Experiment], options: RunnerOptions
) -> Dict[str, List[List[Dict[str, Any]]]]:
    """Expand experiments into run groups, applying ``--resume`` and ``--limit``.

    Returns
    -------
    dict
        Experiment id -> list of run groups still to execute.
    """
    done = set() if options.force else _store.existing_run_ids(options.root)
    out: Dict[str, List[List[Dict[str, Any]]]] = {}
    for exp in experiments:
        groups = _config.expand(exp)
        if options.resume and not options.force:
            groups = [
                [s for s in group if s["run_id"] not in done] for group in groups
            ]
            groups = [g for g in groups if g]
        if options.limit is not None:
            trimmed, count = [], 0
            for group in groups:
                if count >= options.limit:
                    break
                trimmed.append(group)
                count += len(group)
            groups = trimmed
        out[exp.id] = groups
    return out


def _execute(
    experiment: _config.Experiment,
    groups: List[List[Dict[str, Any]]],
    options: RunnerOptions,
    on_progress: Callable[[int], None] | None = None,
    session: str = "",
) -> Dict[str, Any]:
    """Run one experiment's groups and write shards as batches complete."""
    n_jobs = options.timing_jobs if experiment.timing_critical else options.n_jobs
    executed = failed = 0
    started = time.perf_counter()

    batch_size = effective_batch_size(options, n_jobs)
    for batch_index, batch in enumerate(_batched(groups, batch_size)):
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_worker)(group) for group in batch
        )

        rows: List[Dict[str, Any]] = []
        series: Dict[str, Dict[str, Any]] = {}
        benchmark_of: Dict[str, str] = {}
        for result in results:
            rows.extend(result["rows"])
            series.update(result["series"])
        for row in rows:
            benchmark_of[row["run_id"]] = row["benchmark"]
            if row.get("error"):
                failed += 1
        executed += len(rows)

        _store.write_shard(
            shard_id=f"{experiment.id}_{session}_{batch_index:05d}",
            rows=rows,
            series=series,
            experiment=experiment.id,
            benchmark_of=benchmark_of,
            root=options.root,
        )
        if on_progress is not None:
            on_progress(len(rows))

    return {
        "executed": executed,
        "failed": failed,
        "wall_seconds": time.perf_counter() - started,
        "n_jobs": n_jobs,
    }


def run(
    experiments: Sequence[_config.Experiment], options: RunnerOptions
) -> RunnerReport:
    """Execute a set of experiments and consolidate the store.

    Parallel experiments run first and saturate the machine. Timing-critical
    experiments run afterwards, serially, so contention from the rest of the
    sweep stays out of their latency measurements.
    """
    report = RunnerReport()
    planned = plan(experiments, options)
    report.planned_runs = sum(
        len(g) for groups in planned.values() for g in groups
    )

    all_expanded = {exp.id: _config.expand(exp) for exp in experiments}
    total_defined = sum(
        len(g) for groups in all_expanded.values() for g in groups
    )
    report.skipped_runs = total_defined - report.planned_runs

    if options.dry_run:
        for exp in experiments:
            groups = planned[exp.id]
            n_runs = sum(len(g) for g in groups)
            report.per_experiment[exp.id] = {
                "name": exp.name,
                "groups": len(groups),
                "runs": n_runs,
                "timing_critical": exp.timing_critical,
                "estimated_serial_seconds": estimate_seconds(groups),
            }
        return report

    parallel_exps = [e for e in experiments if not e.timing_critical]
    serial_exps = [e for e in experiments if e.timing_critical]

    # Unique per invocation, so this run cannot overwrite shards left by a
    # previous, possibly interrupted, one.
    session = uuid.uuid4().hex[:8]

    started = time.perf_counter()
    bar = tqdm(
        total=report.planned_runs, desc="runs", unit="run", ncols=90,
        disable=not options.verbose,
    )
    try:
        for exp in list(parallel_exps) + list(serial_exps):
            groups = planned[exp.id]
            if not groups:
                report.per_experiment[exp.id] = {
                    "name": exp.name, "executed": 0, "failed": 0,
                    "wall_seconds": 0.0, "note": "nothing to do (resumed)",
                }
                continue
            bar.set_description(
                f"{exp.id} ({'serial' if exp.timing_critical else 'parallel'})"
            )
            stats = _execute(exp, groups, options, on_progress=bar.update,
                             session=session)
            stats["name"] = exp.name
            report.per_experiment[exp.id] = stats
            report.executed_runs += stats["executed"]
            report.failed_runs += stats["failed"]
    finally:
        bar.close()

    counts = _store.consolidate(options.root)
    report.wall_seconds = time.perf_counter() - started
    report.per_experiment["_store"] = counts
    return report


__all__ = [
    "RunnerOptions", "RunnerReport", "run", "plan", "estimate_seconds",
    "effective_batch_size",
    "SECONDS_PER_STEP",
]
