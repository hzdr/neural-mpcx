#!/usr/bin/env python
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

"""Run the benchmark suite in parallel.

Usage
-----
::

    python examples/Benchmarks/run_experiments.py --all
    python examples/Benchmarks/run_experiments.py --all --dry-run
    python examples/Benchmarks/run_experiments.py --only exp1 exp5 --n-jobs 16
    python examples/Benchmarks/run_experiments.py --only exp1 --limit 20

Experiments are declared in ``configs/*.json``; edit those, not this file.
Results land in ``results/`` as Parquet, and ``reproduce_all.py`` turns them
into every figure and table.

``--resume`` is on by default, so an interrupted sweep continues where it
stopped. ``--force`` re-simulates, and ``--prune`` retracts runs the configs no
longer define.
"""

from __future__ import annotations

import os

# Pin BLAS before NumPy, torch or CasADi load anywhere. The loky workers inherit
# this environment, and the example modules assert it again at their own import
# time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bench import config as bench_config  # noqa: E402
from bench import runner as bench_runner  # noqa: E402
from bench import store as bench_store  # noqa: E402


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all", action="store_true", help="run every experiment in configs/"
    )
    selection.add_argument(
        "--only", nargs="+", metavar="EXP",
        help="run only these experiment ids, e.g. --only exp1 exp4",
    )

    parser.add_argument(
        "--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="parallel workers for ordinary experiments (default: cores - 1)",
    )
    parser.add_argument(
        "--timing-jobs", type=int, default=1,
        help="workers for timing-critical experiments. The default of 1 keeps "
             "the latency measurements uncontended. Raising it puts contention "
             "into the reported worst case.",
    )
    parser.add_argument(
        "--host-label", default=None,
        help="a name for this machine, recorded in RUNINFO.json. Optional, and "
             "nothing is collected without it: the sweep is identified by an "
             "opaque local id, the CPU model and the OS. Pass something you are "
             "willing to publish, since the store is committed.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="run groups dispatched per batch; a crash-safe shard is written "
             "after each. Default: max(32, 4 x n_jobs), which keeps every "
             "worker fed. A batch smaller than the pool leaves workers idle.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-simulate runs already present in the store",
    )
    parser.add_argument(
        "--prune", action="store_true",
        help="before running, drop stored runs the current configs no longer "
             "define. The store is append-only and keyed by a digest of the "
             "whole spec, so narrowing a swept range leaves the old runs behind "
             "and the figures keep drawing them; --force does not retract them "
             "either. Scoped to the experiments in this invocation, so "
             "--only exp3 --prune leaves exp1 alone. With --dry-run it reports "
             "what it would drop and changes nothing.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of runs per experiment (for calibration)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report the design sizes and a wall-clock estimate, run nothing",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="run the smoke-scale copies in configs_smoke/ into results_smoke/. "
             "The same code paths and cases as the full study over tiny designs "
             "and short simulations, so the whole pipeline finishes in minutes. "
             "Use it to check the pipeline before committing a cluster to the "
             "real sweep. A smoke run produces no results.",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None, help="override configs/"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="override results/"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the progress bar")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    config_dir = args.config_dir
    if args.smoke:
        config_dir = config_dir or HERE / "configs_smoke"
        args.out = args.out or HERE / "results_smoke"
        print("SMOKE MODE: reduced designs and short runs, producing no "
              "results.\n")
    experiments = bench_config.load_all_experiments(config_dir)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {e.id for e in experiments}
        if unknown:
            print(
                f"error: unknown experiment id(s) {sorted(unknown)}; "
                f"available: {sorted(e.id for e in experiments)}",
                file=sys.stderr,
            )
            return 2
        experiments = [e for e in experiments if e.id in wanted]

    options = bench_runner.RunnerOptions(
        n_jobs=args.n_jobs,
        timing_jobs=args.timing_jobs,
        batch_size=args.batch_size,
        resume=not args.force,
        force=args.force,
        limit=args.limit,
        dry_run=args.dry_run,
        root=args.out,
        verbose=not args.quiet,
    )

    if args.prune:
        keep = {
            exp.id: {s["run_id"] for group in bench_config.expand(exp)
                     for s in group}
            for exp in experiments
        }
        dropped = bench_store.prune(keep, root=args.out, dry_run=args.dry_run)
        if dropped:
            verb = "would be dropped" if args.dry_run else "dropped"
            for exp_id, info in sorted(dropped.items()):
                cases = ", ".join(f"{c} (x{n})"
                                  for c, n in sorted(info.get("cases", {}).items()))
                print(f"prune: {exp_id}: {info.get('metrics', 0)} stale run(s) "
                      f"{verb}"
                      + (f", {info['timeseries']} time-series row(s)"
                         if info.get("timeseries") else "")
                      + (f"\n  {cases}" if cases else ""))
            print()
        else:
            print("prune: nothing stale; the store matches its configs.\n")

    if args.dry_run:
        report = bench_runner.run(experiments, options)
        total_serial = 0.0
        print(f"\n{'experiment':<8} {'runs':>7} {'groups':>7}  {'mode':<9} "
              f"{'est. serial':>12} {'est. wall':>10}")
        print("-" * 62)
        for exp in experiments:
            info = report.per_experiment[exp.id]
            serial = info["estimated_serial_seconds"]
            total_serial += serial
            jobs = args.timing_jobs if info["timing_critical"] else args.n_jobs
            mode = "serial" if info["timing_critical"] else "parallel"
            print(
                f"{exp.id:<8} {info['runs']:>7} {info['groups']:>7}  {mode:<9} "
                f"{_fmt_duration(serial):>12} {_fmt_duration(serial / max(1, jobs)):>10}"
            )
        print("-" * 62)
        print(f"{'total':<8} {report.planned_runs:>7} "
              f"{'':>7}  {'':<9} {_fmt_duration(total_serial):>12}")
        if report.skipped_runs:
            print(f"\n{report.skipped_runs} run(s) already in the store would be "
                  f"skipped (--force to redo).")
        print(f"\nWorkers: {args.n_jobs} parallel / {args.timing_jobs} for timing.")
        print("Estimates come from a 20-core reference machine and are "
              "indicative, not predictions.")
        return 0

    print(f"Running {len(experiments)} experiment(s): "
          f"{', '.join(e.id for e in experiments)}")
    print(f"Workers: {args.n_jobs} parallel, {args.timing_jobs} for timing-critical.")
    print(f"Planned: {report_planned(experiments, options)} run(s).\n")

    report = bench_runner.run(experiments, options)

    bench_store.write_runinfo(
        {
            **({"host_label": args.host_label} if args.host_label else {}),
            "experiments": [e.id for e in experiments],
            "n_jobs": args.n_jobs,
            "timing_jobs": args.timing_jobs,
            "planned_runs": report.planned_runs,
            "executed_runs": report.executed_runs,
            "skipped_runs": report.skipped_runs,
            "failed_runs": report.failed_runs,
            "wall_seconds": round(report.wall_seconds, 1),
        },
        root=args.out,
    )
    normalizers = bench_store.compute_normalizers(args.out)

    print(f"\nDone in {_fmt_duration(report.wall_seconds)}.")
    print(f"  executed : {report.executed_runs}")
    print(f"  skipped  : {report.skipped_runs} (already in the store)")
    print(f"  errored  : {report.failed_runs}")
    if report.failed_runs:
        print("  Errored runs are kept in metrics.parquet with completed=False "
              "and an 'error' message. They are counted, not dropped.")
    print(f"  store    : {bench_store.results_dir(args.out)}")
    if normalizers:
        print("  nominal IAE (the normalization denominators):")
        for bench_key, info in normalizers.items():
            print(f"    {bench_key:<5} {info['iae_tracked']:.4g} "
                  f"[{info['unit']}] over {info['n_runs']} run(s)")
    nxt = "reproduce_all.py --smoke" if args.smoke else "reproduce_all.py"
    print(f"\nNext:  python examples/Benchmarks/{nxt}")
    return 0


def report_planned(experiments, options) -> int:
    """Count the runs a sweep would execute, for the pre-run banner."""
    planned = bench_runner.plan(experiments, options)
    return sum(len(g) for groups in planned.values() for g in groups)


if __name__ == "__main__":
    raise SystemExit(main())
