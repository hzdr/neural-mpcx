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

"""Regenerate every figure and table from the stored results.

Usage
-----
::

    python examples/Benchmarks/reproduce_all.py
    python examples/Benchmarks/reproduce_all.py --only exp2 exp4
    python examples/Benchmarks/reproduce_all.py --check

This simulates nothing. It reads the committed ``results/``, so a fresh clone
reproduces the study's graphics in seconds. ``run_experiments.py`` regenerates
the results themselves.

Figures are written as PDF and SVG, tables as CSV and LaTeX, following the
analysis notebooks.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bench import config as bench_config  # noqa: E402
from bench import plotstyle  # noqa: E402
from bench import store as bench_store  # noqa: E402
from bench.figures import exp1, exp2, exp3, exp4, exp5  # noqa: E402
from bench.tables import builders as table_builders  # noqa: E402


FIGURE_BUILDERS = {
    "exp1": exp1.build,
    "exp2": exp2.build,
    "exp3": exp3.build,
    "exp4": exp4.build,
    "exp5": exp5.build,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", nargs="+", metavar="EXP",
        help="restrict figures to these experiment ids (tables always cover all)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report what the store contains and exit without drawing anything",
    )
    parser.add_argument(
        "--failures", action="store_true",
        help="report where the solver failed and how badly, then exit. Gives "
             "the severity per case and, for the worst case, the factor levels "
             "that separate failure from success.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="read results_smoke/ and write to figures_smoke/ and tables_smoke/",
    )
    parser.add_argument("--results", type=Path, default=None, help="override results/")
    parser.add_argument("--figures", type=Path, default=None, help="override figures/")
    parser.add_argument("--tables", type=Path, default=None, help="override tables/")
    return parser


def _check(metrics, root) -> int:
    """Summarize store coverage, so a partial sweep is visible."""
    info = bench_store.load_runinfo(root)
    print(f"store            : {bench_store.results_dir(root)}")
    print(f"runs             : {len(metrics)}")
    if info:
        print(f"generated        : {info.get('generated_utc', '?')}")
        print(f"git sha          : {info.get('git_sha', '?')[:12]}")
        print(f"wall clock       : {info.get('wall_seconds', '?')} s")
    print()
    print(f"{'experiment':<10} {'benchmark':<22} {'runs':>6} {'completed':>10} "
          f"{'failed solves':>14} {'time series':>12}")
    print("-" * 80)
    for (exp, bench_key), group in metrics.groupby(["experiment", "benchmark"]):
        bench = bench_config.BENCHMARKS[bench_key]
        has_ts = bench_store.has_timeseries(exp, bench_key, root=root)
        failed = int(group["n_failed_solves"].clip(lower=0).sum())
        print(f"{exp:<10} {bench.label:<22} {len(group):>6} "
              f"{int(group['completed'].sum()):>10} {failed:>14} "
              f"{'yes' if has_ts else 'no':>12}")
    missing = sorted(set(FIGURE_BUILDERS) - set(metrics["experiment"].unique()))
    if missing:
        print(f"\nnot in the store yet: {', '.join(missing)}")
        print("run:  python examples/Benchmarks/run_experiments.py --all")
    return 0


SEVERITY_BUCKETS = [
    (0.0, 0.05, "<5%   transient"),
    (0.05, 0.25, "5-25%  intermittent"),
    (0.25, 0.50, "25-50% degraded"),
    (0.50, 0.90, "50-90% mostly infeasible"),
    (0.90, 1.01, ">90%   infeasible throughout"),
]


def _failures(metrics, root) -> int:
    """Triage report: where the solver failed, how badly, and driven by what.

    A success rate gives the same number to a run that lost two steps and a run
    infeasible from step nine. This separates them.
    """
    import numpy as np

    metrics = metrics.copy()
    metrics["_failed"] = metrics["n_failed_solves"].clip(lower=0)
    metrics["_frac"] = metrics["_failed"] / metrics["n_solves"].replace(0, np.nan)
    affected = metrics[metrics["_failed"] > 0]

    print(f"{metrics['_failed'].sum():,} failed solves across "
          f"{len(affected)} of {len(metrics)} runs "
          f"({metrics['n_solves'].sum():,} solves attempted).\n")
    if affected.empty:
        print("No solver failures anywhere in the store.")
        return 0

    print(f"{'experiment':<8} {'benchmark':<20} {'case':<26} "
          f"{'runs':>6} {'hit':>5} {'steps':>8} {'median severity':>16}")
    print("-" * 94)
    for (exp, bench_key, case), group in metrics.groupby(
        ["experiment", "benchmark", "case"]
    ):
        hit = group[group["_failed"] > 0]
        if hit.empty:
            continue
        label = bench_config.BENCHMARKS[bench_key].label
        print(f"{exp:<8} {label:<20} {case:<26} {len(group):>6} {len(hit):>5} "
              f"{int(hit['_failed'].sum()):>8} "
              f"{float(np.nanmedian(hit['_frac'])):>15.0%}")

    print("\nHow bad, per affected run:")
    frac = affected["_frac"].to_numpy(dtype=float)
    for lo, hi, name in SEVERITY_BUCKETS:
        n = int(((frac >= lo) & (frac < hi)).sum())
        bar = "#" * int(round(40 * n / max(1, len(frac))))
        print(f"  {name:<28} {n:>5}  {bar}")

    # Report only the factors that separate failure from success in the worst
    # case. A factor whose failure rate is flat across its levels explains
    # nothing, and listing it would suggest it does.
    worst = affected.groupby(["experiment", "benchmark", "case"])["_failed"].sum()
    exp, bench_key, case = worst.idxmax()
    scope = metrics[(metrics.experiment == exp) & (metrics.benchmark == bench_key)
                    & (metrics["case"] == case)]
    print(f"\nWorst case: {exp} / {bench_config.BENCHMARKS[bench_key].label} "
          f"/ {case}")
    candidates = [c for c in bench_config.SPEC_FIELDS
                  if c in scope.columns and scope[c].nunique() > 1]
    for col in candidates:
        rate = scope.groupby(col).apply(
            lambda g: (g["_failed"] > 0).mean(), include_groups=False
        )
        if rate.max() - rate.min() < 0.5:
            continue  # flat across its levels: explains nothing
        print(f"  driven by {col}:")
        for value, r in rate.items():
            if r > 0:
                # Sampled levels carry float noise (0.8200000000000001).
                # Print them as the values they stand for.
                shown = f"{value:g}" if isinstance(value, float) else str(value)
                print(f"    {col} = {shown:<8} -> {r:.0%} of runs fail")
        cut = rate[rate > 0].index.max()
        safe = rate[rate == 0].index.min() if (rate == 0).any() else None
        if safe is not None:
            print(f"    cliff between {cut:g} and {safe:g}; "
                  f"no failures at or beyond {safe:g}")

    try:
        ts = bench_store.load_timeseries(exp, bench_key, scope[scope._failed > 0]["run_id"],
                                         root=root)
        bad = ts[~ts["solve_ok"]]
        if not bad.empty:
            first = bad.groupby("run_id")["step"].min()
            print(f"  first failure at step {int(first.median())} of "
                  f"{int(scope['n_solves'].median())} (median). Compare with "
                  f"the severity above to see whether it recovered.")
    except FileNotFoundError:
        pass

    print("\nFull breakdown: tables/T6_solver_failures.csv")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.results = args.results or HERE / "results_smoke"
        args.figures = args.figures or HERE / "figures_smoke"
        args.tables = args.tables or HERE / "tables_smoke"
        print("SMOKE MODE: figures built from the reduced smoke run.\n")

    try:
        metrics = bench_store.load_metrics(args.results)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        return _check(metrics, args.results)

    if args.failures:
        return _failures(metrics, args.results)

    plotstyle.apply()
    normalizers = bench_store.compute_normalizers(args.results)
    metrics = bench_store.attach_normalized_iae(metrics, normalizers, args.results)
    metrics = bench_store.attach_normalized_tv(metrics)

    fig_dir = Path(args.figures or bench_config.FIGURES_DIR)
    tab_dir = Path(args.tables or bench_config.TABLES_DIR)

    wanted = set(args.only) if args.only else set(FIGURE_BUILDERS)
    written = []
    for exp_id, builder in FIGURE_BUILDERS.items():
        if exp_id not in wanted:
            continue
        paths = builder(metrics, fig_dir, root=args.results)
        written += paths
        n_fig = len(paths) // 2  # each figure is written as PDF and SVG
        status = f"{n_fig} figure(s)" if paths else "no data in the store"
        print(f"  {exp_id}: {status}")

    tables = table_builders.build_all(metrics, tab_dir, root=args.results)
    print(f"  tables: {len(tables) // 2} table(s)")

    print(f"\nFigures -> {fig_dir}")
    print(f"Tables  -> {tab_dir}")
    print(f"{len(written) // 2} figures and {len(tables) // 2} tables written.")

    if normalizers:
        print("\nAbsolute IAE behind every normalized number (for the captions):")
        for bench_key, info in normalizers.items():
            print(f"  {bench_config.BENCHMARKS[bench_key].label:<22} "
                  f"{info['iae_tracked']:.5g} [{info['unit']}]")

    incomplete = int((~metrics["completed"]).sum())
    if incomplete:
        print(f"\n{incomplete} of {len(metrics)} runs did not complete. They are "
              f"included in every aggregate and marked in every figure; see the "
              f"N_total / N_completed columns in tables/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
