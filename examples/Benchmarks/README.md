# Benchmark suite

Five experiments over the CSTR and cascaded-two-tank neural-MPC benchmarks. The
runs execute in parallel into a Parquet store, and every figure and table in the
study is rebuilt from that store without simulating anything.

## What it produces

`reproduce_all.py` writes 25 figures to `figures/` (PDF and SVG) and 10 tables to
`tables/` (CSV and a LaTeX fragment).

| figure | what it shows |
|--------|---------------|
| `exp1_fan_{cstr,cts}` | closed loop banded by noise level, with the input below |
| `exp1_offset_dots_{cstr,cts}` | steady-state offset and tracking cost per noise level |
| `exp2_fan_{cstr,cts}` | 50 initial conditions, colored by distance from nominal |
| `exp3_fan_{cstr,cts}` | 40 mismatched plants, colored by drift from the model |
| `exp3_disturbance_{CA0,Tin}_cstr` | response to an unmeasured step, per magnitude |
| `exp3_disturbance_leak2_cts` | the same for the two-tank leak |
| `exp3_offset_dots_{cstr,cts}` | signed residual offset across the mismatch sample |
| `exp3_envelope_iae_{cstr,cts}` | normalized IAE over the two plant parameters |
| `exp3_envelope_offset_{cstr,cts}` | signed offset over the same two parameters |
| `exp3_neural_vs_nmpc_cstr` | neural against physics MPC under identical mismatch |
| `exp4_rtf_wcet_{cstr,cts}` | worst-case real-time factor against horizon |
| `exp5_closed_loop_{cstr,cts}` | nominal closed-loop performance |
| `combined_fan_{cstr,cts}` | one benchmark's four closed-loop views, lettered (a)-(d) |
| `combined_envelope` | both benchmarks' mismatch maps, lettered (a)-(d) |

| table | what it holds |
|-------|---------------|
| `T0_robustness_at_a_glance` | one row per experiment and benchmark |
| `T1_measurement_noise` | one row per benchmark and noise level |
| `T2_initial_condition_summary` | initial-condition robustness per benchmark |
| `T2_worst_initial_conditions` | the five hardest starts, with their initial states |
| `T3_mismatch_and_disturbance_cases` | every mismatch and disturbance case |
| `T3_tolerance_summary` | how far each plant parameter may drift |
| `T4_rtf_wcet_grid` | worst-case real-time factor, hidden size by horizon |
| `T4_real_time_feasibility` | the full solve-time distribution |
| `T5_normalizers` | the absolute IAE behind every normalized number |
| `T6_solver_failures` | where the solver failed, and how badly |

## Verify the pipeline in two minutes

Before committing a cluster to the full sweep, run the smoke-scale copy. It uses
the same code paths and the same cases over tiny designs and short simulations:

```bash
python examples/Benchmarks/run_experiments.py --all --smoke   # 64 runs, ~75 s on 8 cores
python examples/Benchmarks/reproduce_all.py --smoke           # 25 figures + 10 tables, ~20 s
```

It reads `configs_smoke/` and writes to `results_smoke/`, `figures_smoke/` and
`tables_smoke/`, leaving the real store untouched. A smoke run produces no
results: its simulations are 10-30 steps, so almost nothing reaches the setpoint
band and the success rates are low by construction. What it establishes is that
every code path runs, every figure draws and every table builds.

## Reproducing the figures and tables

```bash
python examples/Benchmarks/reproduce_all.py
```

This reads the committed `results/` and simulates nothing, so a fresh clone
rebuilds all 25 figures and 10 tables in seconds.

Two other modes report on the store without drawing: `--check` gives its
coverage per experiment, and `--failures` gives the severity of every solver
failure and the factor levels that separate failure from success.

## Re-running the simulations

```bash
python examples/Benchmarks/run_experiments.py --all
```

790 runs, about 6.2 core-hours. Measured on a 20-core workstation at the nominal
configuration:

| benchmark | steps/run | solve time | wall time per run |
|-----------|-----------|-----------|-------------------|
| CSTR | 60 | ~225 ms | ~13.5 s |
| two-tank | 1050 | ~38 ms | ~40 s |

Wall clock is roughly `6.2 h / n_jobs`, plus about 37 min for the serial timing
experiment. Each worker holds ~800 MB resident, so pick `--n-jobs` from memory
and not from core count: `min(cores, memory_GB / 1)`. `--dry-run` prints the
design sizes and an estimate before anything runs.

`--resume` is on by default and is content-addressed, so an interrupted sweep
restarted against the same `--out` picks up where it stopped. It dedupes
*sequential* invocations only. Two nodes started at once against one directory
would both plan the same outstanding work and duplicate it; their shards will not
clobber each other, but the effort is wasted. To split across nodes, give each
one its own `--out` and its own `--only`/`--config-dir` slice, then consolidate.

Other flags: `--only exp1 exp4`, `--limit 20` (calibration), `--force`
(re-simulate), `--prune` (see below), `--batch-size` (how often a crash-safe
shard is written), `--host-label` (name this machine in the provenance record).

## The five experiments

| id | what varies | what it answers |
|----|-------------|-----------------|
| `exp1` | noise σ ∈ {0, 0.1, 0.5, 1, 2, 5} % of range, 20 seeded replicates | how much measurement noise the recurrent state tolerates |
| `exp2` | initial condition: a 50-point Latin hypercube, ±15 % around nominal (±50 % for the two-tank) | whether the controller works from a *region*, not one point |
| `exp3` | plant mismatch (a narrow 40-point LHS and a 150-point LHS over the wider envelope) and unmeasured step disturbances | how far the plant may drift before retraining, and how the loop rejects a load change |
| `exp4` | LSTM hidden size × prediction horizon, plus the physics NMPC | whether the controller fits inside its control period |
| `exp5` | nothing, the nominal reference | reproduces the closed-loop figure, and its IAE normalizes the whole study |

Edit `configs/*.json` to change any of this. An unknown parameter name there
raises at expansion time and lists the valid fields.

## Repository layout

```
run_experiments.py   expands configs/*.json into runs, executes them with joblib,
                     writes results/ as Parquet
reproduce_all.py     reads results/, writes figures/ and tables/
bench/
  config.py          benchmark descriptors, run specs, JSON expansion
  seeds.py           the common-random-numbers contract and the LHS designs
  adapters.py        bridges a run spec to the example scripts' simulate()
  runner.py          joblib driver: batching, resume, timing-critical serialization
  metrics.py         IAE, total variation, settling, offset, recovery, violations, RTF
  store.py           the Parquet store
  plotstyle.py       the publication style, from the analysis notebooks
  figures/, tables/  one module per experiment, plus figures/combined.py for
                     the figures that span several
```

The simulations stay in the example scripts (`examples/CSTR/neural_mpc_cstr.py`,
`examples/Cascaded_Two_Tank_System/neural_mpc_cts.py`,
`examples/CSTR/nmpc_cstr.py`). Each exposes a `RunConfig` dataclass and a
`simulate(cfg)`; running a script directly with no arguments still reproduces
what it did before this suite existed.

## Known pitfalls

**Editing a config does not retract the runs it replaces.** The store is
append-only and a `run_id` digests the whole spec, so narrowing a swept range
stops referring to the old runs without removing them. They stay in
`metrics.parquet`, joined to their case, and the figures keep drawing them,
because a figure plots the values *in the store* and not the ones in the JSON.
`--force` re-simulates the new points beside the old ones. Run `--prune` after
any config edit that changes a swept value, a sampled range or a case name:

```bash
python examples/Benchmarks/run_experiments.py --only exp3 --prune --dry-run
python examples/Benchmarks/run_experiments.py --only exp3 --prune
```

It is scoped to the experiments in the invocation, so pruning one leaves the
others alone, and `--dry-run` reports without writing.

**Failure detection.** `solve_mpc` neither raises nor returns a status, and under
the default warm start `_last_solution` is stale on failure: it still holds the
previous successful solve. Failures therefore come from the delta of the
cumulative `mpc.failures` counter, and the harvested prediction is written as NaN
on a failed step so a stale repeat cannot contaminate an NRMSE.

**Two-tank checkpoint scales.** The `cts-lstm-batched-*-repro` checkpoints were
trained on min-max scaled I/O, the others on raw meters and volts. Feeding a
normalized model raw meters produces a dead control loop and no error, so
`NORMALIZED_CHECKPOINTS` in `neural_mpc_cts.py` records which is which and
applies the unit boundary automatically. The `-repro` family also has no input
gain at an empty tank, because the data it was identified from never goes below
2.16 m. The study therefore uses `cts-lstm-batched-128`, and the hidden-size
sweep covers {32, 64, 128}; 8 and 16 fail every solve from this scenario's
empty-tank start.

**Timing runs are serialized.** `exp4` is flagged `timing_critical`, so it runs at
`n_jobs=1` after everything else. Ten workers contending for ten cores would
measure contention between them. `--timing-jobs N` overrides it.

**A store can be assembled on more than one machine.** `--check` prints one line
per sweep, so run it before quoting a solve time across experiments:

```bash
python examples/Benchmarks/reproduce_all.py --check
```

`RUNINFO.json` appends one record per sweep and never rewrites an older one.

```bash
python examples/Benchmarks/run_experiments.py --all --host-label "cluster node"
```

**Normalized IAE.** Every IAE is divided by the nominal (`exp5`) run of its own
benchmark, so `1.0` means "as good as the reported figure" and both benchmarks
share one axis. The absolute values live in `results/normalizers.json` and in
`tables/T5_normalizers.csv`; they belong in the captions.

**No aggregate drops a failed run.** `completed = no failed solve and the tracked
variable inside the ±50 % band at the end of the run`. Three band widths appear
in this suite and they are not interchangeable: `reached_band` and so `completed`
use ±50 % (`metrics.reached_band`, `band_frac=0.5`), `settling_steps` uses ±25 %,
and `recovery_steps` uses ±5 %. Every table reports
`N_total` beside `N_completed`, every median is taken over all runs with
non-finite values ranked worst, and every figure marks a failed run with an X in
a distinct color.

## Limitations

- **No offset-free estimator on the neural path.**
  `AugmentedExtendedKalmanFilter` exists in the library but is not wired to a
  neural-dynamics MPC; doing so is new controller machinery, not a refactor.
  `exp3_offset_dots` therefore ships as the plain signed dot plot.
- **`exp3_neural_vs_nmpc` covers the CSTR only.** There is no `nmpc_cts.py`, so
  the two-tank benchmark has no physics-MPC baseline to pair against.
- The CSTR neural controller and the physics NMPC differ in one pre-existing
  respect besides the model: the NMPC bounds `F` below at 5 h⁻¹ (0.05 of its
  normalized range) against the neural controller's 0. The figure comparing them
  carries this in its caption.
