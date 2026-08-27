# Changelog

All notable changes to NeuralMPCX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [3.0.3] - 2026-08-27

### Added

- `examples/Benchmarks/`: a reproducible benchmark suite over the CSTR and
  cascaded-two-tank neural-MPC examples. Five experiments, declared as JSON
  tables, run in parallel into a Parquet store; a second script rebuilds every
  figure and table from that store without simulating.
  - `run_experiments.py` expands `configs/*.json` into run specs and executes
    them through joblib/loky. A `run_id` digests the whole spec, so `--resume`
    skips work already in the store and a re-run regenerates the same keys. The
    driver writes a shard as each batch completes, so an interruption costs at
    most the batch in flight. `--prune` retracts runs the configs no longer
    define, `--dry-run` sizes a sweep before it starts, and `--smoke` runs the
    reduced copies in `configs_smoke/` in about a minute.
  - `reproduce_all.py` reads `results/` and writes 22 figures and 8 tables into
    `figures/` and `tables/`. It simulates nothing, so a clone carrying the
    store reproduces the study's graphics in seconds. `--check` reports store
    coverage; `--failures` reports where the solver failed and which factor
    levels separate failure from success.
  - `bench/` holds the machinery: benchmark descriptors and JSON expansion
    (`config.py`), the seeding contract with the Latin-hypercube designs
    (`seeds.py`), the bridge to each example script's `simulate()`
    (`adapters.py`), the joblib driver (`runner.py`), closed-loop metrics
    (`metrics.py`), the Parquet store (`store.py`), the publication style
    (`plotstyle.py`), and one figure and one table module per experiment.
  - The experiments: measurement noise at six levels with 20 seeded replicates
    (exp1); initial conditions over a 50-point Latin hypercube per benchmark
    (exp2); plant-model mismatch and unmeasured step disturbances, with the
    neural-versus-physics comparison on the CSTR (exp3); solve-time scaling in
    LSTM hidden size and horizon against the control period (exp4); and the
    nominal closed-loop run whose IAE normalizes the study (exp5). 790 runs,
    about 6.2 core-hours on the reference machine.
  - Two properties carry the paired design. The noise seed depends on `(benchmark, replicate)` and nothing
    else, and the suite draws each design point once and freezes it into the
    spec. Failed solves come from the delta of `mpc.failures`, since `solve_mpc`
    neither raises nor returns a status and `_last_solution` still holds the
    previous successful solve after a failure.
  - No aggregate drops a failed run. Every table reports `N_total` beside
    `N_completed`, every median ranks non-finite values worst, and every figure
    marks a failed run with an X in a distinct color.
  - `exp4` carries `timing_critical` and runs at `n_jobs=1` after every other
    experiment, so its real-time factors measure the controller alone.
  - An unknown parameter name in a config raises at expansion time and lists the
    valid fields.

## [3.0.2] - 2026-08-04

### Fixed

- **Installation:** `pyproject.toml` restricted NumPy to the environment marker
  `python_version <= '3.9'`, even though the project requires `>=3.9`, so a
  clean install on Python 3.10-3.12 skipped NumPy entirely. The marker now
  splits into `< '3.10'` (keeping the `< 2.0.0` cap) and `>= '3.10'`.
- Tag-pinned clone instructions in the README, so users reproduce the released
  version rather than the moving `main` branch.

### Changed

- Added `Programming Language :: Python :: 3.9`-`3.12` classifiers to
  `pyproject.toml`, matching the Python versions the README reports as tested.
- `codemeta.json` `softwareRequirements` now lists `tqdm`, `matplotlib`, and
  `pandas`, matching the runtime dependencies declared in `pyproject.toml`.

## [3.0.1] - 2026-06-26
### Fixed
- Fixed headers on files that are legacy from csnlp library.
- Fixed headers on files that are majorly new with small pieces from csnlp library.
- Fixed mpc wrapper header with a map showing legacy vs new code. 71% of the file is new code. 29% is legacy.
- Fixed control.py header. It's fully new. no legacy code there.

## [3.0.0] - 2026-06-24

### Added

- Measured-disturbance (feedforward) support across neural and conventional MPC.
  Feed the current measurement into the prediction model and hold it constant over
  the horizon, or pass a forecast to override it. Conventional MPC wires `d` into
  `F(x_k, u_k, d_k)`; neural MPC adds `d` as an LSTM input channel and threads it
  through the warmup. New `disturbance_context` argument on `solve_mpc` and
  `n_disturbances` on `CasadiLSTM`.
- CSTR feedforward examples (`neural_mpc_cstr_disturbance.py`,
  `nmpc_cstr_disturbance.py`) that treat `Q_dot` as a measured disturbance fixed
  at -4250 kJ/h and optimize `F` alone.
- `lstm_training.py` and `lstm_training.ipynb`: a function-based LSTM training
  walkthrough for the CTS (load, window, build, train, evaluate) with
  colorblind-safe plots, a `RUN_TRAINING` toggle, and a `REPRO_MODEL_NAME` guard
  that writes `<model>-repro.pt` without overwriting the shipped model.
- Training diagnostics after Karpathy's "makemore part 3": per-epoch
  update-to-data ratio (target ~1e-3), pre-clip gradient norm, and LSTM
  cell-state saturation in `train_lstm`'s history, a `plot_diagnostics()` helper,
  an init-scale check, and the update:data ratio shown live in the progress bar.

### Changed

- **BREAKING:** The neural prediction model consumes controls only. The dynamics
  function is now `F(u, h0, c0[, d])` instead of `F(x, u, h0, c0)`; past outputs
  feed only the numeric warmup. `CasadiLSTM.forward` takes the control sequence
  directly.
- **BREAKING:** `solve_mpc` selects its inputs by mode. Neural mode uses
  `state_context`/`action_context` (and `disturbance_context` when declared) and
  ignores `state`; conventional mode uses the latest `state` plus the new
  `disturbance` and derives `u0` from the last applied action. The new
  `disturbance` argument is appended to the signature, so positional calls still
  parse.
- **BREAKING:** The neural NLP rolls every column from the persisted LSTM state.
  It spans `T = N` columns with no `x[:,:n_context] == x0` pinning; `x[:,0]` is
  the first predicted state and `u[:,0]` is the action applied now. `x0`/`u0`
  remain only as cost parameters.
- README notation aligned to the process-control convention (`u` input; `x`, `y`
  states/outputs; `y = x` for neural models), the free-running rollout written
  `X_hat = F(U, h0, c0)`, with matching symbols across the CSTR and CTS examples.
- Retuned the NMPC and neural MPC settings in the CSTR example for better tracking.

### Removed

- **BREAKING:** `input_order` parameter from `CasadiLSTM` and `set_neural_dynamics`.
- **BREAKING:** `remove_bounds_on_initial_action` and `input_bias_scope` kwargs
  from `set_neural_dynamics`.

### Fixed

- Single shooting works end to end on both paths: the neural single-shooting
  dynamics no longer raise `UnboundLocalError`, and the conventional CSTR example
  reads its trajectory through `solution.value()`.
- Best-state restore in the training loop keeps the best checkpoint through to the
  end of training.
- Module headers in `lstm_pytorch.py` and `metrics.py`.

## [2.1.0] - 2026-06-11

### Added

- Feature: discrete-time Extended Kalman Filter (EKF) for nonlinear systems to neuralmpcx.util.estimators.
- `control.py`: Implemented shared integrator per output row in `_assemble_mimo_ss`. Balance stable subsystem when integrators are present.

### Changed

- Refactored `nmpc_cstr.py` example. With `USE_EKF` enabled, an Extended Kalman Filter reconstructs the full state from the noisy temperature measurements and the MPC consumes the estimate.

- Refactored all examples plots layout to be better presented.

### Fixed

- `control.py`: Fixed pure-gain spurious state in `_assemble_mimo_sse`

## [2.0.0] - 2026-06-10

### Added

- Feature: stateful LSTM with persisted hidden state

### Changed

- Refactored `CasadiLSTM`, `_CasadiLSTMCore` and `Mpc`
- Refactored examples and added measurement noise simulation

## [1.1.0] - 2026-04-10

### Changed

- Replaced `scipy.signal.tf2ss` controllable canonical form with a Gramian-based balanced realization in `mimo_tf2ss`, reducing system matrix condition numbers by orders of magnitude
- Added `balanced=True` parameter to `mimo_tf2ss`; falls back to canonical form with `UserWarning` for systems with integrators or non-Hurwitz modes
- Narrowed broad exception handler in `io.py` and added security warning to `load()` docstring about pickle deserialization risk, following NumPy conventions

### Added

- NeuralMPCX logo and updated README badges

## [1.0.0] - 2026-03-20

### Added

- Initial stable release: full library with `CasadiLSTM`, `_CasadiLSTMCore`, `Mpc`, and supporting modules (`cache`, `data`, `debug`, `solutions`, `warmstart`, `constraints`, `nlp`, `objective`, `estimators`, `io`, `math`, `control`)
- Examples: CSTR neural MPC, Cascaded Two-Tank System, MPC Grinding Circuit
- RODARE integration via `.rodare.json` for automatic DOI registration

### Fixed

- Enabled NLP gradient scaling for the MUMPS solver to fix `L_s` setpoint tracking; KKT condition number was ~1e21 due to `L_s` transfer function gains (0.001–0.032) sitting 1000x below other outputs

## [0.1.0] - 2026-03-13

### Added

- Pre-release working version (predates public git history)
