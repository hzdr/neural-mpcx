# Changelog

All notable changes to NeuralMPCX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [3.1.2] - 2026-08-31

### Added

- **`T1_measurement_noise`**, the noise sweep as its own table.
- **`T4_rtf_wcet_grid`**, the worst-case real-time factor as a hidden-size by
  horizon grid, both benchmarks side by side and the physics NMPC last. The
  15-column per-run table stays as `T4_real_time_feasibility`. 
- **`combined_fan_{cstr,cts}` and `combined_envelope`**, lettered 2x2 figures
  built in `bench/figures/combined.py`. 

### Changed

- **`exp1_fan` reads its noise level off a colorbar** instead of a six-entry
  legend, matching the `exp2` and `exp3` fans, and gains the median over all six
  levels as a black line. 
- **`T0` drops its `RTF_max` column.** 

### Fixed

- **`T5_normalizers.tex` did not compile.** 
- **Two table captions carried a raw `IAE_norm`.** 
- **`T0`'s caption and the README claimed a ±5 % success band.**
- **`RUNINFO.json` lost the provenance of every earlier sweep.** 
- **`.gitignore` was ignoring ten figures.**

## [3.1.1] - 2026-08-27

### Fixed

- **CasADi 3.8 broke the package on import.** 3.8 renamed
  `casadi/tools/structure3.py` to `structure.py`, so `import neuralmpcx` raised
  `ModuleNotFoundError` and took the whole test suite with it.
  `core/solutions.py` tries the old name first and falls back to the new one,
  leaving 3.6 and 3.7 on the module they already used.
- **`input_spacing > 1` crashed under CasADi 3.8.** `cs.GenDM_ones` no longer
  exists, so `Mpc.action` died inside `util.math.repeat`. That call now reads
  `cs.DM.ones`, which takes the same int or `(rows, cols)` tuple, returns the
  same shapes, and dates back to CasADi 3.0.
- `Mpc(..., tuning_parameters=None)` no longer raises `AttributeError` in
  `solve_mpc`. The constructor accepts `None`, then `solve_mpc` called
  `.update()` on it. `None` now reads as an empty dict.
- `Mpc.action` declares the three values it returns. The annotation promised
  two, while callers unpack `u, u_exp, u0`.

### Changed

- `mypy src` reports no issues across 27 modules. The old config pinned
  `python_version = "3.9"`, which mypy 2.3 rejects, so the check aborted before
  reading a line of source. The target is now `3.10`, and the config skips the
  CasADi and NumPy stubs: CasADi 3.8 ships a `casadi.pyi` mypy cannot parse, and
  the NumPy 2.5 stubs need a 3.12 target. The 52 errors this exposed spanned 15
  modules. Most were CasADi shape values that wanted an explicit `int` at the
  boundary.
- `WarmStartStrategy.generate` returns `Iterator`, matching the `chain` object
  it hands back. The old `Generator` annotation implied a `yield` the method
  does not have.
- `remove_variable_bounds` and `remove_constraints` spell their `idx` default as
  `Optional[...]`, per PEP 484.

## [3.1.0] - 2026-08-27

### Added
- `MultistartNlp`: An :class:`Nlp` solved from multiple starting points 
(serial best-of-N).solve the same problem from several initial guesses 
and keep the best solution.
- `AugmentedExtendedKalmanFilter`: joint state and input/output bias estimation
  for nonlinear models, giving offset-free NMPC under plant-model mismatch. The
  input bias enters the dynamics (`x⁺ = f(x, u + S_du·δu)`) rather than the
  feedthrough, and the augmented Jacobians come from CasADi AD.
- Bias channels are selected rather than assumed, under the detectability
  budget `n_bias ≤ ny`; the constructor runs the rank test and
  `detectability_report()` exposes it.
- `MovingHorizonEstimator`: constrained state and input/output bias estimation.
  Estimates the same quantities as `AugmentedExtendedKalmanFilter` over the same
  augmented model, but by minimizing an arrival cost plus weighted process and
  measurement residuals across a window of the last `horizon` measurements
  rather than by a single linear correction — so **bounds on the estimate are
  enforced** (`x_lb`/`x_ub`, `du_bias_lb`/`ub`, `dy_bias_lb`/`ub`), which is the
  reason to prefer it. The process and measurement noises are eliminated
  analytically, leaving a bound-constrained least-squares NLP with no equality
  constraints, which cannot go infeasible.
  - The default `arrival_cost="ekf"` anchors the window on a companion
    `AugmentedExtendedKalmanFilter`'s *one-step-predicted* estimate from the
    cycle the oldest slot belongs to. With that anchoring and no active bounds
    it reproduces the filter **exactly** on a linear model, at any horizon;
    tests assert this to `1e-6` for `horizon ∈ {1, 3, 8}`, through the
    window-fill phase included. `arrival_cost="constant"` is the heuristic
    alternative.
  - `Q_x`/`Q_du`/`Q_dy`/`R`/`P0` are **covariances**, as everywhere else in the
    module — not the inverse-covariance weights the moving-horizon literature
    (and do-mpc's `P_x`/`P_v`/`P_w`) uses. They must be positive *definite*
    rather than merely semi-definite, since the cost weights by their inverse:
    `Q_du=np.zeros((1,1))` is legal on the filters and rejected here.
  - The same `n_bias ≤ ny` detectability budget applies, delegated to the
    companion filter rather than reimplemented.
  - `retune()` rewrites the weights in place — the problem is not rebuilt and
    the window is kept — because the weights live in the NLP's parameter vector.
    `reset()` discards the window.
  - Diagnostics: `last_cost` (a better divergence alarm than the innovation,
    since mismatch is spread across the window and active bounds clip the
    residual), `last_status`, `last_solve_time_s`, `n_solver_failures`,
    `n_arrival_repairs`, `window_fill`, `z_traj`/`x_traj`.
- `AugmentedExtendedKalmanFilter`'s augmented model construction moved to a
  module-level `_augmented_model` shared with `MovingHorizonEstimator`. The
  expression graph is unchanged, so the filter's numbers are bit-identical.
- `tests/`: the test suite, 251 tests over the library and the benchmark
  harness, run with `pytest -q` from the repository root.
  - Tests needing PyTorch call `pytest.importorskip("torch")` and skip
    without it, as do those reading the `hidden_size=8` checkpoints under
    `examples/`, so a clone without them still runs the rest.

### Fixed

- **`AugmentedKalmanFilter`'s input bias now reaches the plant model.** Two
  defects, both found while building `AugmentedExtendedKalmanFilter` in the
  previous entry and deliberately left there because correcting them changes
  the numbers the shipped linear-SS path and the grinding-circuit example
  produce. **Behaviour-changing, not additive** — see the migration note below.
  - `A_aug` gains its `Bd @ S_du` cross-block, so the augmented prediction is
    `x⁺ = Ad·x + Bd·(u + S_du·δu)` — the linear form of the AEKF's
    `x⁺ = f(x, u + S_du·δu)`. It previously reached the model only through `Dd`
    in `C_aug`, so with `Dd = 0` it was a **completely dead state**: it
    random-walked under `Q_du`, corrected nothing, and was unobservable, while
    still being reported as an estimate. The shipped grinding template converts
    to `D = 0` (`max|D| == 0.0` over all 16 entries), so this was the shipped
    configuration, not a corner case.
  - Bias channels are now **selected** rather than assumed, through `du_index` /
    `dy_index` with the AEKF's defaults (no input bias, an output bias per
    output). The old unconditional `nu + ny` augmentation was provably
    undetectable for any plant with an input. `Q_du`/`Q_dy` are sized to the
    selection; `du_bias_est` / `dy_bias_est` stay **full width** (zero off the
    selection), so `get_mpc_biases()` remains a drop-in for existing consumers.
- `bias_detectability(A, B, C, du_index, dy_index)` is now public, and both
  augmented filters delegate their rank test to it so a selection cannot be accepted 
  at commissioning and then refused at build.
  - It tests the textbook condition **incrementally**: the bias states must
    *add* `n_bias` to the rank of the plant's own `[[I − A], [C]]`, rather than
    reach the absolute `nx + n_bias`. Where the plant is detectable the baseline
    *is* `nx` and the two are identical, so no AEKF result moves. Where it is
    not, the incremental form separates two failures the single number conflates:
    a **selection** that is unidentifiable (raises, and names the channels), and
    a **realization** carrying modes at `z = 1` that no measurement sees (warns
    — it is equally true of the plain Kalman filter on that model, so refusing
    only the augmented one would be incoherent).
  - The canonical unidentifiable selection is an output bias on a channel the
    model already integrates: bias and integrator are both free and feed the
    same output, so their split follows the drift covariances rather than the
    data — and an integrating output is offset-free without one. 
- `solve_mpc` rejects a non-2-D `action_context` with a named `ValueError`
  instead of dying on `IndexError: tuple index out of range` from the column
  arithmetic. The docstring had advertised a flat `(T_ctx,)` history for the
  single-input case, which never worked; it now states the 2-D requirement.

### Migration

- **A tuned `q_du` changes meaning.** It used to feed a dead state on any model
  with `D = 0`; it now moves the plant model. Re-check any value tuned against
  the old behaviour.
- `AugmentedKalmanFilter(Ad, Bd, Cd)` no longer augments every input. Pass
  `du_index` explicitly to keep an input bias, and size `Q_du`/`Q_dy` to the
  selection rather than to the full signal width.

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
