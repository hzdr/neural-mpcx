# Changelog

All notable changes to NeuralMPCX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
