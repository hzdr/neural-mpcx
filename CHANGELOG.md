# Changelog

All notable changes to NeuralMPCX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
