# Changelog

All notable changes to NeuralMPCX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-13
## [1.0.0] - 2026-03-20
## [1.1.0] - 2026-04-10

### Added

- Initial release of NeuralMPCX
- Neural MPC with LSTM-based dynamic models and CasADi integration
- Classic linear MPC with state-space models
- Classic nonlinear MPC (NMPC) with symbolic CasADi dynamics
- Constraint handling (state, input, terminal, soft constraints)
- Warm-starting and multi-start support
- Augmented Kalman Filter for state and bias estimation
- MIMO transfer function to state-space conversion utilities
- Example: Cascaded Two-Tank System (neural MPC)
- Example: Grinding Circuit (linear MPC)
- Example: CSTR (nonlinear MPC and neural MPC)
