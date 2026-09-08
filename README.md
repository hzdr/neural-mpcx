# NeuralMPCX _(neuralmpcx)_

![NeuralMPCX Logo](fig/NeuralMPCX_LOGO_banner.png)

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://rodare.hzdr.de/badge/1180898465.svg)](https://rodare.hzdr.de/badge/latestdoi/1180898465)

Model Predictive Control toolkit with neural MPC support (CasADi-friendly).

NeuralMPCX is a Python library for building and deploying Model Predictive Controllers with linear, nonlinear, and neural dynamics. The software interfaces CasADi and IPOPT to solve constrained optimal control problems with recurrent neural networks (RNN, LSTM) and state-space systems.

Note on Naming: The repository and directory use the name `neural-mpcx`. The package name in Python and package managers is `neuralmpcx`. Install with `pip install -e .` and import with `import neuralmpcx`.

## Table of Contents

- [Background](#background)
- [Install](#install)
  - [Dependencies](#dependencies)
  - [Manual PyTorch Installation](#manual-pytorch-installation)
- [Usage](#usage)
  - [Classic Linear MPC](#classic-linear-mpc)
  - [Classic Nonlinear MPC](#classic-nonlinear-mpc)
  - [Neural MPC](#neural-mpc)
- [Features](#features)
- [RNN-Based Dynamics in MPC](#rnn-based-dynamics-in-mpc)
  - [Warmup Algorithm](#warmup-algorithm)
  - [Measured Disturbances](#measured-disturbances)
  - [Training an LSTM for Neural MPC](#training-an-lstm-for-neural-mpc)
- [State Estimation](#state-estimation)
  - [Augmented Kalman Filter](#augmented-kalman-filter)
  - [Extended Kalman Filter](#extended-kalman-filter)
  - [Augmented Extended Kalman Filter](#augmented-extended-kalman-filter)
  - [Moving Horizon Estimator](#moving-horizon-estimator)
- [Benchmarks](#benchmarks)
- [Development](#development)
  - [Code Style and Checks](#code-style-and-checks)
  - [Pre-Commit Hooks](#pre-commit-hooks)
  - [Docstrings](#docstrings)
- [API](#api)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

Model Predictive Control (MPC) solves an open-loop optimal control problem at each sampling step using a model of the plant. First-principles nonlinear models yield accurate physical predictions, but developing differential-algebraic equations for complex industrial plants demands substantial engineering effort.

Recurrent neural networks offer black-box models identified from input-output time series. Standard neural network rollouts start from arbitrary initial states (often zero). In receding-horizon control, an uninitialized hidden state distorts short-horizon predictions. NeuralMPCX adapts the context-window warmup method from Forgione et al. (2023) [5] to online MPC: past input-output pairs seed the recurrent state before each optimization step.

The library builds on CasADi symbolic graphs and solves resulting nonlinear programs with IPOPT. Users formulate cost functions, input constraints, and state bounds in Python, while the solver evaluates algorithmic derivatives of both physical and neural dynamic models.

Academic references:
- [1] Adhau, S., Gros, S. and Skogestad, S. (2024). Reinforcement learning based MPC with neural dynamical models. *European Journal of Control*, 80(A), 101048. https://doi.org/10.1016/j.ejcon.2024.101048
- [2] Schoukens, M. and Noël, J. P. (2017). Three Benchmarks Addressing Open Challenges in Nonlinear System Identification. *IFAC-PapersOnLine*, 50(1), 448-453. https://doi.org/10.1016/j.ifacol.2017.08.071
- [3] Schoukens, M., Mattsson, P., Wigren, T. and Noël, J. P. Cascaded tanks benchmark combining soft and hard nonlinearities. 4TU.ResearchData. https://doi.org/10.4121/12960104
- [4] Chen, X. S., Zhai, J. Y., Li, S. H. and Li, Q. (2007). Application of model predictive control in ball mill grinding circuit. *Minerals Engineering*, 20(11), 1099-1108. https://doi.org/10.1016/j.mineng.2007.04.007
- [5] Forgione, M., Muni, A., Piga, D. and Gallieri, M. (2023). On the adaptation of recurrent neural networks for system identification. *Automatica*, 155, 111092. https://doi.org/10.1016/j.automatica.2023.111092
- [6] Fiedler, F., Karg, B., Lüken, L., Brandner, D., Heinlein, M., Brabender, F. and Lucia, S. (2023). do-mpc: Towards FAIR nonlinear and robust model predictive control. *Control Engineering Practice*, 140, 105676. https://doi.org/10.1016/j.conengprac.2023.105676

## Install

Clone the repository and install with pip:

```bash
git clone --branch v3.1.4 https://github.com/hzdr/neural-mpcx.git
cd neural-mpcx
pip install -e .
```

To install with PyTorch support:

```bash
pip install -e .[torch]
```

To install development tools:

```bash
pip install -e .[dev]
```

### Dependencies

Core dependencies:
- `numpy >= 1.26.4`
- `casadi >= 3.6.6`
- `joblib >= 1.4.2`
- `gymnasium >= 0.29.1`
- `scipy >= 1.10.0`
- `matplotlib >= 3.5.0`
- `pandas >= 1.5.0`

Python support:
- Python >= 3.9 (tested on 3.9, 3.10, 3.11, and 3.12)

### Manual PyTorch Installation

Install PyTorch wheels for CPU:

```bash
pip install torch>=2.0 torchvision>=0.15 torchaudio>=2.0 --index-url https://download.pytorch.org/whl/cpu
```

Install PyTorch wheels for CUDA 12.4:

```bash
pip install torch>=2.0 torchvision>=0.15 torchaudio>=2.0 --index-url https://download.pytorch.org/whl/cu124
```

WSL2 users install NVIDIA drivers on Windows. The Linux environment runs CUDA wheels without host driver changes.

## Usage

### Classic Linear MPC

```python
import numpy as np
from neuralmpcx.wrappers import Mpc

# Define discrete-time LTI state-space system: x+ = Ad @ x + Bd @ u, y = Cd @ x
nx, nu, ny = 4, 2, 2
Ad = np.eye(nx) * 0.9
Bd = np.ones((nx, nu)) * 0.1
Cd = np.eye(ny, nx)

mpc = Mpc(nlp=None, prediction_horizon=10, control_horizon=10)
mpc.state("x", size=nx)
mpc.control("u", size=nu)

# Set control bounds
mpc.bounds("u", lb=[-1.0, -1.0], ub=[1.0, 1.0])

# Register linear dynamics
def lti_dynamics(x, u):
    return Ad @ x + Bd @ u

mpc.set_dynamics(lti_dynamics)

# Solve optimal control problem
x0 = np.array([0.5, -0.2, 0.1, 0.0])
u_opt = mpc.solve_mpc(state=x0, state_indices=[0, 1, 2, 3])
```

See [examples/MPC_Grinding_Circuit/mpc_grinding_circuit.py](examples/MPC_Grinding_Circuit/mpc_grinding_circuit.py) for a 4x4 grinding circuit benchmark.

### Classic Nonlinear MPC

See [examples/CSTR/nmpc_cstr.py](examples/CSTR/nmpc_cstr.py) for output-feedback NMPC on a Continuous Stirred Tank Reactor.

### Neural MPC

```python
import torch
from neuralmpcx.wrappers import Mpc
from neuralmpcx.neural import CasadiLSTM

# Load trained PyTorch LSTM weights into CasadiLSTM
model = CasadiLSTM(
    n_context=10,
    n_inputs=1,
    hidden_size=128,
    horizon=10,
    proj_size=1,
)
model.load_state_dict(torch.load("model.pt"))

mpc = Mpc(nlp=None, prediction_horizon=10, control_horizon=10)
mpc.state("x", size=1)
mpc.control("u", size=1)
mpc.set_neural_dynamics(model=model, n_warmup=1)

# In the control loop, pass historical context buffers
u_opt = mpc.solve_mpc(
    state_context=state_context,
    action_context=action_context,
    state_indices=[0],
    setpoint=target_value,
)
```

See [examples/Cascaded_Two_Tank_System/neural_mpc_cts.py](examples/Cascaded_Two_Tank_System/neural_mpc_cts.py) for the Cascaded Two-Tank System benchmark.

## Features

- Neural MPC with CasADi-compiled recurrent models
- Recurrent neural networks for system identification and predictive control
- Classical linear and nonlinear MPC formulations with CasADi and IPOPT
- Constraint management (state bounds, control limits, terminal regions, soft slack variables)
- Warm-starting and real-time execution
- Differentiable objective terms and custom stage regularization
- Simulation and logging utilities

## RNN-Based Dynamics in MPC

NeuralMPCX integrates recurrent neural networks as dynamics models inside CasADi optimization graphs. The software converts PyTorch models into symbolic CasADi functions, maintains rolling context arrays (`action_context` and `state_context`), and preserves hidden states between control steps.

Process control symbol conventions define inputs as $u$ and measured outputs as $y$. The neural model predicts outputs from inputs; NeuralMPCX equates $y = x$, meaning the projected hidden state represents the predicted state.

Three indices track closed-loop quantities:
- The superscript $(k)$ denotes the receding-horizon step (one plant sample and one `solve_mpc()` invocation).
- The subscript $i$ denotes the column along prediction horizon $N$.
- The superscript $j$ denotes the IPOPT interior-point iteration, running from initial guess $j = 0$ to convergence $j = n_s$.

The symbol $U^{(k,j)}_{:,\,i}$ designates the input vector at prediction time $i$ evaluated at iteration $j$ of step $k$.

During execution, the `Mpc` instance stores hidden and cell states $(h_0^{(k)}, c_0^{(k)})$. Dynamics function $\hat{X} = F(U, h_0, c_0)$ rolls the network forward over horizon $N$. Control inputs drive the recurrence without measurement feedback inside the symbolic graph.

### Warmup Algorithm

Teacher forcing maintains the stored states $(h_0^{(k)}, c_0^{(k)})$ outside the NLP:

1. Context Step: For each context sample $(u^{(k)}_i, y^{(k)}_i)$, the measured state sets the first-layer hidden state ($y = x$). Actions feed the input, while cell states propagate through the network.
2. Hybrid Warmup Phase: For step counts $k < n_{\mathrm{warmup}}$, each solve evaluates the full $n_c$-step window of measured pairs through `estimate_numeric`, seeded with the prior recurrent state.
3. Steady-State Phase: For $k \ge n_{\mathrm{warmup}}$, each solve updates the recurrent state by one step through `step_numeric`, using the latest measured pair $(u^{(k)}_{-1}, y^{(k)}_{-1})$. This update requires $O(1)$ operations per step.
4. Optimization: The resulting state $(h_0^{(k)}, c_0^{(k)})$ sets parameters `h0` and `c0`. IPOPT rolls the LSTM model forward over prediction horizon $N$.
5. Recovery: `mpc.reset_lstm_state()` resets stored buffers and triggers a fresh multi-step warmup from `state_context` and `action_context`.

### Measured Disturbances

A measured disturbance (feedforward variable) is an observed, non-manipulated exogenous input. Register disturbances with `mpc.disturbance(name, size)`. NeuralMPCX allocates an NLP parameter of shape `(size, prediction_horizon)`. By default, `solve_mpc()` holds the latest measurement constant across the prediction horizon. To supply a planned trajectory, pass `dynamic_pars={<name>: array}`.

For neural MPC, configure the model with `n_disturbances=nd` and enable support with `allow_disturbances=True` in `set_neural_dynamics`. Pass historical values to `disturbance_context` as an array of shape `(n_context, nd)`.

### Training an LSTM for Neural MPC

Train the identification network on input-output sequences divided into overlapping windows. Split each window at index $n_c$ into a context region and a prediction region. The objective function penalizes multi-step prediction error across the prediction horizon, combined with a consistency penalty across the context window:

$$
\mathcal{L} = Q_{\mathrm{mse}}\,\mathrm{MSE}\!\left(\hat{y}_{n_c:},\, y_{n_c:}\right) + \alpha\,\mathrm{MSE}\!\left(\hat{y}_{:n_c},\, y_{:n_c}\right)
$$

Ensure the identification model uses identical values for `n_context`, `hidden_size`, and `proj_size` as `CasadiLSTM`. When using disturbances, order inputs as `[u, d]`. Complete training scripts reside in [examples/Cascaded_Two_Tank_System/lstm_training.py](examples/Cascaded_Two_Tank_System/lstm_training.py) and [examples/Cascaded_Two_Tank_System/lstm_training.ipynb](examples/Cascaded_Two_Tank_System/lstm_training.ipynb).

## State Estimation

NeuralMPCX includes observers for state and disturbance reconstruction.

### Augmented Kalman Filter

`AugmentedKalmanFilter` estimates plant states alongside input and output disturbances for linear systems:

```python
from neuralmpcx.util.estimators import AugmentedKalmanFilter
from neuralmpcx.util.control import mimo_tf2ss
import numpy as np

ss = mimo_tf2ss(G, ny=4, nu=4, Ts=30.0)
kf = AugmentedKalmanFilter(
    Ad=ss.Ad, Bd=ss.Bd, Cd=ss.Cd, Dd=ss.Dd,
    Q_x=np.eye(ss.nx) * 0.1,
    Q_du=np.eye(ss.nu) * 0.01,
    Q_dy=np.eye(ss.ny) * 0.01,
    R=np.eye(ss.ny) * 1.0,
)

for t in range(T):
    kf.predict(u=u_cmd)
    kf.update(y=y_measured)
    u_opt = mpc.solve_mpc(state=kf.x_est, state_indices=state_indices,
                          dynamic_pars=kf.get_mpc_biases())
```

### Extended Kalman Filter

`ExtendedKalmanFilter` reconstructs unmeasured states from partial nonlinear measurements using algorithmic differentiation for Jacobians:

```python
from neuralmpcx.util.estimators import ExtendedKalmanFilter
import numpy as np

C = np.array([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

ekf = ExtendedKalmanFilter(
    f=mpc.dynamics,
    h=C,
    Q=np.eye(4) * 1e-6,
    R=np.eye(2) * 1e-5,
    x0=x0_guess,
    P0=np.eye(4) * 0.05,
)

for t in range(T):
    u_opt = mpc.solve_mpc(state=ekf.x_est, state_indices=state_indices)
    ekf.predict(u=u_opt)
    ekf.update(y=plant.measure())
```

### Augmented Extended Kalman Filter

`AugmentedExtendedKalmanFilter` combines nonlinear state estimation with random-walk bias tracking on inputs (`du_index`) and outputs (`dy_index`).

The estimator splits innovation between state and bias states according to covariance matrices `Q_x`, `Q_du`, and `Q_dy`. Large values in `Q_x` cause state estimates to follow high-frequency measurement variations, preventing bias states from tracking steady-state errors. Small values in `Q_x` force state estimates to match the model, allowing bias states to absorb low-frequency mismatch.

Augmenting a system with $n_{\mathrm{bias}}$ disturbance states requires:

$$
\mathrm{rank}\begin{bmatrix} I - A & -B_d \\ C & C_d \end{bmatrix} = n_x + n_{\mathrm{bias}}
$$

Because this matrix contains $n_x + n_y$ rows, an observable model supports at most $n_y$ bias states, where $n_y$ is the count of measured channels. Assigning biases to all inputs and outputs creates an undetectable system. The default setup tracks output biases across all measured channels ($n_{\mathrm{bias}} = n_y$).

### Moving Horizon Estimator

`MovingHorizonEstimator` solves a constrained least-squares problem over the previous $N$ measurements using CasADi and IPOPT. Use this observer when states or biases must respect physical bounds:

```python
from neuralmpcx.util.estimators import MovingHorizonEstimator
import numpy as np

mhe = MovingHorizonEstimator(
    f=mpc.dynamics, h=C,
    horizon=10,
    du_index=[0], dy_index=[0],
    Q_x=np.eye(4) * 1e-6, Q_du=np.eye(1) * 1e-4, Q_dy=np.eye(1) * 1e-4,
    R=np.eye(2) * 1e-5,
    x_lb=[0.0, 0.0, -np.inf, -np.inf],
    du_bias_ub=[0.15],
)

for t in range(T):
    u_opt = mpc.solve_mpc(state=mhe.x_est, state_indices=state_indices,
                          dynamic_pars=mhe.get_mpc_biases())
    mhe.predict(u=u_opt)
    mhe.update(y=plant.measure())
```

On linear models without active constraints and with `arrival_cost="ekf"`, the estimator produces results identical to `AugmentedExtendedKalmanFilter`.

## Benchmarks

The `examples/Benchmarks/` directory provides five test suites: measurement noise sensitivity, initial condition robustness, model mismatch with disturbances, real-time feasibility, and nominal closed-loop performance.

Run all benchmarks:

```bash
python examples/Benchmarks/run_experiments.py --all --n-jobs 10
```

Regenerate figures and tables from stored test data:

```bash
python examples/Benchmarks/reproduce_all.py
```

Review [examples/Benchmarks/README.md](examples/Benchmarks/README.md) for configuration schemas.

## Development

### Code Style and Checks

NeuralMPCX uses Black for code formatting, Ruff for linting, and mypy for static type checks.

Format code:
```bash
black src tests
```

Lint code:
```bash
ruff check --fix src tests
ruff check src tests
```

Run type checks:
```bash
mypy src
```

Run unit tests:
```bash
pytest -q
```

Validate README specification compliance:
```bash
python tests/test_readme_spec.py
```

### Pre-Commit Hooks

Install pre-commit hooks to run style checks on each commit:

```bash
pre-commit install
pre-commit run --all-files
```

### Docstrings

Public functions and classes use NumPy-style docstrings with parameter types, descriptions, and return values.

## API

NeuralMPCX exposes public control and optimization interfaces under `neuralmpcx`:

- `neuralmpcx.wrappers.Mpc`: MPC controller interface. Manages state definitions, control horizons, cost functions, constraints, CasADi NLP construction, and solver calls.
- `neuralmpcx.neural.CasadiLSTM`: PyTorch LSTM model converter. Generates CasADi symbolic expressions for recurrent simulation and trajectory optimization.
- `neuralmpcx.util.estimators`:
  - `AugmentedKalmanFilter`: Linear state and disturbance estimator for offset-free tracking.
  - `ExtendedKalmanFilter`: Nonlinear observer using algorithmic differentiation for Jacobians.
  - `AugmentedExtendedKalmanFilter`: Nonlinear state estimator with input and output bias tracking.
  - `MovingHorizonEstimator`: Constrained receding-horizon state and bias estimator.
- `neuralmpcx.util.control`: Control system utilities including transfer function conversion (`mimo_tf2ss`), discretization, and state-space transformations.

Directory layout:

```
src/neuralmpcx/
  core/            # Cache, solution containers, warm-start utilities
  multistart/      # Initial point generation for non-convex NLPs
  neural/          # PyTorch and CasADi neural network integration
  nlps/            # NLP parameters, variables, and constraints
  util/            # Control math, state estimators, I/O helpers
  wrappers/        # User-facing Mpc wrapper
```

## Maintainers

- Ênio Lopes Júnior ([e.lopes-junior@hzdr.de](mailto:e.lopes-junior@hzdr.de))
- Sebastian Felix Reinecke ([s.reinecke@hzdr.de](mailto:s.reinecke@hzdr.de))

## Thanks

- Filippo Airaldi for [casadi-nlp](https://github.com/FilippoAiraldi/casadi-nlp), licensed under the MIT License. Portions of the NLP wrapper derive from this work.
- Helmholtz-Zentrum Dresden-Rossendorf (HZDR) for project support and infrastructure.

Cite this software in academic publications:

```bibtex
@software{neuralmpcx2026,
  title  = {NeuralMPCX: A Model Predictive Control library that supports classic MPC and neural MPC with CasADi},
  author = {Lopes J{\'u}nior, {\^E}nio and Reinecke, Sebastian Felix},
  year   = {2026},
  url    = {https://github.com/hzdr/neural-mpcx},
  doi    = {10.14278/rodare.4991}
}
```

## Contributing

Contributions are welcome. Submit pull requests on GitHub.

Submit bug reports and feature requests at https://github.com/hzdr/neural-mpcx/issues.

Contribution requirements:
- Code checks must pass (`black`, `ruff`, `mypy`, `pytest`).
- Use NumPy-style docstrings for public interfaces.
- Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).

Review [CONTRIBUTING.md](CONTRIBUTING.md) for licensing and workflow guidelines.

## License

Apache License 2.0. Copyright 2026 Helmholtz-Zentrum Dresden-Rossendorf e.V. (HZDR).

See [LICENSE](LICENSE) for terms.

Portions derive from casadi-nlp by Filippo Airaldi under the MIT License. See [LICENSE-MIT](LICENSE-MIT).