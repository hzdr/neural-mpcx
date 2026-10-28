# NeuralMPCX _(neuralmpcx)_

![NeuralMPCX Logo](fig/NeuralMPCX_LOGO_banner.png)

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![PyPI](https://img.shields.io/pypi/v/neuralmpcx)](https://pypi.org/project/neuralmpcx/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://rodare.hzdr.de/badge/1180898465.svg)](https://rodare.hzdr.de/badge/latestdoi/1180898465)

Model Predictive Control toolkit with neural MPC support (CasADi-friendly).

NeuralMPCX is a Python library for building and deploying Model Predictive Controllers with linear, nonlinear, and neural dynamics. The software interfaces CasADi and IPOPT to solve constrained optimal control problems with recurrent neural networks (RNN, LSTM) and state-space systems.

Note on Naming: The repository and directory use the name `neural-mpcx`. The package name in Python and package managers is `neuralmpcx`. Install with `pip install neuralmpcx` and import with `import neuralmpcx`.

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

Install the package directly from PyPI:

```bash
pip install neuralmpcx
```

To install with PyTorch support:

```bash
pip install "neuralmpcx[torch]"
```

To install from source for development:

```bash
git clone https://github.com/hzdr/neural-mpcx.git
cd neural-mpcx
pip install -e ".[dev]"
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
from neuralmpcx import Nlp
import casadi as cs

# Define discrete-time LTI state-space system: x+ = Ad @ x + Bd @ u, y = Cd @ x
nx, nu, ny = 4, 2, 2
Ad = np.eye(nx) * 0.9
Bd = np.ones((nx, nu)) * 0.1
Cd = np.eye(ny, nx)
N = 10
M = 5

tuning_pars = {
    "Y_lb":   np.asarray([0.0, 0.0]),
    "Y_ub":   np.asarray([1.0, 1.0]),
    "Q": np.diag([1.0, 0.1]),
    "R": np.diag([0.2, 0.25]),
    "w": np.asarray(100.0),  # Positive penalty for output-bound violations
}

nlp = Nlp(sym_type="MX")
mpc = Mpc(nlp=nlp, prediction_horizon=N, control_horizon=M, tuning_parameters=tuning_pars)
x, _ = mpc.state("x", size=nx)
u, _, _ = mpc.action("u", size=nu, lb=-1.0, ub=1.0)

# Set parameters:

# Setpoint
SP = mpc.parameter("SP", (ny, 1))  # The parameter name must be "SP"; solve_mpc() expects it internally.

# Tuning parameters - Constraints, gain matrices, weights etc.
Y_lb = mpc.parameter("Y_lb", (ny, 1))
Y_ub = mpc.parameter("Y_ub", (ny, 1))
Q = mpc.parameter("Q", (ny, ny))
R = mpc.parameter("R", (nu, nu))
w = mpc.parameter("w", (1, 1))

# Register linear dynamics
def lti_dynamics(nx, nu, Ad, Bd):
    x_cs = cs.MX.sym("x", nx)
    u_cs = cs.MX.sym("u", nu)
    Ad_cs = cs.DM(Ad)
    Bd_cs = cs.DM(Bd)
    return cs.Function("F", [x_cs, u_cs], [Ad_cs @ x_cs + Bd_cs @ u_cs])

mpc.set_dynamics(lti_dynamics(nx, nu, Ad, Bd))

# Predicted outputs: exclude the initial-state column from the N + 1 states
Y = cs.DM(Cd) @ x[:, 1:]

# Add slack variables
s1, _, _ = mpc.variable("s1", (ny, N - 1), lb=0)
s2, _, _ = mpc.variable("s2", (ny, 1), lb=0)

# Soft output bounds, including the terminal prediction
mpc.constraint("Y_lb", cs.repmat(Y_lb, 1, N - 1) - s1, "<=", Y[:, :-1])
mpc.constraint("Y_ub", Y[:, :-1], "<=", cs.repmat(Y_ub, 1, N - 1) + s1)
mpc.constraint("Y_lb_f", Y_lb - s2, "<=", Y[:, -1])
mpc.constraint("Y_ub_f", Y[:, -1], "<=", Y_ub + s2)

# Set objective function
Q_cost = sum(
    cs.bilin(Q, Y[:, t] - SP) + w * cs.sum1(s1[:, t])
    for t in range(N - 1)
)

Q_term = cs.bilin(Q, Y[:, -1] - SP) + w * cs.sum1(s2)

R_cost = sum(
    cs.bilin(R, u[:, t])
    for t in range(M)
)

mpc.minimize(Q_cost + R_cost + Q_term)

# Initialize solver
mpc.init_solver()

# Solve optimal control problem
x0 = np.array([0.5, -0.2, 0.1, 0.0])
u_opt = mpc.solve_mpc(state=x0, state_indices=list(range(nx)), setpoint=[1.0, 1.0])
```

See [examples/MPC_Grinding_Circuit/mpc_grinding_circuit.py](examples/MPC_Grinding_Circuit/mpc_grinding_circuit.py) for a 4x4 grinding circuit benchmark.

### Classic Nonlinear MPC

This example keeps the linear MPC setup and replaces the dynamics with an illustrative nonlinear ODE, discretized with RK4 as in the CSTR implementation.

```python
import numpy as np
from neuralmpcx.wrappers import Mpc
from neuralmpcx import Nlp
import casadi as cs

# Define a nonlinear continuous-time model with output map y = Cd @ x
nx, nu, ny = 4, 2, 2
Cd = np.eye(ny, nx)
dt = 1.0  # Sampling interval in the ODE's time units
N = 10
M = 5

tuning_pars = {
    "Y_lb":   np.asarray([0.0, 0.0]),
    "Y_ub":   np.asarray([1.0, 1.0]),
    "Q": np.diag([1.0, 0.1]),
    "R": np.diag([0.2, 0.25]),
    "w": np.asarray(100.0),  # Positive penalty for output-bound violations
}

nlp = Nlp(sym_type="MX")
mpc = Mpc(nlp=nlp, prediction_horizon=N, control_horizon=M, tuning_parameters=tuning_pars)
x, _ = mpc.state("x", size=nx)
u, _, _ = mpc.action("u", size=nu, lb=-1.0, ub=1.0)

# Set parameters:

# Setpoint
SP = mpc.parameter("SP", (ny, 1))  # The parameter name must be "SP"; solve_mpc() expects it internally.

# Tuning parameters - Constraints, gain matrices, weights etc.
Y_lb = mpc.parameter("Y_lb", (ny, 1))
Y_ub = mpc.parameter("Y_ub", (ny, 1))
Q = mpc.parameter("Q", (ny, ny))
R = mpc.parameter("R", (nu, nu))
w = mpc.parameter("w", (1, 1))

# Discretize the nonlinear ODE with RK4; set_dynamics expects x+ = F(x, u)
def nonlinear_dynamics(nx, nu, dt):
    x_cs = cs.MX.sym("x", nx)
    u_cs = cs.MX.sym("u", nu)
    B = 0.1 * cs.DM.ones(nx, nu)

    def rhs(x, u):
        return -0.1 * x - 0.05 * x**3 + B @ u

    k1 = rhs(x_cs, u_cs)
    k2 = rhs(x_cs + 0.5 * dt * k1, u_cs)
    k3 = rhs(x_cs + 0.5 * dt * k2, u_cs)
    k4 = rhs(x_cs + dt * k3, u_cs)
    x_next = x_cs + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return cs.Function("F", [x_cs, u_cs], [x_next])

mpc.set_dynamics(nonlinear_dynamics(nx, nu, dt))

# Predicted outputs: exclude the initial-state column from the N + 1 states
Y = cs.DM(Cd) @ x[:, 1:]

# Add slack variables
s1, _, _ = mpc.variable("s1", (ny, N - 1), lb=0)
s2, _, _ = mpc.variable("s2", (ny, 1), lb=0)

# Soft output bounds, including the terminal prediction
mpc.constraint("Y_lb", cs.repmat(Y_lb, 1, N - 1) - s1, "<=", Y[:, :-1])
mpc.constraint("Y_ub", Y[:, :-1], "<=", cs.repmat(Y_ub, 1, N - 1) + s1)
mpc.constraint("Y_lb_f", Y_lb - s2, "<=", Y[:, -1])
mpc.constraint("Y_ub_f", Y[:, -1], "<=", Y_ub + s2)

# Set objective function
Q_cost = sum(
    cs.bilin(Q, Y[:, t] - SP) + w * cs.sum1(s1[:, t])
    for t in range(N - 1)
)

Q_term = cs.bilin(Q, Y[:, -1] - SP) + w * cs.sum1(s2)

R_cost = sum(
    cs.bilin(R, u[:, t])
    for t in range(M)
)

mpc.minimize(Q_cost + R_cost + Q_term)

# Initialize solver
mpc.init_solver()

# Solve optimal control problem
x0 = np.array([0.5, -0.2, 0.1, 0.0])
u_opt = mpc.solve_mpc(state=x0, state_indices=list(range(nx)), setpoint=[1.0, 1.0])
```

See [examples/CSTR/nmpc_cstr.py](examples/CSTR/nmpc_cstr.py) for output-feedback NMPC on a Continuous Stirred Tank Reactor.

### Neural MPC

This single-input, single-output example uses the saved two-tank model. Run it from the repository root; outputs are in metres and inputs in volts, matching the training data.

```python
import casadi as cs
import numpy as np
import torch

from neuralmpcx import Nlp
from neuralmpcx.neural import CasadiLSTM
from neuralmpcx.wrappers import Mpc

N, C = 10, 10  # Prediction horizon and measured context length
ny, nu = 1, 1  # Predicted outputs (tank 2 level) and control inputs

# Match the checkpoint's input, hidden, projection, and layer sizes
model = CasadiLSTM(
    n_context=C,
    n_inputs=nu,
    hidden_size=128,
    horizon=N,
    proj_size=ny,
    num_layers=1,
)
model_path = "examples/Cascaded_Two_Tank_System/models/cts-lstm-batched-128.pt"
model.load_state_dict(torch.load(model_path, map_location="cpu"))

mpc = Mpc(
    nlp=Nlp(sym_type="MX"),
    prediction_horizon=N,
    control_horizon=N,
    tuning_parameters={},
    n_context=C,
    neural=True,
)

# All N state columns are predictions; context stays outside the optimization
x, _ = mpc.state("x", size=ny)
u, u_exp, _ = mpc.action("u", size=nu, lb=0.0, ub=10.0)
SP = mpc.parameter("SP", (ny, 1))  # The parameter name must be "SP"; solve_mpc() expects it internally.

# Full C-sample warmup on the first solve; one-sample updates afterward
mpc.set_neural_dynamics(model=model, n_warmup=1)

mpc.minimize(sum(
    cs.sumsqr(x[:, k] - SP) + 1e-3 * cs.sumsqr(u_exp[:, k])
    for k in range(N)
))
mpc.init_solver()

# Startup fallback: repeat the initial measurement and previously applied input
y0 = np.array([0.0])
u_previous = np.array([0.0])
state_context = np.tile(y0, (C, 1))       # (C, ny), oldest to newest
action_context = np.tile(u_previous, (C, 1))  # (C, nu), oldest to newest
target = np.array([[5.0]])              # Column vector, matching measured outputs

u_opt = mpc.solve_mpc(
    state_context=state_context,
    action_context=action_context,
    state_indices=[0],  # The context contains only tank 2 level
    setpoint=target,
)
```

Use actual, aligned input/output histories when available. After applying the input `u_applied` and measuring the next output `y_next` from the plant, update both buffers before the next `solve_mpc()` call:

```python
state_context = np.vstack((state_context, np.asarray(y_next).reshape(1, ny)))[-C:]
action_context = np.vstack((action_context, np.asarray(u_applied).reshape(1, nu)))[-C:]
```

For a different model, match the checkpoint architecture, set `ny` to the model output size, and use the same `N` and `C` in the model and controller. If contexts contain additional measured outputs, `state_indices` selects the model outputs in their trained order; supply the setpoint as a column vector in the full measurement order. Keep the sampling interval and all scaling consistent with training.

See [neural_mpc_cts.py](examples/Cascaded_Two_Tank_System/neural_mpc_cts.py) for the full two-tank loop in plant units and [neural_mpc_cstr.py](examples/CSTR/neural_mpc_cstr.py) for a controller with two inputs and four outputs using normalized data.

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

`AugmentedKalmanFilter` estimates linear plant states and selected random-walk input/output biases. This small transfer-function model has a lagging output and an integrating output; only the first output receives an estimated bias, as in the grinding-circuit example.

```python
import numpy as np
from neuralmpcx.util.control import TransferFunctionTerm as TF, mimo_tf2ss
from neuralmpcx.util.estimators import AugmentedKalmanFilter

# Continuous transfer functions; keys are (output, input), starting at zero
G = {
    (0, 0): [TF(gain=1.0, time_constants=[5.0])],  # 1 / (5s + 1)
    (1, 0): [TF(gain=0.1, has_integrator=True)],   # 0.1 / s
    (1, 1): [TF(gain=-0.1, has_integrator=True)],  # -0.1 / s
}
ss = mimo_tf2ss(
    G, ny=2, nu=2, Ts=1.0,
    balanced=False,  # The model contains an integrator
    remove_unobservable_integrators=True,
)

du_index, dy_index = [], [0]  # No input bias; exclude the integrating output
n_aug = ss.nx + len(du_index) + len(dy_index)
kf = AugmentedKalmanFilter(
    Ad=ss.Ad, Bd=ss.Bd, Cd=ss.Cd, Dd=ss.Dd,
    du_index=du_index, dy_index=dy_index,
    Q_x=np.eye(ss.nx) * 1e-5,
    Q_dy=np.eye(len(dy_index)) * 1e-4,
    R=np.eye(ss.ny) * 1e-3,
    x0=np.zeros(ss.nx),
    dy_bias0=np.zeros(ss.ny),  # Bias seeds use the full signal width
    P0=np.eye(n_aug) * 0.1,
)

# Simulated measurements with a constant bias on output 0
rng = np.random.default_rng(0)
x_true = np.zeros((ss.nx, 1))  # Use the returned realization's state coordinates
u_applied = np.array([[0.2], [0.1]])
for _ in range(40):
    x_true = ss.Ad @ x_true + ss.Bd @ u_applied
    y_measured = ss.Cd @ x_true + ss.Dd @ u_applied + np.array([[0.1], [0.0]])
    y_measured += rng.normal(scale=np.sqrt(1e-3), size=(ss.ny, 1))
    kf.predict(u=u_applied)
    kf.update(y=y_measured, u=u_applied)  # Supply u when Dd is nonzero

state_estimate = kf.x_est  # (nx, 1), excluding bias states
biases = kf.get_mpc_biases()  # "du_bias": (nu, 1), "dy_bias": (ny, 1)
```

If you already have discrete matrices, pass them directly to the filter. `mimo_tf2ss` converts the continuous transfer functions and discretizes them at `Ts`. Removing redundant integrator states preserves the input/output response, but initial states and covariances must match the returned state coordinates.

For every augmented estimator, `Q_du` and `Q_dy` cover only the selected channels; omit the covariance for an empty selection. `P0` covers all `nx + len(du_index) + len(dy_index)` augmented states, while bias seeds and returned bias vectors use the full input/output widths, with unselected returned entries set to zero. At most `ny` bias states can be estimated, and the measurements must distinguish them from plant states; an output already modeled by an integrator can make an additional output bias unidentifiable. The constructors check the bias selection by default.

See [mpc_grinding_circuit.py](examples/MPC_Grinding_Circuit/mpc_grinding_circuit.py) for deviation signals, operating-point offsets, and the full output-feedback loop.

### Extended Kalman Filter

`ExtendedKalmanFilter` uses a discrete-time CasADi map `f(x, u) -> x_next` and a measurement map `h(x) -> y` or matrix `C`. Jacobians are generated automatically. The following two-state model measures only its first state; the augmented EKF and MHE snippets below reuse this model and its dimensions.

```python
import casadi as cs
import numpy as np
from neuralmpcx.util.estimators import ExtendedKalmanFilter

nx, nu, ny = 2, 1, 1
x_sym = cs.MX.sym("x", nx)
u_sym = cs.MX.sym("u", nu)
dt = 0.1
rhs = cs.vertcat(
    -x_sym[0] + x_sym[1] + u_sym[0],
    -2.0 * x_sym[1] - cs.sin(x_sym[0]),
)
f = cs.Function("f", [x_sym, u_sym], [x_sym + dt * rhs])  # Euler discretization
C = np.array([[1.0, 0.0]])

ekf = ExtendedKalmanFilter(
    f=f, h=C,
    Q=np.eye(nx) * 1e-6,
    R=np.eye(ny) * 1e-4,
    x0=np.zeros(nx),
    P0=np.eye(nx) * 0.1,
)

rng = np.random.default_rng(0)
x_true = np.array([[0.5], [-0.2]])
u_applied = np.array([[0.2]])
for _ in range(40):
    x_true = f(x_true, u_applied).full()
    y_measured = C @ x_true + rng.normal(scale=0.01, size=(ny, 1))
    ekf.predict(u=u_applied)
    ekf.update(y=y_measured)

state_estimate = ekf.x_est  # (nx, 1), including the unmeasured second state
```

For your own nonlinear MPC, reuse the same discrete dynamics as the controller (`f=mpc.dynamics` when it takes exactly two inputs, `x` and `u`). Discretize continuous-time equations before passing them to the estimator. `C` selects measured channels, which can differ from the outputs tracked by MPC; `Q` and `P0` are `(nx, nx)`, and `R` is `(ny, ny)` for those measurements.

See [nmpc_cstr.py](examples/CSTR/nmpc_cstr.py) for RK4 dynamics and reconstruction of unmeasured concentrations from temperature measurements.

### Augmented Extended Kalman Filter

`AugmentedExtendedKalmanFilter` adds selected random-walk biases to a nonlinear model as `f(x, u + du_bias)` and `h(x) + dy_bias`. Run the EKF snippet first to define `f`, `C`, `nx`, `nu`, and `ny`; this example estimates one output bias.

```python
import numpy as np
from neuralmpcx.util.estimators import AugmentedExtendedKalmanFilter

du_index, dy_index = [], [0]
n_aug = nx + len(du_index) + len(dy_index)
aekf = AugmentedExtendedKalmanFilter(
    f=f, h=C,
    du_index=du_index, dy_index=dy_index,
    Q_x=np.eye(nx) * 1e-6,
    Q_dy=np.eye(len(dy_index)) * 1e-5,
    R=np.eye(ny) * 1e-4,
    x0=np.zeros(nx),
    dy_bias0=np.zeros(ny),
    P0=np.eye(n_aug) * 0.1,
    u_lin=np.array([0.2]),  # Input operating point for the detectability check
)

rng = np.random.default_rng(0)
x_true = np.array([[0.5], [-0.2]])
u_applied = np.array([[0.2]])
for _ in range(40):
    x_true = f(x_true, u_applied).full()
    y_measured = C @ x_true + 0.1 + rng.normal(scale=0.01, size=(ny, 1))
    aekf.predict(u=u_applied)
    aekf.update(y=y_measured)

state_estimate = aekf.x_est
biases = aekf.get_mpc_biases()
```

To estimate an input bias instead, select its channel in `du_index`, set `dy_index=[]`, and replace `Q_dy` with `Q_du` sized for the selected inputs. Pass the applied input to `predict` without adding the estimated bias yourself. Nonlinear detectability checks use the linearization at `x0` and `u_lin`; they do not guarantee observability throughout a trajectory. Tune `Q_x` for state/model uncertainty and `Q_du`/`Q_dy` for bias drift, relative to measurement covariance `R`.

### Moving Horizon Estimator

`MovingHorizonEstimator` estimates states and selected biases over a rolling window with physical bounds. Reuse `f`, `C`, `nx`, `nu`, and `ny` from the EKF snippet. The constructor creates an IPOPT solver by default, and each `update` solves the estimation problem; startup measurements fill the window automatically.

```python
import numpy as np
from neuralmpcx.util.estimators import MovingHorizonEstimator

du_index, dy_index = [], [0]
n_aug = nx + len(du_index) + len(dy_index)
mhe = MovingHorizonEstimator(
    f=f, h=C,
    horizon=5,  # Past intervals; the window contains up to 6 state estimates
    du_index=du_index, dy_index=dy_index,
    Q_x=np.eye(nx) * 1e-6,
    Q_dy=np.eye(len(dy_index)) * 1e-5,
    R=np.eye(ny) * 1e-4,
    x0=np.zeros(nx),
    dy_bias0=np.zeros(ny),
    P0=np.eye(n_aug) * 0.1,
    x_lb=[-1.0, -1.0], x_ub=[1.0, 1.0],
    dy_bias_lb=[-0.2], dy_bias_ub=[0.2],  # Bounds use the full output width
    u_lin=np.array([0.2]),
    arrival_cost="ekf",
    on_solver_failure="raise",  # Surface failed solves while experimenting
)

rng = np.random.default_rng(0)
x_true = np.array([[0.5], [-0.2]])
u_applied = np.array([[0.2]])
for _ in range(40):
    x_true = f(x_true, u_applied).full()
    y_measured = C @ x_true + 0.1 + rng.normal(scale=0.01, size=(ny, 1))
    mhe.predict(u=u_applied)  # Required before each update: records the driving input
    mhe.update(y=y_measured)

state_estimate = mhe.x_est
biases = mhe.get_mpc_biases()
```

Supply positive-definite **covariance matrices**, rather than inverse covariance weights, for `Q_x`, the selected bias covariances, `R`, and `P0`. Keep states reasonably scaled. With `arrival_cost="ekf"`, an unconstrained linear example agrees with `AugmentedExtendedKalmanFilter` to solver tolerance. The default failure mode, `"fallback"`, returns the companion filter's unconstrained estimate, so use `"raise"` when silently losing the bounds is unacceptable; `mhe.last_status` and `mhe.n_solver_failures` expose solver outcomes.

For MPC feedback, pass the full `observer.x_est` as `state` and use `state_indices=list(range(observer.nx))` when the controller has the same state order. Augmented observers also return `observer.get_mpc_biases()` for `dynamic_pars`. To apply those biases, the controller must declare parameters named `"du_bias"` and `"dy_bias"` and apply them as `f(x, u + du_bias)` and `h(x) + dy_bias`; see the grinding-circuit example for the linear case. Map measured-output biases into the controller's output order if its outputs differ from the estimator's measurements.

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
  doi    = {10.14278/rodare.5114}
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