![NeuralMPCX Logo](fig/NeuralMPCX_LOGO_banner.png)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![DOI](https://rodare.hzdr.de/badge/1180898465.svg)](https://rodare.hzdr.de/badge/latestdoi/1180898465)

--------------------------------------------------------------------------------
NeuralMPCX is a Python library for building and deploying Model Predictive Controllers with classic and neural dynamical models. You write constrained MPC with RNN/LSTM models in a CasADi/IPOPT workflow. The library handles CasADi RNN integration, warm-starting, constraint management, real-time feasibility, and both LTI state-space and neural dynamics. You can run neural and classical MPC controllers side by side.

---

## Features

- [ ]  **Neural MPC** (RNN dynamic models)
- [ ]  Recurrent Neural Networks for system dynamics modelling
- [ ]  Classic MPC with CasADi-based optimization
- [ ]  Constraint handling (state, input, terminal, soft constraints)
- [ ]  Warm-starting & real-time iteration
- [ ]  Differentiable cost terms & custom regularization
- [ ]  Simulation utilities + logging

---

## Installation

> **Note:** NeuralMPCX is not yet available on PyPI. Install it locally from the downloaded repository.

### 1. Download or clone the repository

```bash
git clone --branch v3.1.0 https://github.com/hzdr/neural-mpcx.git
cd neural-mpcx
```

### 2. Install the package

**Basic install (core dependencies only):**

```bash
pip install -e .
```

**With PyTorch support** (for neural network training and deployment):

```bash
pip install -e .[torch]
```

This installs:
```
torch >= 2.0.0
torchvision >= 0.15.1
torchaudio >= 2.0.1
```

**For development** (testing, linting, type checking):

```bash
pip install -e .[dev]
```

### Dependencies

**Core Dependencies** (installed automatically):

```
numpy >= 1.26.4
casadi >= 3.6.6
joblib >= 1.4.2
gymnasium >= 0.29.1
scipy >= 1.10.0
matplotlib >= 3.5.0
pandas >= 1.5.0
```

### Manual PyTorch Installation

If you prefer to install PyTorch separately (e.g., to choose a specific CUDA version):

**CPU only:**
```bash
pip install torch>=2.0 torchvision>=0.15 torchaudio>=2.0 --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.4, recommended for recent GPUs):**
```bash
pip install torch>=2.0 torchvision>=0.15 torchaudio>=2.0 --index-url https://download.pytorch.org/whl/cu124
```

> **WSL2 users:** GPU support works out of the box. Install the NVIDIA driver on **Windows** only (not inside WSL), then use the CUDA command above.

**Supported Python versions:**

- Python >= 3.9
- CasADi >= 3.6.6

Tested on Python 3.9, 3.10, 3.11, and 3.12.

---

## Getting Started

### Neural MPC

See [`examples/Cascaded_Two_Tank_System/neural_mpc_cts.py`](examples/Cascaded_Two_Tank_System/neural_mpc_cts.py) for a Neural MPC deployment that adapts the controller from [**[1]**](#1), tested on the Cascaded Two-Tank System (CTS) benchmark. The CTS has two states (tank levels $h_1$ and $h_2$), a single pump-voltage input $u$, nonlinear $\sqrt{h}$ outflow dynamics, and a sampling period $T_s = 4$ s. A single-layer LSTM (hidden size 128) with a linear projection models the dynamics; the MPC uses prediction horizon $N = 10$ and context length $n_c = 10$, tracking the $h_2$ setpoint. [**[2]**](#2) describes the CTS in detail, and [**[3]**](#3) hosts the LSTM training and test datasets.

### Classic Linear MPC

See [`examples/MPC_Grinding_Circuit/mpc_grinding_circuit.py`](examples/MPC_Grinding_Circuit/mpc_grinding_circuit.py) for a classic MPC deployment. It reproduces an adapted version of the controller from [**[4]**](#4): a constrained MPC for the 4x4 grinding circuit ($S_p,D_m,F_c,L_s$ as controlled outputs; $F_f,F_m,F_d,V_p$ as manipulated inputs).

The plant model is a discrete-time LTI state-space system ($A_d, B_d, C_d, D_d$), generated from the paper’s transfer-function matrix $G(s)$ with Pade dead-time approximations and ZOH discretization.

The original paper uses a step-response DMC. Here, the same problem is formulated as an NLP over a state-space model using CasADi/IPOPT (multi-shooting), with unified soft constraints via slacks and a large penalty $w$.

The paper’s 16 transfer functions $`G_{ij}(s)`$ are assembled into a single state-space model offline and discretized. The MPC uses this discrete SS model directly.

### Classic Nonlinear MPC

See [`examples/CSTR/nmpc_cstr.py`](examples/CSTR/nmpc_cstr.py) for a Nonlinear MPC (NMPC) controller on the Continuous Stirred Tank Reactor (CSTR) benchmark. Based on the do-mpc CSTR benchmark from [**[6]**](#6) (Fiedler et al., 2023), the reactor has four states — concentrations $C_A$, $C_B$ and temperatures $T_R$ (reactor) and $T_K$ (jacket) — and two inputs, the feed flow rate $F_F$ and the heat-removal rate $\dot{Q}$. It runs two parallel reactions ($A \to B$ and $B \to C$) and one side reaction ($2A \to D$) with nonlinear Arrhenius kinetics.

The plant model uses symbolic CasADi expressions with Arrhenius kinetics and 4th-order Runge-Kutta (RK4) integration, so the optimizer gets exact first-order derivatives. The NLP is formulated with CasADi/IPOPT (multi-shooting), soft state constraints via slack variables, and a quadratic stage and terminal cost.

The example also demonstrates output-feedback NMPC: only the reactor and jacket temperatures $T_R$, $T_K$ are measured online (as in a real plant, where concentrations require lab analysis), and an `ExtendedKalmanFilter` reconstructs the full state — including the unmeasured concentrations $C_A$, $C_B$ — from the noisy temperature measurements before each MPC solve.

A Neural MPC version of the same process is available at [`examples/CSTR/neural_mpc_cstr.py`](examples/CSTR/neural_mpc_cstr.py), where a trained LSTM replaces the explicit dynamics. You can compare physics-based NMPC against data-driven Neural MPC on the same benchmark.

### State Estimation with Kalman Filters

NeuralMPCX provides Kalman filter implementations for state and bias estimation in MPC applications:

```python
from neuralmpcx.util.estimators import AugmentedKalmanFilter
from neuralmpcx.util.control import mimo_tf2ss
import numpy as np

# Create state-space model from transfer functions
ss = mimo_tf2ss(G, ny=4, nu=4, Ts=30.0)

# Create augmented Kalman filter for bias estimation
kf = AugmentedKalmanFilter(
    Ad=ss.Ad, Bd=ss.Bd, Cd=ss.Cd, Dd=ss.Dd,
    Q_x=np.eye(ss.nx) * 0.1,   # Process noise for states
    Q_du=np.eye(ss.nu) * 0.01, # Process noise for input bias
    Q_dy=np.eye(ss.ny) * 0.01, # Process noise for output bias
    R=np.eye(ss.ny) * 1.0,     # Measurement noise
)

# In MPC loop
for t in range(T):
    kf.predict(u=dev_u)
    kf.update(y=y_measured - y_offset)

    # Pass bias estimates directly to MPC
    u_opt = mpc.solve_mpc(state=x_est, state_indices=state_indices,
                          dynamic_pars=kf.get_mpc_biases())
```

The `AugmentedKalmanFilter` estimates plant state, input bias, and output bias at the same time, so you get offset-free MPC tracking even with plant-model mismatch.

For nonlinear plants, the `ExtendedKalmanFilter` estimates the full state from partial, noisy measurements. It takes the discrete-time dynamics as a CasADi function — for example, the exact prediction model already registered on the MPC — and derives the Jacobians automatically via CasADi algorithmic differentiation (no finite differences):

```python
from neuralmpcx.util.estimators import ExtendedKalmanFilter
import numpy as np

# Only some outputs are measured online: y = C @ x
C = np.array([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

ekf = ExtendedKalmanFilter(
    f=mpc.dynamics,      # the same CasADi function used inside the NLP
    h=C,                 # or a casadi.Function h(x) -> y for nonlinear sensors
    Q=np.eye(4) * 1e-6,  # process noise
    R=np.eye(2) * 1e-5,  # measurement noise
    x0=x0_guess,
    P0=np.eye(4) * 0.05,
)

# In MPC loop
for t in range(T):
    u_opt = mpc.solve_mpc(state=ekf.x_est, state_indices=state_indices, ...)  # feed the estimate, not the true state
    y_meas = plant.measure()
    ekf.predict(u=u_opt)
    ekf.update(y=y_meas)
```

See [`examples/CSTR/nmpc_cstr.py`](examples/CSTR/nmpc_cstr.py) for a complete output-feedback NMPC deployment where the EKF reconstructs unmeasured reactor concentrations from temperature measurements alone.

### Offset-free nonlinear MPC: `AugmentedExtendedKalmanFilter`

`AugmentedExtendedKalmanFilter` is the EKF plus the `AugmentedKalmanFilter`'s biases: it estimates the plant state of a *nonlinear* model together with random-walk input and output biases, giving offset-free tracking under plant-model mismatch. The augmented Jacobians come from CasADi AD, so the input bias enters the dynamics exactly — `x⁺ = f(x, u + S_du·δu)`, `y = h(x) + S_dy·δy`.

```python
from neuralmpcx.util.estimators import AugmentedExtendedKalmanFilter

aekf = AugmentedExtendedKalmanFilter(
    f=mpc.dynamics, h=C,
    du_index=[0],        # actuator offset on the first MV
    dy_index=[0],        # sensor/model bias on the first measured output
    Q_x=np.eye(4) * 1e-6, Q_du=np.eye(1) * 1e-4, Q_dy=np.eye(1) * 1e-4,
    R=np.eye(2) * 1e-5,
)
for t in range(T):
    u_opt = mpc.solve_mpc(state=aekf.x_est, state_indices=state_indices,
                          dynamic_pars=aekf.get_mpc_biases(), ...)
    y_meas = plant.measure()
    aekf.predict(u=u_opt)
    aekf.update(y=y_meas)
```

Two things to know before using it, both enforced by the class:

**Bias channels are a budget.** With `n_bias` integrating bias states the augmented system is detectable only if `rank([[I − A, −B_d], [C, C_d]]) == nx + n_bias`, and that matrix has just `nx + ny` rows — so **at most `ny` bias states can ever be estimated**, where `ny` counts *measured* channels. Asking for a bias on every input *and* every output is undetectable for any plant with an input: the filter still returns numbers, but the δu/δy split is decided by the ratio of `Q_du` to `Q_dy` rather than by the data. The constructor runs the rank test and refuses; `detectability_report()` returns the rank, the deficiency and the cap, and `check_detectability=False` overrides. The default is output bias on every channel and no input bias — `n_bias == ny` exactly, the textbook offset-free choice.

**The bias only absorbs what `Q_x` leaves it.** The state and the bias compete for the same innovation. Leave `Q_x` wide and the state chases the measurement on its own, the innovation collapses and the bias never moves — the tuning mistake that makes an AEKF look like it does nothing. Tightening `Q_x` forces the state to obey the model and hands the steady-state mismatch to the bias, which is what it is for.

The linear `AugmentedKalmanFilter` estimates the same quantities the same way — `A_aug` carries the `Bd @ S_du` cross-block, so its `x⁺ = Ad·x + Bd·(u + S_du·δu)` is this filter's statement written for a linear model, and both take the same `du_index`/`dy_index` under the same budget. (Before the current release it did not: `A_aug`'s cross-block was zero, so `du_bias` reached the model only through the feedthrough `Dd` and did nothing at all when `Dd == 0`.)

**Detectability is tested incrementally.** `bias_detectability(A, B, C, du_index, dy_index)` is public, and both augmented filters delegate to it. It asks whether the bias states *add* `n_bias` to the rank of the plant's own `[[I − A], [C]]`, rather than whether the rank reaches `nx + n_bias`. Where the plant is detectable those are the same statement. Where it is not, they differ usefully: a model carrying integrators that no measurement observes only earns a warning (that is equally true of the plain Kalman filter on it), while a bias the data genuinely cannot separate still raises — and the offending channels are named. The case worth knowing is an **output bias on a channel the model already integrates**: bias and integrator are both free and feed the same output, so their split follows `Q_dy` versus `Q_x` rather than the plant, and an integrating output is offset-free without one anyway.

### Constrained estimation: `MovingHorizonEstimator`

`MovingHorizonEstimator` estimates the same quantities as `AugmentedExtendedKalmanFilter` — plant state plus selected input and output biases — but instead of one linear correction per step it re-solves a small least-squares problem over the last `horizon` measurements. The drop-in surface is identical; only `horizon` is new.

```python
from neuralmpcx.util.estimators import MovingHorizonEstimator

mhe = MovingHorizonEstimator(
    f=mpc.dynamics, h=C,
    horizon=10,
    du_index=[0], dy_index=[0],
    Q_x=np.eye(4) * 1e-6, Q_du=np.eye(1) * 1e-4, Q_dy=np.eye(1) * 1e-4,
    R=np.eye(2) * 1e-5,
    x_lb=[0.0, 0.0, -np.inf, -np.inf],   # concentrations cannot go negative
    du_bias_ub=[0.15],                   # the actuator offset is physically capped
)
for t in range(T):
    u_opt = mpc.solve_mpc(state=mhe.x_est, state_indices=state_indices,
                          dynamic_pars=mhe.get_mpc_biases(), ...)
    y_meas = plant.measure()
    mhe.predict(u=u_opt)                 # order is mandatory here, not conventional
    mhe.update(y=y_meas)
```

**Reach for it when you have bounds worth enforcing.** That is the whole argument. A Kalman filter's correction is an unconstrained linear update, so it will report a negative concentration or an actuator offset larger than the valve can produce; here `x_lb`/`x_ub` and the bias bounds constrain the estimate itself. The second benefit is that the nonlinear model is used as written across the window rather than linearized once per step.

**It is the same estimator when the bounds are slack.** On a linear model with the default `arrival_cost="ekf"` and no active bound, it reproduces `AugmentedExtendedKalmanFilter` *exactly*, for any horizon — the test suite asserts this to `1e-6` at `horizon ∈ {1, 3, 8}`. So the horizon buys constraint handling and robustness to nonlinearity, not consistency, and there is nothing to gain from a long window if you have neither.

**It costs an NLP per control cycle.** Budget roughly two orders of magnitude more model evaluation than a filter, with second derivatives, landing in the same cycle as the MPC solve. `last_solve_time_s` reports it. Tuning transfers unchanged from the AEKF, with one exception: the covariances must be positive *definite*, because the cost weights by their inverse — `Q_du=np.zeros((1,1))` is legal on the filter and rejected here.

`retune()` changes the weights in place without rebuilding the problem or discarding the window; `reset()` does discard it.

## RNN-Based Dynamics in MPC

NeuralMPCX uses recurrent neural networks as the dynamics model inside the MPC. This required three adaptations:

- Converting PyTorch RNNs into CasADi
- Providing rolling context arrays of past actions and states (`action_context` and `state_context`, each `n_context` timesteps long) that feed the LSTM warmup
- Persisting the LSTM hidden/cell states across MPC solves

Only LSTM networks are supported so far.

NeuralMPCX adopts the **process-control symbol convention** throughout: $u$ is the input, while $x$ and $y$ are the states and measured outputs. A neural model maps inputs $u$ to outputs $y$ directly, so the toolkit treats it as **$y = x$** — the LSTM's (projected) hidden output *is* the predicted state.

An RNN's initial state, often set to zero, shapes its first predictions. In MPC, this initial state determines how the network reads the system's dynamics and how well it predicts future states. NeuralMPCX adapts the context-window warm-up of Forgione et al. [**[5]**](#5) to the per-step setting of an online MPC: instead of resetting the recurrent state to zero, it estimates the LSTM hidden/cell states from a context window of the last $n_c$ measured input–output pairs by teacher forcing (see [The warmup algorithm](#the-warmup-algorithm) below). The model thereby acts as its own state estimator, so the prediction starts from a state consistent with the measured past.

Three indices locate any quantity in the closed loop:

- the superscript $(k)$ is the **receding-horizon step** — one plant sample and one `solve_mpc()` call;
- the subscript $i$ is the **column along the free horizon** $N$;
- the second superscript $j$ is the **IPOPT interior-point iteration**, running from the warm-started initial guess $j = 0$ to the converged iterate $j = n_s$.

Combined, they name one quantity: $`U^{(k,j)}_{:,\,i}`$ is the input at prediction time $i$ of the candidate trajectory IPOPT tests at iteration $j$ of step $k$ (the colon spans all input channels).

At deployment, NeuralMPCX makes the LSTM **stateful**: its hidden and cell states $(h_0^{(k)}, c_0^{(k)})$ persist on the `Mpc` instance across `solve_mpc()` calls and enter the optimization problem as plain NLP parameters `h0`/`c0`. The dynamics function $\hat{X} = F(U, h_0, c_0)$ unrolls the LSTM symbolically over the prediction horizon: the rollout is **free-running** — only the control columns drive the recursion, the hidden state feeds back on itself, and no measurement enters the symbolic graph. No state estimation happens inside the optimizer, which keeps the NLP graph small and the solve fast. (Past outputs are not an input to the rollout; they only seed `h0`/`c0` through the warmup below.)

The warmed-up states carry only the step index $k$, never $j$: a separate numeric step estimates $(h_0^{(k)}, c_0^{(k)})$ once, before the solve begins, and they stay **fixed across every IPOPT iteration** $j$. The solver re-evaluates $F$ and its derivatives at each $j$ as the decision variables change, but each evaluation restarts the LSTM from the same $(h_0^{(k)}, c_0^{(k)})$.

Because the persisted $(h_0^{(k)}, c_0^{(k)})$ already encode the current measurement, the neural NLP needs **no anchor column**: it spans exactly `N = prediction_horizon` columns, and every column $`X^{(k,j)}_{:,\,i}`$ is a genuine prediction rolled forward from `h0`/`c0`. The multi-shooting dynamics constraint is $`F(U^{(k,j)}, h_0^{(k)}, c_0^{(k)})_{:,\,0:} = X^{(k,j)}_{:,\,0:}`$ (every column is a prediction). The most recent measurement and applied action survive only as the cost parameters `x0`/`u0` — used for terms like $\Delta u$ and initial/terminal penalties — and are **not** inputs to `F`. On convergence at $j = n_s$, `solve_mpc()` applies the first free input $`u^{(k)} = U^{(k,n_s)}_{:,\,0}`$ as the receding-horizon action (`solution.vals["u"][:, 0]`), and $`X^{(k,n_s)}_{:,\,0}`$ is the first predicted future state. (Classic MPC instead keeps the usual anchor column `x[:, 0] == x0` and spans `N + 1` columns.)

### The warmup algorithm

The persisted $(h_0^{(k)}, c_0^{(k)})$ are maintained numerically, outside the NLP, using **teacher forcing** — exactly mirroring how the network was trained:

1. **Teacher-forced context step.** For each context sample $`(u^{(k)}_{:,\,i}, y^{(k)}_{:,\,i})`$, the first-layer hidden state is *overwritten by the measured state* (a well-defined substitution since $y = x$ and the projected hidden state matches the output dimension), the action is fed as input, and the cell states and any deeper-layer hidden states propagate untouched. Each measurement therefore corrects the memory directly while the unmeasurable part is carried forward.

2. **Hybrid warmup phase** (the first $n_w$ solves, $k < n_w$ where $n_w$ is `n_warmup`). Each `solve_mpc()` call re-runs the teacher-forced pass over the *full* $n_c$-step window of measured pairs $`\{(u^{(k)}_{:,\,i}, y^{(k)}_{:,\,i})\}_{i=-n_c}^{-1}`$ via `estimate_numeric`, *seeded with the previous solve's* states (zeros at $k = 0$). This blends fresh measurements with accumulated memory while the buffers settle.

3. **Steady-state phase** (from $k = n_w$ onward). Each solve advances the stored states by exactly *one* teacher-forced step via `step_numeric`, using only the newest measured pair $`(u^{(k)}_{:,\,-1}, y^{(k)}_{:,\,-1})`$ — cutting the per-solve update from $\mathcal{O}(n_c)$ to $\mathcal{O}(1)$.

4. **Inside the NLP.** The resulting $(h_0^{(k)}, c_0^{(k)})$ are passed as the parameter values of `h0`/`c0`, and `F` rolls the LSTM forward symbolically over the prediction horizon $N$ from them (fixed across all IPOPT iterations $j$).

5. **Recovery.** `mpc.reset_lstm_state()` zeros the buffers and the solve counter, forcing a fresh multi-step warm-up from the rolling context arrays — use it if you suspect the persisted state has drifted from the plant. The `state_context`/`action_context` arrays act as the recovery truth data, so keep updating them every step.

In pseudocode:

```
on each solve_mpc(state_context, action_context, ...):       # step k
    if k < n_warmup:                                          # n_w
        (h, c) = estimate_numeric(u_ctx, y_ctx, seed=(h, c) or zeros)  # full n_c window
    else:
        (h, c) = step_numeric(u_ctx[-1], y_ctx[-1], h, c)              # one step
    solve NLP with parameters h0 = h, c0 = c                  # fixed across IPOPT iters j
    k += 1
```

Setting this up takes three steps:

```python
from neuralmpcx.neural import CasadiLSTM

model = CasadiLSTM(
    n_context=10, n_inputs=1, hidden_size=128,
    horizon=10, proj_size=1,
)
model.load_state_dict(torch.load("model.pt"))

mpc.set_neural_dynamics(model=model, n_warmup=1)

# mpc.is_warmed_up        -> True once n_warmup solves have run
# mpc.reset_lstm_state()  -> force re-warmup from the context window
```

### Measured disturbances (feedforward)

A *measured disturbance* (feedforward variable) is an exogenous input you can measure but not manipulate. NeuralMPCX supports it in **both** the conventional and neural paths: declare it with `mpc.disturbance(name, size)` and it becomes a `(size, prediction_horizon)` NLP **parameter** that feeds the prediction model. At solve time `solve_mpc()` **holds the latest measured disturbance constant** across the prediction horizon by default (the industrial feedforward behavior). How you supply that measurement differs by mode: neural MPC takes a rolling `disturbance_context` window (it also feeds the LSTM warmup), while conventional MPC takes a single `disturbance` (the latest measurement). In either case, pass an explicit forecast via `dynamic_pars={<name>: (size, N)}` to override the hold-constant default.

**Neural MPC.** If you trained the LSTM on the disturbance channel `d` (input columns ordered `[u, d]`), pass `n_disturbances` to the model and `allow_disturbances=True` to `set_neural_dynamics`; the rollout then consumes `d` like the controls (`d` is a parameter, not a decision variable). Because the LSTM is stateful, `d` must also enter the numeric warmup, so here `disturbance_context` is a rolling window `(n_context, nd)` mirroring `state_context`/`action_context` (its last `n_context` rows seed `h0`/`c0`).

```python
model = CasadiLSTM(
    n_context=10, n_inputs=nu, n_disturbances=nd,   # core input is [u, d]
    hidden_size=128, horizon=10, proj_size=ny,
)
model.load_state_dict(torch.load("model.pt"))

mpc.disturbance("d", size=nd)                        # (nd, prediction_horizon) parameter
mpc.set_neural_dynamics(model=model, allow_disturbances=True, n_warmup=1)

# hold-constant default (no dynamic_pars["d"] needed)
u_opt = mpc.solve_mpc(state_context=state_context, state_indices=state_indices,
                      action_context=action_context, setpoint=setpoint,
                      disturbance_context=d_ctx)   # d_ctx: (n_context, nd)
```

**Conventional MPC.** The disturbance is wired into the step dynamics `F(x_k, u_k, d_k)`. There is no warmup, so you just pass the latest measurement as `disturbance`. It is optional: you may instead supply the full trajectory yourself via `dynamic_pars[<name>]`.

```python
mpc.disturbance("d", size=nd)                        # (nd, prediction_horizon) parameter
mpc.set_dynamics(F)                                  # F(x, u, d) -> x_next

# hold the latest measured disturbance constant over the horizon
u_opt = mpc.solve_mpc(state=state, state_indices=state_indices,
                      setpoint=setpoint, disturbance=d_meas)  # d_meas: (nd,)
```

## Training an LSTM for Neural MPC

The neural dynamics model is an LSTM identified **offline** from measured
input–output data — the training counterpart to [RNN-Based Dynamics in MPC](#rnn-based-dynamics-in-mpc)
above. A worked, end-to-end recipe for the Cascaded Two-Tank System ships with
the repository:

- [`examples/Cascaded_Two_Tank_System/lstm_training.py`](examples/Cascaded_Two_Tank_System/lstm_training.py)
  — a runnable script covering the full workflow (dataset → windowing → training → evaluation);
- [`examples/Cascaded_Two_Tank_System/lstm_training.ipynb`](examples/Cascaded_Two_Tank_System/lstm_training.ipynb)
  — its step-by-step notebook companion, which explains each stage and reproduces the same model.

Two ideas make the resulting model usable inside a constrained MPC, and both are
trained in so they carry over to deployment unchanged.

**Context-aware initial state.** A plain RNN starts every rollout from a zeroed
$(h_0, c_0)$ and pays a long transient before its predictions are trustworthy —
unaffordable at every MPC solve. Instead, the model *estimates* $(h_0, c_0)$ from
the last $n_c$ measured input–output pairs by teacher forcing (the layer-0 hidden
state is overwritten by the measured output at each context step, a well-defined
substitution since $y = x$), and the loss is applied only **after** that warmup.
This is exactly the context-window estimation the controller reproduces
numerically before each solve (see [The warmup algorithm](#the-warmup-algorithm)),
so the trained network acts as its own state estimator at deployment.

**Predictions over the horizon $N$.** The long identification sequence is cut into
short overlapping windows; each window splits at $n_c$ into a teacher-forced
**context** region and a free-running **prediction** region. The loss scores the
multi-step open-loop error on the prediction region — precisely the quantity the
MPC rolls forward over its horizon $N$ — with a smaller state-consistency term on
the warmup region keeping the estimated $(h_0, c_0)$ meaningful (Forgione et al.
[**[5]**](#5)). In `compute_loss` the composite objective is, schematically:

$$
\mathcal{L} = \underbrace{Q_{\mathrm{mse}}\,\mathrm{MSE}\!\left(\hat{y}_{n_c:},\, y_{n_c:}\right)}_{\text{free-running prediction}}
\;+\; \underbrace{\alpha\,\mathrm{MSE}\!\left(\hat{y}_{:n_c},\, y_{:n_c}\right)}_{\text{warmup state consistency}}.
$$

**Plugging the model in.** Train with the **same** `n_context`, `hidden_size` and
`proj_size` you later pass to `CasadiLSTM`; the saved `.pt` then loads straight
into the controller via `model.load_state_dict(...)` and
`mpc.set_neural_dynamics(...)` (the three-step snippet in [The warmup
algorithm](#the-warmup-algorithm)). If you train on a measured-disturbance
channel, order the model's input columns `[u, d]` so it matches the
[Measured disturbances](#measured-disturbances-feedforward) path.

## Project Structure

```
src/neuralmpcx/
  core/            # Cache, solutions, warmstart, debug
  multistart/      # Start point generation for warm-starting
  neural/          # LSTM/RNN integration with CasADi
  nlps/            # NLP building blocks (parameters, variables, constraints)
  util/            # Utilities: control, estimators, math, io
    control.py     # Transfer functions, state-space, LQR
    estimators.py  # Kalman filters for state estimation
  wrappers/        # MPC wrappers (Mpc)
  __init__.py      # __version__ here

examples/
```

### Reproducing the benchmark study

`examples/Benchmarks/` runs five experiments -- measurement-noise sensitivity,
initial-condition robustness, plant-model mismatch and disturbances, real-time
feasibility, and nominal closed-loop performance -- across all available cores,
and regenerates every figure and table from committed results:

```bash
# regenerate all figures and tables from the stored results (seconds)
python examples/Benchmarks/reproduce_all.py

# re-run the whole study (hours; --dry-run first to see the plan)
python examples/Benchmarks/run_experiments.py --all --n-jobs 10
```

Experiments are declared in `examples/Benchmarks/configs/*.json`. See
[examples/Benchmarks/README.md](examples/Benchmarks/README.md) for the design,
the common-random-numbers contract, and how failed runs are accounted for.

---

---

## Testing

```bash
pip install -e ".[dev]"
ruff check src tests
mypy src
pytest -q
```

---

## Development Workflow

### Code Formatting and Linting

**Black** formats code. **Ruff** lints it.

#### Format Code with Black

```bash
# Check what would be reformatted
black --check src tests

# Format all code
black src tests

# Format specific files
black src/neuralmpcx/neural/casadi_lstm.py
```

#### Lint Code with Ruff

```bash
# Check all linting issues
ruff check src tests

# Auto-fix safe issues
ruff check --fix src tests

# Show what can be fixed
ruff check --fix --show-fixes src tests
```

#### Type Checking with mypy

```bash
# Run type checker
mypy src
```

#### Combined Workflow

Run all checks before committing:

```bash
# 1. Format with black
black src tests

# 2. Auto-fix with ruff
ruff check --fix src tests

# 3. Check remaining issues
ruff check src tests

# 4. Run type checking
mypy src

# 5. Run tests
pytest -q
```

### Pre-commit Hooks

Pre-commit hooks run these checks on every commit:

```bash
pre-commit install
pre-commit run --all-files
```

### Docstring Style

All public APIs use **NumPy-style docstrings**. Example:

```python
def my_function(param1, param2):
    """Brief description of the function.

    Extended description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    type
        Description of return value.
    """
```

---

---

## Contributing

Contributions are welcome. Follow these guidelines:

- Use `pre-commit` hooks (ruff/black/mypy/end-of-file-fixer)
- Follow **NumPy-style docstrings** for all public APIs (see Development Workflow section)
- Follow Conventional Commits (`feat:`, `fix:`, `docs:`, etc.)
- Open issues with minimal reproducible examples
- Run all tests and linting checks before submitting PRs

```bash
pre-commit install
```

---

## Citation

For academic work, cite:

```
@software{neuralmpcx2026,  title        = {NeuralMPCX: A Model Predictive Control library that supports classic MPC and neural MPC with CasADi},  author       = {Lopes Júnior, Ênio and Reinecke, Sebastian Felix},  year         = {2026},  url          = {https://github.com/hzdr/neural-mpcx}}
```

---

## License

**Apache License 2.0**. See [LICENSE.txt](./LICENSE.txt).

Portions derive from **casadi-nlp** by Filippo Airaldi, under the **MIT License**. See [LICENSE-MIT](./LICENSE-MIT).
Original project: [https://github.com/FilippoAiraldi/casadi-nlp](https://github.com/FilippoAiraldi/casadi-nlp)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)

---

## Maintainers & Contact

- Ênio Lopes Júnior
- Sebastian Felix Reinecke
- Issues: https://github.com/hzdr/neural-mpcx/issues

---

## Changelog

See [CHANGELOG.md](./CHANGELOG.md).

### References

<a id="1">[1]</a>
Adhau, S., Gros, S. and Skogestad, S. (2024).
["Reinforcement learning based MPC with neural dynamical models"](https://doi.org/10.1016/j.ejcon.2024.101048).
*European Journal of Control*, 80(A), 101048.

<a id="2">[2]</a>
Schoukens, M. and Noël, J. P. (2017).
[Three Benchmarks Addressing Open Challenges in Nonlinear System Identification](https://doi.org/10.1016/j.ifacol.2017.08.071).
*20th World Congress of the International Federation of Automatic Control*, Toulouse, France, July 9–14, 2017, pp. 448–453. ([preprint](https://drive.google.com/file/d/1vhn7udGe_anebb2-Wl94gcdZXeK6yxsK/view?usp=share_link))

<a id="3">[3]</a>
Schoukens, M., Mattsson, P., Wigren, T. and Noël, J. P.
[Cascaded tanks benchmark combining soft and hard nonlinearities](https://doi.org/10.4121/12960104).
4TU.ResearchData, Dataset.

<a id="4">[4]</a>
Chen, X. S., Zhai, J. Y., Li, S. H. and Li, Q. (2007).
["Application of model predictive control in ball mill grinding circuit"](https://doi.org/10.1016/j.mineng.2007.04.007).
*Minerals Engineering*, 20(11), 1099–1108.

<a id="5">[5]</a>
Forgione, M., Muni, A., Piga, D. and Gallieri, M. (2023).
["On the adaptation of recurrent neural networks for system identification"](https://doi.org/10.1016/j.automatica.2023.111092).
*Automatica*, 155, 111092.

<a id="6">[6]</a>
Fiedler, F., Karg, B., Lüken, L., Brandner, D., Heinlein, M., Brabender, F. and Lucia, S. (2023).
["do-mpc: Towards FAIR nonlinear and robust model predictive control"](https://doi.org/10.1016/j.conengprac.2023.105676).
*Control Engineering Practice*, 140, 105676.