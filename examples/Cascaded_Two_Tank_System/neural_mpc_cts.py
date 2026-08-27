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

"""
Neural MPC benchmark for cascaded two-tank system using LSTM dynamics.

Benchmark script for performance evaluation of Neural MPC controller with LSTM model.
Reproduces the controller from:
"Reinforcement learning based MPC with neural dynamical models" by S. Adhau et al.
https://www.sciencedirect.com/science/article/pii/S0947358024001080

Initial hidden state estimation uses a context window of past observations, following:
"On the adaptation of recurrent neural networks for system identification" by M. Forgione et al.
https://www.sciencedirect.com/science/article/pii/S0005109823002510

This benchmark measures computation time statistics in a controlled environment
with single-threaded execution to minimize timing variance.
"""

from __future__ import annotations

import os

# --- BENCHMARK SETUP: NEUTRAL ENVIRONMENT ---
# Define environment variables before loading heavy numerical libraries
# to force single-threaded execution and reduce jitter.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import time  # measure computation time
import gc  # Garbage Collector control

import casadi as cs
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
import sys

from neuralmpcx import Nlp
from neuralmpcx.wrappers import Mpc
from neuralmpcx.neural import CasadiLSTM
from gymnasium.spaces import Box

# -----------------------------------------------------------------------------
# USER CONFIGURATION FOR BENCHMARK TESTING
# -----------------------------------------------------------------------------
#
MISMATCH_FACTOR = 1
GAIN_MISMATCH = 1
N_CONTEXT = 10
HIDDEN_SIZE = 128
HORIZON = 10
N_WARMUP = 1
NUM_ITER = 1050
EXPERIMENT_ID = "experiment_3.1.1"
MODEL_NAME = "cts-lstm-batched-128"
USE_MEAS_NOISE = False
# Measurement-noise standard deviation, as a percentage of the tank level
# operating range ([0, 10] m). 1.0 -> sigma = 0.1 m on both levels.
NOISE_SIGMA_PCT = 1.0
_SHOOTING="multi"

STATE_KEYS = ["h_1", "h_2"]
SAMPLE_TIME_S = 4.0
TRACK_INDEX = 1  # h_2 is the tracked controlled variable
LEVEL_RANGE = 10.0  # m, shared full scale of both tanks

# The checkpoints used here are trained on raw metres and volts, so the whole
# loop -- plant, context windows, setpoints and the NLP -- lives in plant units
# and no unit conversion happens anywhere.

# Piecewise-constant setpoint schedule [h_1, h_2], in metres.
SETPOINT_VALUES = [
    [[0.0], [5.0]],
    [[0.0], [8.0]],
    [[0.0], [2.0]],
    [[0.0], [5.0]],
]
SETPOINT_TIMESTAMPS = [0, 200, 400, 600]


@dataclass
class RunConfig:
    """Every knob of one closed-loop cascaded-two-tank run.

    The defaults mirror the module-level constants above, so ``RunConfig()``
    reproduces exactly what ``python neural_mpc_cts.py`` does. The parallel
    experiment runner in ``examples/Benchmarks`` builds one of these per run
    instead of editing the constants.

    Notes
    -----
    Passing ``cfg=None`` to :class:`NtiSystem` or :class:`NeuralMpc` keeps the
    pre-refactor behaviour of reading the module constants and class attributes,
    which is what ``mpc_hpo_cts.py`` relies on when it assigns
    ``NeuralMpc.horizon`` / ``NeuralMpc.pars_init`` before instantiating.
    """

    # --- plant/model mismatch (plant only; the LSTM never sees these) --------
    mismatch_factor: float = MISMATCH_FACTOR  # valve/outflow terms (k1, k2, k3)
    gain_mismatch: float = GAIN_MISMATCH  # pump actuator gain (k4)

    # --- controller ---------------------------------------------------------
    n_context: int = N_CONTEXT
    hidden_size: int = HIDDEN_SIZE
    horizon: int = HORIZON
    n_warmup: int = N_WARMUP
    model_name: Optional[str] = None  # None -> MODEL_NAME
    shooting: str = _SHOOTING
    pars_init: Optional[dict] = None  # None -> NeuralMpc.pars_init

    # --- simulation ---------------------------------------------------------
    num_iter: int = NUM_ITER
    x0: Optional[Sequence[float]] = None  # None -> (x1_init, x2_init)

    # --- measurement noise --------------------------------------------------
    use_meas_noise: bool = USE_MEAS_NOISE
    noise_sigma_pct: float = NOISE_SIGMA_PCT
    seed: int = 69

    # --- unmeasured step disturbance (controller is never told) -------------
    dist_kind: str = "none"  # "none" | "leak2"
    dist_magnitude: float = 0.0  # relative increase of k3, e.g. 0.10 -> +10 %
    dist_onset: int = 0  # simulation step at which the step is applied

    # --- bookkeeping --------------------------------------------------------
    experiment_id: str = EXPERIMENT_ID
    capture_lstm_state: bool = False

    def resolved_model_name(self) -> str:
        """Checkpoint stem.

        Defaults to :data:`MODEL_NAME` when explicitly given; otherwise the name
        is derived from the hidden size, so a hidden-size sweep stays inside a
        single, consistently-trained checkpoint family.
        """
        if self.model_name:
            return self.model_name
        return f"cts-lstm-batched-{self.hidden_size}"


class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    handlers=[_TqdmLoggingHandler()],
)

try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd()
project_root = current_dir.parent.parent
library_dir = project_root
if str(library_dir) not in sys.path:
    sys.path.insert(0, str(library_dir))

class NtiSystem(gym.Env[npt.NDArray[np.floating], float]):
    """Nonlinear cascaded two-tank system with discrete-time dynamics (Ts=4 s).

    Models a cascaded two-tank system where water flows from tank 1 to tank 2,
    with outflow proportional to sqrt(height) and optional overflow handling.

    Model dynamics
    --------------
    x1[k+1] = k6 * x1[k] - k1 * sqrt(x1[k]) + k4 * u[k]
    x2[k+1] = k7 * x2[k] + k2 * sqrt(x1[k]) - k3 * sqrt(x2[k]) + k5 * overflow

    States
    ------
    x[0] : h1
        Tank 1 water level [m], range [0, 10]
    x[1] : h2
        Tank 2 water level [m], range [0, 10]

    Inputs
    ------
    u[0]
        Pump input voltage [V], range [0, 10]

    Parameters
    ----------
    mismatch_factor : float
        Parametric mismatch for sqrt(h) terms (valves/outflow). Reducing it
        simulates a clog; increasing it simulates a leak.
    gain_mismatch : float
        Actuator mismatch (pump strength).

    Attributes
    ----------
    nx : int
        Number of states (2).
    nu : int
        Number of inputs (1).
    k1, k2, k3, k4, k5, k6, k7 : float
        Model parameters identified for Ts=4 s.
    x_bnd : tuple
        State bounds (lower, upper), each shape (2, 1).
    a_bnd : tuple
        Action bounds (lower, upper).
    use_meas_noise : bool
        Whether to add measurement noise.

    Notes
    -----
    The parameters (k1..k7) are identified for 4-second sampling.
    Do not change Δt unless the model is re-parameterized accordingly.
    """

    nx = 2
    nu = 1

    # Nominal coefficients identified for Ts = 4 s. The mismatch factors below
    # scale these on the *plant* only; the LSTM is never told.
    K1_NOM, K2_NOM, K3_NOM = (
        0.265885591506958,
        0.1621260792016983,
        0.15335486829280853,
    )
    K4_NOM = 0.16618020832538605
    k5, k6 = 1.0285956859588623, 1.0295900106430054
    k7 = 0.9935693740844727

    x1_init, x2_init = 0.0, 0.0
    x1_max, x2_max = 10.0, 10.0
    x_bnd = (np.asarray([[0.0], [0.0]]), np.asarray([[10.0], [10.0]]))
    a_bnd = (0.0, 10.0)
    e_bnd = (0.0, 1e-1)

    action_space = Box(*a_bnd, (nu,), np.float64)

    def __init__(self, cfg: "Optional[RunConfig]" = None) -> None:
        """Build the plant for one run.

        Parameters
        ----------
        cfg : RunConfig, optional
            Run configuration. When ``None`` the module-level constants are
            used, reproducing the pre-refactor behaviour.
        """
        super().__init__()
        cfg = cfg if cfg is not None else RunConfig()
        self.cfg = cfg

        self.mismatch_factor = float(cfg.mismatch_factor)
        self.gain_mismatch = float(cfg.gain_mismatch)

        self.k1 = self.K1_NOM * self.mismatch_factor
        self.k2 = self.K2_NOM * self.mismatch_factor
        self.k3_nom = self.K3_NOM * self.mismatch_factor
        self.k3 = self.k3_nom
        self.k4 = self.K4_NOM * self.gain_mismatch

        self.use_meas_noise = bool(cfg.use_meas_noise)
        # Gaussian measurement noise, sigma as a percentage of the [0, 10] m
        # level range so the number is comparable across both benchmarks.
        self.noise_sigma = np.full(
            (self.nx, 1), cfg.noise_sigma_pct / 100.0 * LEVEL_RANGE, dtype=np.float64
        )

        if cfg.x0 is not None:
            self.x0 = np.asarray(cfg.x0, dtype=np.float64).reshape(self.nx, 1)
        else:
            self.x0 = np.asarray(
                [self.x1_init, self.x2_init], dtype=np.float64
            ).reshape(self.nx, 1)

        self._k = 0  # simulation step counter, drives the step disturbance
        self.x = self.x0.copy()

    def _apply_disturbance(self) -> None:
        """Set the tank-2 outflow coefficient for the current step.

        ``leak2`` cracks a drain valve open on tank 2 at ``dist_onset``: an
        unmeasured, persistent step in ``k3``. It deliberately acts on the same
        coefficient ``mismatch_factor`` scales, so the persistent-mismatch and
        step-disturbance axes stay physically comparable.
        """
        active = self.cfg.dist_kind == "leak2" and self._k >= self.cfg.dist_onset
        self.k3 = self.k3_nom * (1.0 + self.cfg.dist_magnitude if active else 1.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Reset system state to initial conditions.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional reset options

        Returns
        -------
        state : np.ndarray
            Initial state [x1_init, x2_init], shape (nx, 1)
        info : dict
            Additional information (empty)
        """
        super().reset(seed=seed, options=options)
        self.x = self.x0.copy()
        self._k = 0
        self.k3 = self.k3_nom
        return self.x.copy(), {}

    def step(self, action: npt.NDArray[np.floating]):
        """Advance one discrete time step.

        Dynamics
        --------
        xp1 = k6 * x1 - k1 * sqrt(x1) + k4 * u
        xp2 = k7 * x2 + k2 * sqrt(x1) - k3 * sqrt(x2)
        x1_new = clip(xp1, 0, x1_max)
        x2_new = clip(xp2 + k5 * overflow, 0, x2_max)

        Parameters
        ----------
        action : np.ndarray
            Control input (pump voltage), shape (nu, 1)

        Returns
        -------
        observation : np.ndarray
            New state [h1, h2], shape (nx, 1)
        reward : float
            Always 0.0 (unused)
        terminated : bool
            Always False
        truncated : bool
            Always False
        info : dict
            Additional information (empty)
        """
        u = np.asarray(action, dtype=np.float64).reshape(self.nu, 1)
        u = np.clip(u, self.a_bnd[0], self.a_bnd[1])

        self._apply_disturbance()

        xp1 = self.k6 * self.x[0] - self.k1 * np.sqrt(self.x[0]) + self.k4 * u
        xp2 = (
            self.k7 * self.x[1]
            + self.k2 * np.sqrt(self.x[0])
            - self.k3 * np.sqrt(self.x[1])
        )
        xov1 = np.maximum(xp1 - self.x1_max, 0.0)

        x_new = np.empty((2, 1))
        x_new[0] = np.clip(xp1, 0.0, self.x1_max)
        x_new[1] = np.clip(xp2 + self.k5 * xov1, 0.0, self.x2_max)

        if self.use_meas_noise:
            # Always draw, then scale. Never branch on sigma == 0: the RNG must
            # advance identically across noise levels so that a given replicate
            # sees the *same* realisation at every sigma (common random numbers).
            noise = self.np_random.standard_normal((self.nx, 1)) * self.noise_sigma
            x_new = x_new + noise
        x_new[0] = np.clip(x_new[0], 0.0, self.x1_max)
        x_new[1] = np.clip(x_new[1], 0.0, self.x2_max)
        self.x = x_new
        self._k += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return self.x.copy(), reward, terminated, truncated, info


class NeuralMpc(Mpc[cs.MX]):
    """Neural MPC controller using LSTM dynamics for two-tank system control.

    Implements a receding-horizon optimal controller that uses a trained LSTM
    neural network as the internal prediction model. The LSTM hidden state is
    estimated using a context window of past observations, enabling closed-loop
    state estimation without explicit system identification of initial conditions.

    Attributes
    ----------
    horizon : int
        Prediction horizon length.
    discount_factor : float
        Discount factor for cost function (1.0 = no discounting).
    n_context : int
        Number of past time steps used for RNN state estimation.
    n_inputs : int
        Number of control inputs (1: pump voltage).
    n_outputs : int
        Number of measured outputs (1: h2 level).
    sequence_length : int
        Total sequence length: horizon + n_context.
    batch_size : int
        Batch size for neural network (1).
    pars_init : dict
        Default tuning parameters including state bounds, cost matrices,
        and slack variable weights.

    Notes
    -----
    - Only h2 (tank 2 level) is used for control (nx=1 in MPC, though plant has 2 states).
    - Soft constraints on state bounds are enforced via slack variables with penalty.
    - Hard constraints are applied to the first output (h2).
    - Cost function includes tracking error, control effort, and terminal cost.
    - Uses IPOPT solver with custom tolerance settings for real-time feasibility.

    References
    ----------
    .. [1] Adhau, S. et al. (2024). "Reinforcement learning based MPC with
           neural dynamical models."
    .. [2] Forgione, M. et al. (2022). "Learning in MPC: Learning Initial
           State Estimation for Recurrent Neural Network Dynamics."
    """

    horizon = HORIZON
    discount_factor = 1.0
    n_context = N_CONTEXT  # used for initial RNN state estimation
    n_inputs = 1
    n_outputs = 1
    sequence_length = horizon + n_context
    batch_size = 1

    pars_init = {
        "x_lb": np.asarray(0),
        "x_ub": np.asarray(10),
        "x_lb_f": np.asarray(0),
        "x_ub_f": np.asarray(10),
        "b": np.asarray(0.0),
        "H_s": np.asarray(1e3),
        "h_s": np.asarray(0),
        "c_s": np.asarray(0),
        "H_lt": np.asarray([[1.0, 0], [0, 1e-3]]),
        "h_lt": np.asarray([0, 0]),
        "c_lt": np.asarray(0),
        "H_0": np.asarray(0.0),
        "h_0": np.asarray(0.0),
        "c_0": np.asarray(0),
        "w": np.asarray(100),  # penalty weight for bound violations
        "x_scaling": np.asarray([0.1], dtype=float),
        "u_scaling": np.asarray([0.1], dtype=float),
    }

    def __init__(self, cfg: "Optional[RunConfig]" = None) -> None:
        """Initialize Neural MPC with LSTM dynamics and cost function.

        Parameters
        ----------
        cfg : RunConfig, optional
            Run configuration. When ``None`` the class attributes and module
            constants are used unchanged, which is what ``mpc_hpo_cts.py``
            relies on when it assigns ``NeuralMpc.horizon`` /
            ``NeuralMpc.pars_init`` before instantiating.

        Notes
        -----
        Sets up the MPC problem with:
        - State variables over the prediction horizon
        - Control action variables with input constraints
        - Slack variables for soft state constraints
        - Neural dynamics using pre-trained LSTM model
        - Quadratic cost function with terminal and stage costs
        - IPOPT solver with custom tolerances
        """
        # cfg wins when supplied; otherwise fall back to the class attributes so
        # that callers which mutate them (the HPO scripts) keep working.
        if cfg is None:
            horizon = self.horizon
            n_context = self.n_context
            hidden_size = HIDDEN_SIZE
            n_warmup = N_WARMUP
            model_name = MODEL_NAME
            shooting = _SHOOTING
            pars_init = self.pars_init
        else:
            horizon = int(cfg.horizon)
            n_context = int(cfg.n_context)
            hidden_size = int(cfg.hidden_size)
            n_warmup = int(cfg.n_warmup)
            model_name = cfg.resolved_model_name()
            shooting = cfg.shooting
            pars_init = cfg.pars_init if cfg.pars_init is not None else self.pars_init

        N = horizon
        gamma = self.discount_factor

        nx, nu = NtiSystem.nx - 1, NtiSystem.nu
        a_bnd = NtiSystem.a_bnd

        nlp = Nlp(sym_type="MX")
        super().__init__(
            nlp,
            N,
            tuning_parameters=pars_init,
            n_context=n_context,
            shooting=shooting,
            neural=True,
        )

        # Recorded per instance (shadowing the class attributes) so a caller can
        # read back what this controller was actually built with.
        self.horizon = horizon
        self.n_context = n_context
        self.hidden_size = hidden_size
        self.sequence_length = horizon + n_context
        self.shooting = shooting
        self.model_name = model_name

        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        x_lb_f = self.parameter("x_lb_f", (nx,))
        x_ub_f = self.parameter("x_ub_f", (nx,))

        b = self.parameter("b")
        H_s = self.parameter("H_s", (nx, nx))
        h_s = self.parameter("h_s", (nx,))
        c_s = self.parameter("c_s")

        H_lt = self.parameter("H_lt", (nx + nu, nx + nu))
        h_lt = self.parameter("h_lt", (nx + nu,))
        c_lt = self.parameter("c_lt")

        H_0 = self.parameter("H_0", (nx, nx))
        h_0 = self.parameter("h_0", (nx,))
        c_0 = self.parameter("c_0")

        w = self.parameter("w", (nx, 1))
        x_scaling = self.parameter("x_scaling", (nx, 1))
        self.parameter("u_scaling", (nu, 1))
        SP = self.parameter("SP", (nx, 1))

        x, x0 = self.state("x", nx, bound_initial=False)
        u, u_exp, u0 = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s1, _, _ = self.variable("s1", (nx, N), lb=0)
        s2, _, _ = self.variable("s2", (nx, 1), lb=0)

        model = CasadiLSTM(
            n_context,
            self.n_inputs,
            hidden_size=hidden_size,
            horizon=N,
            proj_size=1,
        )

        model_filename = f"{model_name}.pt"
        model_path = (
            project_root
            / "examples"
            / "Cascaded_Two_Tank_System"
            / "models"
            / model_filename
        )
        assert model_path.exists(), f"Model file not found at '{model_path}'"
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))

        self.set_neural_dynamics(
            model=model,
            output_bias=b,
            name="F_neural",
            n_warmup=n_warmup,
        )
        if shooting == "single":
            # Single shooting: state() returned None; the trajectory exists only
            # after set_dynamics builds it by forward simulation. Re-bind x to it.
            x = self.states["x"]

        xlb_rep = cs.repmat(x_lb, 1, N)
        xub_rep = cs.repmat(x_ub, 1, N)
        hard_indices = [0]
        self.constraint("s1_hard", s1[hard_indices, :], "==", 0)
        self.constraint("s2_hard", s2[hard_indices, :], "==", 0)
        self.constraint(
            "x_lb",
            xlb_rep * x_scaling - s1,
            "<=",
            x[:, :] * x_scaling,
        )
        self.constraint(
            "x_ub",
            x[:, :] * x_scaling,
            "<=",
            xub_rep * x_scaling + s1,
        )
        self.constraint("x_lb_f", x_lb_f * x_scaling - s2, "<=", x[:, -1] * x_scaling)
        self.constraint("x_ub_f", x[:, -1] * x_scaling, "<=", x_ub_f * x_scaling + s2)

        e_N = x[:, -1] - SP
        e_N = e_N * x_scaling
        S = (gamma**N) * (0.5 * cs.bilin(H_s, e_N) + h_s.T @ e_N + c_s + w.T @ s2)

        e_0 = x0 - SP
        e_0 = e_0 * x_scaling
        V0 = 0.5 * cs.bilin(H_0, e_0) + h_0.T @ e_0 + c_0

        Lt = 0.0

        for k in range(0, N):
            e_k = x[:, k] - SP
            e_k = e_k * x_scaling
            Lt += (gamma**k) * (
                0.5 * cs.bilin(H_lt, cs.vertcat(e_k, u_exp[:, k]))
                + h_lt.T @ cs.vertcat(e_k, u_exp[:, k])
                + c_lt
            )
            Lt += (gamma**k) * (w.T @ s1[:, k])

        self.minimize(V0 + S + Lt)

        opts = {
            "print_time": False,
            "ipopt": {
                "max_iter": 200,
                "sb": "yes",
                "print_level": 0,
                "tol": 1e-4,
                "acceptable_tol": 1e-2,
            },
        }
        self.init_solver(opts, solver="ipopt")


def get_current_setpoint(timestep: int) -> np.ndarray:
    """Return the most recent setpoint not exceeding the given timestep.

    Parameters
    ----------
    timestep : int
        Current simulation timestep

    Returns
    -------
    setpoint : np.ndarray
        Setpoint value [h1, h2], shape (2, 1)
    """
    idx = max(
        i
        for i in range(len(SETPOINT_TIMESTAMPS))
        if SETPOINT_TIMESTAMPS[i] <= timestep
    )
    return np.asarray(SETPOINT_VALUES[idx])


MAX_SEED = np.iinfo(np.uint32).max + 1


def mk_seed(rng: np.random.Generator) -> int:
    """Generate a random seed in [0, 2**32).

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random number generator

    Returns
    -------
    seed : int
        Random seed value
    """
    return int(rng.integers(MAX_SEED))


def simulate(cfg: "Optional[RunConfig]" = None, progress: bool = False) -> dict:
    """Run one closed-loop two-tank simulation and return its raw trajectories.

    This is the single entry point shared by ``python neural_mpc_cts.py`` and
    the parallel experiment runner. It never plots and never writes files.

    Parameters
    ----------
    cfg : RunConfig, optional
        Run configuration; ``None`` means "the module defaults", i.e. exactly
        what the script did before it was parameterised.
    progress : bool, optional
        Show a per-step tqdm bar. Off for batch runs, on for the standalone
        script.

    Returns
    -------
    dict
        ``X`` (nsteps+1, 2) plant levels [m], ``X_pred`` (nsteps,) one-step h_2
        predictions (NaN wherever the solve failed), ``U`` (nsteps,) applied
        pump voltage, ``SP`` (nsteps, 2) setpoints, ``exec_ms`` (nsteps,) solve
        wall time, ``solve_ok`` (nsteps,) bool, ``n_failed_solves`` int, and
        optionally ``c_states`` / ``h_states`` LSTM state trajectories.

    Notes
    -----
    Failed solves are detected from the *delta* of ``mpc.failures`` rather than
    from ``mpc._last_solution``: under the default ``last-successful`` warm
    start the stored solution is not replaced on failure, so it would report the
    previous successful solve and the harvested prediction would silently
    repeat.
    """
    cfg = cfg if cfg is not None else RunConfig()

    simulation_time = int(cfg.num_iter)
    mpc = NeuralMpc(cfg)
    env = NtiSystem(cfg)

    state_indices = [1]

    rng = np.random.default_rng(cfg.seed)
    state, _ = env.reset(seed=mk_seed(rng), options=None)

    X, U, SP, X_pred = [state], [], [], []
    state_context = np.tile(state.T, (mpc.n_context, 1))
    action_context = np.zeros((mpc.n_context, NtiSystem.nu))

    exec_times_ms = []
    solve_ok = []
    c_states, h_states = [], []

    vals0 = None
    input_bias = None
    store_solution = True

    timestep = 0

    gc.disable()

    try:
        pbar = (
            tqdm(total=simulation_time, desc="MPC Simulation", unit="step",
                 ncols=80, colour="green")
            if progress
            else None
        )
        for _t in range(simulation_time):
            sp = get_current_setpoint(timestep)

            # The whole loop -- plant, contexts, NLP -- is in plant units
            # (metres, volts); the LSTM was trained on exactly those.
            failures_before = mpc.failures
            t0 = time.perf_counter()
            u_opt = mpc.solve_mpc(
                state_context=state_context,
                state_indices=state_indices,
                action_context=action_context,
                setpoint=sp,
                input_bias=input_bias,
                vals0=vals0,
                store_solution=store_solution,
            )
            t1 = time.perf_counter()
            ok = mpc.failures == failures_before

            exec_times_ms.append((t1 - t0) * 1000.0)
            solve_ok.append(ok)

            if cfg.capture_lstm_state and mpc._lstm_c is not None:
                c_states.append(
                    np.concatenate([np.asarray(c).ravel() for c in mpc._lstm_c])
                )
                h_states.append(
                    np.concatenate([np.asarray(h).ravel() for h in mpc._lstm_h])
                )

            obs, _, _, _, _ = env.step(np.asarray(u_opt))
            state_context = np.vstack([state_context, obs.T])[-mpc.n_context :]
            action_context = np.vstack([action_context, np.asarray(u_opt).T])[
                -mpc.n_context :
            ]

            if ok and mpc._last_solution is not None:
                if mpc.shooting == "single":
                    # Single shooting: x is a derived expression (not a primal
                    # variable), so evaluate it at the solution.
                    x_pred_traj = mpc._last_solution.value(mpc.states["x"])
                    X_pred.append(np.asarray(x_pred_traj[:, 0], dtype=float))
                else:
                    X_pred.append(
                        np.asarray(mpc._last_solution.vals["x"][:, 0], dtype=float)
                    )
            else:
                X_pred.append(np.asarray([np.nan]).reshape(1, 1))

            X.append(obs)
            U.append(u_opt)
            SP.append(sp)
            timestep += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"solver_ms": f"{exec_times_ms[-1]:.1f}"})
        if pbar is not None:
            pbar.close()
    finally:
        gc.enable()

    solve_ok_arr = np.asarray(solve_ok, dtype=bool)
    out = {
        "X": np.squeeze(np.array(X)),
        "X_pred": np.squeeze(np.array(X_pred)),
        "U": np.squeeze(np.array(U)),
        "SP": np.squeeze(np.array(SP)),
        "exec_ms": np.asarray(exec_times_ms, dtype=np.float64),
        "solve_ok": solve_ok_arr,
        "n_failed_solves": int((~solve_ok_arr).sum()),
        "n_solves": int(solve_ok_arr.size),
    }
    if cfg.capture_lstm_state and c_states:
        out["c_states"] = np.asarray(c_states, dtype=np.float64)
        out["h_states"] = np.asarray(h_states, dtype=np.float64)
    return out


if __name__ == "__main__":

    env = NtiSystem(RunConfig())
    result = simulate(RunConfig(), progress=True)
    X, X_pred = result["X"], result["X_pred"]
    U, SP = result["U"], result["SP"]
    exec_times_ms = list(result["exec_ms"])

    X = np.squeeze(np.array(X))
    X_pred = np.squeeze(np.array(X_pred))
    U = np.squeeze(np.array(U))
    SP = np.squeeze(np.array(SP))

    import pandas as pd

    experiment_id = EXPERIMENT_ID
    save_dir = (
        project_root / "examples" / "Cascaded_Two_Tank_System" / "data" / experiment_id
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    df_system = pd.DataFrame(
        {
            "step": np.arange(len(U)),
            "time_s": np.arange(len(U))*4,
            "h_1": X[1:, 0],  # State at start of step
            "h_2": X[1:, 1],
            "h_2_pred": X_pred[:],
            "u [V]": U[:],  # Control applied
            "h_1_sp": SP[:, 0],  # Setpoint active
            "h_2_sp": SP[:, 1],
            "h_1_next": X[1:, 0],  # Resulting state
            "h_2_next": X[1:, 1],
        }
    )

    df_benchmark = pd.DataFrame(
        {"step": np.arange(len(exec_times_ms)), "exec_time_ms": exec_times_ms}
    )

    system_file = save_dir / "simulation_data.csv"
    bench_file = save_dir / "benchmark_stats.csv"

    df_system.to_csv(system_file, index=False)
    df_benchmark.to_csv(bench_file, index=False)

    print(f"Data saved to: {save_dir}")
    print(f"  System Response : {system_file}")
    print(f"  Benchmark Data  : {bench_file}")

    COLOR_REAL = "#000000"  # real plant state
    COLOR_PRED = "#0072B2"  # one-step model prediction
    COLOR_SP = "#009E73"    # setpoint

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=False)
    fig.suptitle("System Response", fontsize=11, fontweight="bold")
    timesteps = np.arange(X.shape[0])

    lb_states, ub_states = env.x_bnd

    ax_h2 = axes[0, 0]
    ax_h2.plot(
        timesteps, X[:, 1], color=COLOR_REAL, linestyle="-", linewidth=1.5, label="plant"
    )
    ax_h2.plot(
        timesteps[1:], X_pred[:], color=COLOR_PRED, linestyle=":", linewidth=1.2,
        label="pred.",
    )
    ax_h2.plot(
        timesteps[1:], SP[:, 1], color=COLOR_SP, linestyle="--", linewidth=1.0,
        label="setpoint",
    )
    ax_h2.axhline(lb_states[1, 0], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_h2.axhline(ub_states[1, 0], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_h2.set_title(r"$h_2$", fontsize=9)
    ax_h2.set_ylabel(r"$h_2$ [m]")

    ax_h1 = axes[0, 1]
    ax_h1.plot(
        timesteps, X[:, 0], color=COLOR_REAL, linestyle="-", linewidth=1.5, label="plant"
    )
    ax_h1.plot(
        timesteps[1:], SP[:, 0], color=COLOR_SP, linestyle="--", linewidth=1.0,
        label="setpoint",
    )
    ax_h1.axhline(lb_states[0, 0], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_h1.axhline(ub_states[0, 0], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_h1.set_title(r"$h_1$", fontsize=9)
    ax_h1.set_ylabel(r"$h_1$ [m]")

    ax_u = axes[1, 0]
    ax_u.step(timesteps[1:], U, where="post", color=COLOR_PRED, label=r"$u$")
    ax_u.axhline(env.action_space.low[0], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_u.axhline(env.action_space.high[0], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_u.set_title(r"$u$ (input)", fontsize=9)
    ax_u.set_ylabel("u [V]")

    # Fourth panel unused (only three signals to show).
    axes[1, 1].axis("off")

    for ax in (ax_h2, ax_h1, ax_u):
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
        ax.grid(True, which="major", alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("time step")

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    exec_array = np.array(exec_times_ms)
    mean_time = np.mean(exec_array)
    max_time = np.max(exec_array)
    p99_time = np.percentile(exec_array, 99)

    print(f"\n--- MPC Benchmark Stats ({len(exec_array)} samples) ---")
    print(f"Mean Execution Time: {mean_time:.2f} ms")
    print(f"Max Execution Time:  {max_time:.2f} ms")
    print(f"99th Percentile:     {p99_time:.2f} ms")

    fig_bench, ax_bench = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig_bench.suptitle(
        f"Benchmark Results (Mean: {mean_time:.2f}ms, P99: {p99_time:.2f}ms)",
        fontsize=11, fontweight="bold",
    )

    ax_bench[0].plot(exec_array, color=COLOR_PRED, linewidth=1.2, label="Computation Time")
    ax_bench[0].set_xlabel("Simulation Step")
    ax_bench[0].set_ylabel("Time (ms)")
    ax_bench[0].set_title("Execution Time per Step", fontsize=9)
    ax_bench[0].legend(loc="best", fontsize=7, framealpha=0.9)
    ax_bench[0].grid(True, which="major", alpha=0.3)
    ax_bench[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax_bench[0].yaxis.set_minor_locator(AutoMinorLocator())

    ax_bench[1].hist(
        exec_array, bins=30, color="#D55E00", alpha=0.7, edgecolor="black"
    )
    ax_bench[1].axvline(p99_time, color="red", linestyle="--", label="99th Percentile")
    ax_bench[1].axvline(mean_time, color=COLOR_REAL, linestyle="--", label="Mean")
    ax_bench[1].set_xlabel("Time (ms)")
    ax_bench[1].set_ylabel("Frequency")
    ax_bench[1].set_title("Latency Distribution", fontsize=9)
    ax_bench[1].legend(loc="best", fontsize=7, framealpha=0.9)
    ax_bench[1].grid(True, which="major", alpha=0.3)
    ax_bench[1].xaxis.set_minor_locator(AutoMinorLocator())
    ax_bench[1].yaxis.set_minor_locator(AutoMinorLocator())

    plt.show()
