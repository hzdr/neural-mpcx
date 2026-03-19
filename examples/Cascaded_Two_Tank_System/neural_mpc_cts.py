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
from pathlib import Path
from typing import Any
import time  # measure computation time
import gc  # Garbage Collector control

import casadi as cs
import gymnasium as gym
import matplotlib.pyplot as plt
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
WARMUP_TYPE = "X0"  # "NONE", "ZEROS" OR "X0"
IS_ESTIMATOR = True if WARMUP_TYPE != "NONE" else False
NUM_ITER = 1050
EXPERIMENT_ID = "experiment_3.1.1"
MODEL_NAME = "cts-lstm-batched-128"

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

    mismatch_factor = MISMATCH_FACTOR
    gain_mismatch = GAIN_MISMATCH

    k1, k2, k3 = (
        0.265885591506958 * mismatch_factor,
        0.1621260792016983 * mismatch_factor,
        0.15335486829280853 * mismatch_factor,
    )
    k4, k5, k6 = (
        0.16618020832538605 * gain_mismatch,
        1.0285956859588623,
        1.0295900106430054,
    )
    k7 = 0.9935693740844727

    x1_init, x2_init = 0.0, 0.0
    x1_max, x2_max = 10.0, 10.0
    x_bnd = (np.asarray([[0.0], [0.0]]), np.asarray([[10.0], [10.0]]))
    a_bnd = (0.0, 10.0)
    e_bnd = (0.0, 1e-1)

    action_space = Box(*a_bnd, (nu,), np.float64)
    output_noise_low = np.array([-0.01], dtype=np.float64)
    output_noise_high = np.array([0.01], dtype=np.float64)
    use_meas_noise = False

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
        self.x = np.asarray([self.x1_init, self.x2_init]).reshape(self.nx, 1)
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
            noise = self.np_random.uniform(
                self.output_noise_low, self.output_noise_high
            ).reshape(self.nx, 1)
            x_new = x_new + noise

        self.x = x_new
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

    def __init__(self) -> None:
        """Initialize Neural MPC with LSTM dynamics and cost function.

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
        N = self.horizon
        gamma = self.discount_factor

        nx, nu = NtiSystem.nx - 1, NtiSystem.nu
        a_bnd = (0.0, 10.0)

        nlp = Nlp(sym_type="MX")
        super().__init__(
            nlp,
            N,
            tuning_parameters=self.pars_init,
            n_context=self.n_context,
            shooting="multi",
            neural=True,
        )

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

        x, _ = self.state("x", nx, bound_initial=False)
        u, u_exp, u0 = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s1, _, _ = self.variable("s1", (nx, N), lb=0)
        s2, _, _ = self.variable("s2", (nx, 1), lb=0)

        model = CasadiLSTM(
            self.n_context,
            self.n_inputs,
            hidden_size=HIDDEN_SIZE,
            horizon=self.horizon,
            proj_size=1,
            is_estimator=IS_ESTIMATOR,
            input_order="y_then_u",
        )

        model_name = MODEL_NAME
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
            input_order="y_then_u",
            output_bias=b,
            name="F_neural",
            remove_bounds_on_initial_action=True,
        )

        xlb_rep = cs.repmat(x_lb, 1, N)
        xub_rep = cs.repmat(x_ub, 1, N)
        hard_indices = [0]
        self.constraint("s1_hard", s1[hard_indices, :], "==", 0)
        self.constraint("s2_hard", s2[hard_indices, :], "==", 0)
        self.constraint(
            "x_lb",
            xlb_rep * x_scaling - s1,
            "<=",
            x[:, (self.n_context - 1) : -1] * x_scaling,
        )
        self.constraint(
            "x_ub",
            x[:, (self.n_context - 1) : -1] * x_scaling,
            "<=",
            xub_rep * x_scaling + s1,
        )
        self.constraint("x_lb_f", x_lb_f * x_scaling - s2, "<=", x[:, -1] * x_scaling)
        self.constraint("x_ub_f", x[:, -1] * x_scaling, "<=", x_ub_f * x_scaling + s2)

        e_N = x[:, -1] - SP
        e_N = e_N * x_scaling
        S = (gamma**N) * (0.5 * cs.bilin(H_s, e_N) + h_s.T @ e_N + c_s + w.T @ s2)

        e_0 = x[:, self.n_context - 1] - SP
        e_0 = e_0 * x_scaling
        V0 = 0.5 * cs.bilin(H_0, e_0) + h_0.T @ e_0 + c_0

        Lt = 0.0

        for k in range(self.n_context - 1, self.n_context - 1 + N):
            e_k = x[:, k] - SP
            e_k = e_k * x_scaling
            k_abs = k - (self.n_context - 1)
            Lt += (gamma**k_abs) * (
                0.5 * cs.bilin(H_lt, cs.vertcat(e_k, u_exp[:, k]))
                + h_lt.T @ cs.vertcat(e_k, u_exp[:, k])
                + c_lt
            )
            Lt += (gamma**k_abs) * (w.T @ s1[:, k_abs])

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


if __name__ == "__main__":

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
            for i in range(len(setpoint_timestamps))
            if setpoint_timestamps[i] <= timestep
        )
        return np.asarray(setpoint_values[idx])

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

    simulation_time = NUM_ITER
    mpc = NeuralMpc()
    env = NtiSystem()

    setpoint_values = [
        [[0.0], [5.0]],
        [[0.0], [8.0]],
        [[0.0], [2.0]],
        [[0.0], [5.0]],
    ]
    setpoint_timestamps = [0, 200, 400, 600]

    state_indices = [1]

    rng = np.random.default_rng(69)
    state, _ = env.reset(seed=mk_seed(rng), options=None)

    X, U, SP, X_pred = [state], [], [], []
    if WARMUP_TYPE == "X0":
        state_context = np.tile(state.T, (mpc.n_context, 1))
    else:
        state_context = np.zeros((mpc.n_context, NtiSystem.nx))
    action_context = np.zeros((mpc.n_context, NtiSystem.nu))

    exec_times_ms = []

    vals0 = None
    input_bias = None
    store_solution = True

    timestep = 0
    setpoint = np.zeros_like(setpoint_values[0])

    gc.disable()

    try:
        with tqdm(total=simulation_time, desc="MPC Simulation", unit="step", ncols=80, colour="green") as pbar:
            for t in range(simulation_time):
                sp = get_current_setpoint(timestep)

                t0 = time.perf_counter()
                u_opt = mpc.solve_mpc(
                    state,
                    state_context,
                    state_indices,
                    action_context,
                    sp,
                    input_bias,
                    vals0,
                    store_solution
                )
                t1 = time.perf_counter()
                exec_times_ms.append((t1 - t0) * 1000.0)

                obs, _, _, _, _ = env.step(np.asarray(u_opt))
                state = obs
                if WARMUP_TYPE == "NONE":
                    state_context = np.zeros((mpc.n_context, NtiSystem.nx))
                    action_context = np.zeros((mpc.n_context, NtiSystem.nu))
                else:
                    state_context = np.vstack([state_context, obs.T])[-mpc.n_context :]
                    action_context = np.vstack([action_context, np.asarray(u_opt).T])[
                        -mpc.n_context :
                    ]

                if mpc._last_solution is not None:
                    X_pred.append(
                        np.asarray(mpc._last_solution.vals["x"][:, mpc._n_context])
                    )
                else:
                    X_pred.append(np.asarray([np.nan]).reshape(1, 1))

                X.append(obs)
                U.append(u_opt)
                SP.append(sp)
                timestep += 1
                pbar.update(1)
                pbar.set_postfix({"solver_ms": f"{exec_times_ms[-1]:.1f}"})
    finally:
        gc.enable()


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
            "h1": X[1:, 0],  # State at start of step
            "h2": X[1:, 1],
            "h2_pred": X_pred[:],
            "u": U[:],  # Control applied
            "sp_h1": SP[:, 0],  # Setpoint active
            "sp_h2": SP[:, 1],
            "h1_next": X[1:, 0],  # Resulting state
            "h2_next": X[1:, 1],
        }
    )

    df_benchmark = pd.DataFrame(
        {"step": np.arange(len(exec_times_ms)), "exec_time_ms": exec_times_ms}
    )

    system_file = save_dir / "system_response.csv"
    bench_file = save_dir / "benchmark_stats.csv"

    df_system.to_csv(system_file, index=False)
    df_benchmark.to_csv(bench_file, index=False)

    print(f"Data saved to: {save_dir}")
    print(f"  System Response : {system_file}")
    print(f"  Benchmark Data  : {bench_file}")

    fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    fig.suptitle("System Response")
    timesteps = np.arange(X.shape[0])

    axs[0].plot(timesteps, X[:, 1], label="h2")
    axs[0].plot(timesteps[1:], SP[:, 1], linestyle=":", label="SP h2")
    axs[0].plot(timesteps[1:], X_pred[:], linestyle=":", label=r"$h2$ Pred.")
    axs[1].plot(timesteps, X[:, 0], label="h1")
    axs[1].plot(timesteps[1:], SP[:, 0], linestyle=":", label="SP h1")
    axs[2].step(timesteps[1:], U, where="post", label="u")

    lb_states, ub_states = env.x_bnd
    axs[0].axhline(lb_states[1, 0], linestyle="--")
    axs[0].axhline(ub_states[1, 0], linestyle="--")
    axs[1].axhline(lb_states[0, 0], linestyle="--")
    axs[1].axhline(ub_states[0, 0], linestyle="--")
    axs[2].axhline(env.action_space.low[0], linestyle="--")
    axs[2].axhline(env.action_space.high[0], linestyle="--")

    for ax, label in zip(axs, ("$h_2$ [m]", "$h_1$ [m]", "u [V]")):
        ax.set_ylabel(label)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("time step")

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
        f"Benchmark Results (Mean: {mean_time:.2f}ms, P99: {p99_time:.2f}ms)"
    )

    ax_bench[0].plot(exec_array, label="Computation Time")
    ax_bench[0].set_xlabel("Simulation Step")
    ax_bench[0].set_ylabel("Time (ms)")
    ax_bench[0].set_title("Execution Time per Step")
    ax_bench[0].grid(True, alpha=0.3)

    ax_bench[1].hist(exec_array, bins=30, color="orange", alpha=0.7, edgecolor="black")
    ax_bench[1].axvline(p99_time, color="red", linestyle="--", label="99th Percentile")
    ax_bench[1].axvline(mean_time, color="blue", linestyle="--", label="Mean")
    ax_bench[1].set_xlabel("Time (ms)")
    ax_bench[1].set_ylabel("Frequency")
    ax_bench[1].set_title("Latency Distribution")
    ax_bench[1].legend()

    plt.show()
