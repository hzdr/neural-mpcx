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
Nonlinear MPC benchmark for CSTR system using explicit dynamics.

Benchmark script for performance evaluation of NMPC controller with CSTR model.
Based on the do-mpc CSTR example from:
"do-mpc: Towards FAIR nonlinear and robust model predictive control" by F. Fiedler et al.
https://www.sciencedirect.com/science/article/pii/S0967066123002459

This benchmark measures computation time statistics in a controlled environment
with single-threaded execution to minimize timing variance.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from pathlib import Path
import time  # measure computation time
import gc  # Garbage Collector control

import casadi as cs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from gymnasium.spaces import Box
import sys

from neuralmpcx import Nlp
from neuralmpcx.wrappers import Mpc

# -----------------------------------------------------------------------------
# USER CONFIGURATION FOR BENCHMARK TESTING
# -----------------------------------------------------------------------------
#
ALPHA = 1
BETA = 1
HORIZON = 10
WARMUP_TYPE = "X0" # "ZEROS" OR "X0"
NUM_ITER = 1000
EXPERIMENT_ID = "experiment_3.1"

# -----------------------------------------------------------------------------
# Normalization Parameters
# -----------------------------------------------------------------------------
U_NORM_PARAMS = {
    "F": {"min": 0.0, "max": 100.0},
    "Q_dot": {"min": -8500.0, "max": 0.0},
}

Y_NORM_PARAMS = {
    "C_A": {"min": 0.0, "max": 5.1},
    "C_B": {"min": 0.0, "max": 5.1},
    "T_R": {"min": 0.0, "max": 140.0},
    "T_K": {"min": 0.0, "max": 140.0},
}

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


class CSTRSystem(gym.Env):
    """Continuous Stirred Tank Reactor (CSTR) simulation environment.

    Based on the do-mpc CSTR example from Klatt & Engell (1998). Implements
    a nonlinear reactor model with two parallel reactions (A→B and B→C) and
    one side reaction (2A→D), controlled via feed flow rate and heat removal.

    States
    ------
    x[0] : C_A
        Concentration of species A [mol/L], range [0, 5.1]
    x[1] : C_B
        Concentration of species B [mol/L], range [0, 5.1]
    x[2] : T_R
        Reactor temperature [°C], range [0, 140]
    x[3] : T_K
        Jacket (coolant) temperature [°C], range [0, 140]

    Inputs
    ------
    u[0] : F
        Feed flow rate [h^-1], normalized to [0, 1] from physical [0, 100]
    u[1] : Q_dot
        Heat removal rate [kJ/h], normalized to [0, 1] from physical [-8500, 0]

    Parameters
    ----------
    dt_hr : float, optional
        Simulation time step in hours (default 0.005h = 18s).

    Attributes
    ----------
    dt : float
        Time step in hours.
    x0 : np.ndarray
        Initial steady-state condition (shape 4x1).
    nx : int
        Number of states (4).
    nu : int
        Number of inputs (2).
    a_bnd : tuple
        Normalized input bounds (min, max) as (2x1) arrays.

    Notes
    -----
    - All state variables are normalized to [0, 1] for MPC compatibility.
    - Integration uses 4th-order Runge-Kutta (RK4) with physical units.
    - Reaction rates follow Arrhenius kinetics.
    - The model includes exothermic reactions and jacket cooling dynamics.

    References
    ----------
    .. [1] Klatt, K. U., & Engell, S. (1998). "Nonlinear dynamics and
           control of a continuous stirred tank reactor."
    """

    a_bnd = (
        np.array([[0.0], [0.0]], dtype=np.float64),
        np.array([[1.0], [1.0]], dtype=np.float64),
    )

    a_bnd_mpc = (
        np.array([[0.0], [0.0]], dtype=np.float64),
        np.array([[1.0], [1.0]], dtype=np.float64),
    )
    nx = 4
    nu = 2

    def __init__(self, dt_hr=0.005):
        """Initialize CSTR system with physical parameters and steady state.

        Parameters
        ----------
        dt_hr : float, optional
            Simulation time step in hours (default 0.005h = 18s).
        """
        super().__init__()
        self.dt = dt_hr

        self.alpha = ALPHA
        self.beta = BETA

        self.K0_ab = 1.287e12
        self.K0_bc = 1.287e12
        self.K0_ad = 9.043e9

        self.E_A_ab = 9758.3
        self.E_A_bc = 9758.3
        self.E_A_ad = 8560.0

        self.H_R_ab = 4.2
        self.H_R_bc = -11.0
        self.H_R_ad = -41.85

        self.rho = 0.9342
        self.Cp = 3.01
        self.Kw = 4032.0
        self.AR = 0.215
        self.VR = 10.01
        self.mk = 5.0
        self.Cpk = 2.0

        self.CA0 = (5.7 + 4.5) / 2.0 * 1.0
        self.Tin = 130.0

        self.x0 = np.asarray([0.2, 0.5, 120, 120]).reshape(4, 1)

        self.action_space = Box(
            low=self.a_bnd[0], high=self.a_bnd[1], dtype=np.float64
        )

        self.x = self.x0.copy()

    def _denormalize_action(self, u_norm):
        """Convert normalized action to physical units.

        Parameters
        ----------
        u_norm : np.ndarray
            Normalized action in [0, 1], shape (2,) or (2, 1).

        Returns
        -------
        np.ndarray
            Physical action: [F [h^-1], Q_dot [kJ/h]], shape (2,).

        Notes
        -----
        Applies linear scaling:
        - F: [0, 1] → [0, 100] h^-1
        - Q_dot: [0, 1] → [-8500, 0] kJ/h
        """
        F_range = U_NORM_PARAMS["F"]["max"] - U_NORM_PARAMS["F"]["min"]
        Q_range = U_NORM_PARAMS["Q_dot"]["max"] - U_NORM_PARAMS["Q_dot"]["min"]

        F_phys = u_norm[0] * F_range + U_NORM_PARAMS["F"]["min"]
        Q_phys = u_norm[1] * Q_range + U_NORM_PARAMS["Q_dot"]["min"]
        return np.array([F_phys, Q_phys])

    def _normalize_state(self, x_phys):
        """Convert physical state to normalized units.

        Parameters
        ----------
        x_phys : np.ndarray
            Physical state [C_A, C_B, T_R, T_K], shape (4,) or (4, 1).

        Returns
        -------
        np.ndarray
            Normalized state in [0, 1] for each component, shape (4,) or (4, 1).

        Notes
        -----
        Normalization bounds:
        - C_A, C_B: [0, 5.1] mol/L → [0, 1]
        - T_R, T_K: [0, 140] °C → [0, 1]
        """
        x_norm = np.zeros_like(x_phys)

        keys = ["C_A", "C_B", "T_R", "T_K"]
        for i, key in enumerate(keys):
            p_min = Y_NORM_PARAMS[key]["min"]
            p_max = Y_NORM_PARAMS[key]["max"]
            x_norm[i] = (x_phys[i] - p_min) / (p_max - p_min)

        return x_norm

    def reset(self, seed=None, options=None):
        """Reset the environment to initial steady state.

        Parameters
        ----------
        seed : int, optional
            Random seed (currently unused, for Gym compatibility).
        options : dict, optional
            Additional options (currently unused).

        Returns
        -------
        np.ndarray
            Initial normalized state, shape (4,) or (4, 1).
        dict
            Empty info dictionary.
        """
        super().reset(seed=seed)
        self.x = self.x0
        return self._normalize_state(self.x.copy()), {}

    def equations(self, x, u):
        """Compute state derivatives dx/dt using CSTR dynamics.

        Implements the nonlinear ODE system with Arrhenius kinetics for three
        reactions: A→B (exothermic), B→C (endothermic), 2A→D (exothermic).
        Includes jacket cooling dynamics and heat exchange.

        Parameters
        ----------
        x : np.ndarray
            Physical state [C_A, C_B, T_R, T_K], shape (4,).
        u : np.ndarray
            Physical inputs [F, Q_dot], shape (2,).

        Returns
        -------
        np.ndarray
            State derivatives [dC_A/dt, dC_B/dt, dT_R/dt, dT_K/dt], shape (4,).

        Notes
        -----
        Reaction rates use Arrhenius form: k = K0 * exp(-E_A / T_kelvin).
        Heat generation includes reaction enthalpies and jacket heat transfer.
        """
        CA, CB, TR, TK = x[0], x[1], x[2], x[3]
        F, Q_dot = u[0], u[1]

        T_kelvin = TR + 273.15

        k1 = self.beta * self.K0_ab * np.exp(-self.E_A_ab / T_kelvin)
        k2 = self.K0_bc * np.exp(-self.E_A_bc / T_kelvin)
        k3 = self.K0_ad * np.exp(-self.alpha * self.E_A_ad / T_kelvin)
        dCA = F * (self.CA0 - CA) - k1 * CA - k3 * CA**2
        dCB = -F * CB + k1 * CA - k2 * CB

        Q_react = (
            k1 * CA * self.H_R_ab + k2 * CB * self.H_R_bc + k3 * CA**2 * self.H_R_ad
        )

        Q_exchange = (self.Kw * self.AR * (TK - TR)) / (self.rho * self.Cp * self.VR)
        dTR = (Q_react / (-self.rho * self.Cp)) + F * (self.Tin - TR) + Q_exchange

        dTK = (Q_dot + self.Kw * self.AR * (TR - TK)) / (self.mk * self.Cpk)

        return np.array([dCA, dCB, dTR, dTK])

    def step(self, action):
        """Execute one simulation step using RK4 integration.

        Parameters
        ----------
        action : np.ndarray
            Normalized control action [F_norm, Q_dot_norm] in [0, 1], shape (2,).

        Returns
        -------
        np.ndarray
            Normalized next state, shape (4,).
        float
            Reward (0.0, unused in this environment).
        bool
            Terminated flag (False, episode never terminates).
        bool
            Truncated flag (False, no truncation).
        dict
            Empty info dictionary.

        Notes
        -----
        - Action is clipped to [0, 1] bounds before denormalization.
        - Uses 4th-order Runge-Kutta integration with physical units.
        - Physical states are clipped to sanity bounds before normalization.
        """
        action = np.clip(action, self.a_bnd[0], self.a_bnd[1])
        u_phys = self._denormalize_action(action)

        k1 = self.equations(self.x, u_phys)
        k2 = self.equations(self.x + 0.5 * self.dt * k1, u_phys)
        k3 = self.equations(self.x + 0.5 * self.dt * k2, u_phys)
        k4 = self.equations(self.x + self.dt * k3, u_phys)

        self.x = self.x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.x[0] = np.clip(self.x[0], 0, 20)
        self.x[1] = np.clip(self.x[1], 0, 20)
        self.x[2] = np.clip(self.x[2], 0, 200)
        self.x[3] = np.clip(self.x[3], 0, 200)

        return self._normalize_state(self.x.copy()), 0.0, False, False, {}


class NMPC(Mpc[cs.MX]):
    """Nonlinear MPC controller using explicit CSTR dynamics.

    Implements a receding-horizon optimal controller using the known physical
    model equations embedded as symbolic CasADi expressions. Unlike Neural MPC
    which relies on a learned LSTM, this NMPC directly encodes the Arrhenius
    kinetics and RK4 integration into the optimization problem.

    Attributes
    ----------
    horizon : int
        Prediction horizon length (configurable via HORIZON constant).
    discount_factor : float
        Discount factor for cost function (1.0 = no discounting).
    n_context : int
        Number of past time steps for initial condition (1 for NMPC).
    n_inputs : int
        Number of control inputs (2: F, Q_dot).
    n_outputs : int
        Number of system outputs/states (4: C_A, C_B, T_R, T_K).
    pars_init : dict
        Default tuning parameters including state bounds, cost matrices Q and R,
        and slack variable weights.

    Notes
    -----
    - All variables (states, inputs) are normalized to [0, 1] for numerical stability.
    - Dynamics are encoded symbolically using CasADi MX expressions with RK4.
    - Soft constraints on state bounds are enforced via slack variables with penalty.
    - Hard constraints are applied to specific state indices (C_A, C_B, T_K).
    - Cost function includes tracking error, control effort, and terminal cost.
    - Uses IPOPT solver with custom tolerance settings for real-time feasibility.

    References
    ----------
    .. [1] Fiedler, F. et al. (2023). "do-mpc: Towards FAIR nonlinear and
           robust model predictive control."
    """

    horizon = HORIZON
    discount_factor = 1.0
    n_context = 1
    n_inputs = 2
    n_outputs = 4

    @staticmethod
    def _norm_val(val, key):
        return (val - Y_NORM_PARAMS[key]["min"]) / (
            Y_NORM_PARAMS[key]["max"] - Y_NORM_PARAMS[key]["min"]
        )

    pars_init = {
        "x_lb": np.asarray(
            [
                _norm_val(0.1, "C_A"),
                _norm_val(0.1, "C_B"),
                _norm_val(50.0, "T_R"),
                _norm_val(50.0, "T_K"),
            ],
            dtype=float,
        ),
        "x_ub": np.asarray(
            [
                _norm_val(2.0, "C_A"),
                _norm_val(2.0, "C_B"),
                _norm_val(135.0, "T_R"),
                _norm_val(140.0, "T_K"),
            ],
            dtype=float,
        ),
        "x_lb_f": np.asarray(
            [
                _norm_val(0.1, "C_A"),
                _norm_val(0.1, "C_B"),
                _norm_val(50.0, "T_R"),
                _norm_val(50.0, "T_K"),
            ],
            dtype=float,
        ),
        "x_ub_f": np.asarray(
            [
                _norm_val(2.0, "C_A"),
                _norm_val(2.0, "C_B"),
                _norm_val(135.0, "T_R"),
                _norm_val(140.0, "T_K"),
            ],
            dtype=float,
        ),
        "b": np.asarray([0, 0, 0, 0], dtype=float),
        "Q": np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0, 1.0, 0.0, 0.0],
                [0, 0.0, 1e-6, 0.0],
                [0, 0.0, 0.0, 1e-6],
            ],
            dtype=float,
        ),
        "R": np.asarray([[1, 0], [0, 1e-4]], dtype=float),
        "w": np.asarray([0, 0, 1e2, 0], dtype=np.float64),
        "x_scaling": np.asarray([1, 1, 1, 1], dtype=float),
        "u_scaling": np.asarray([1, 1], dtype=float),
    }

    def __init__(self) -> None:
        """Initialize NMPC problem with explicit CSTR dynamics.

        Sets up the optimal control problem including:
        - Decision variables: states x, inputs u, slack variables s1, s2
        - Parameters: state bounds, cost matrices Q/R, setpoint SP
        - Symbolic CSTR dynamics with RK4 integration in CasADi
        - Soft state constraints via slack variables
        - Quadratic stage and terminal costs

        The NMPC uses multi-shooting with explicit symbolic dynamics. Each
        shooting interval applies one RK4 integration step of the CSTR ODEs,
        including Arrhenius kinetics and heat transfer equations. The dynamics
        handle denormalization (MPC units → physical), RK4 integration, and
        renormalization (physical → MPC units).

        Notes
        -----
        - Dynamics are built symbolically using CasADi MX expressions
        - RK4 integration uses dt=0.005h (18 seconds) to match environment
        - Slack variable weights prevent state constraint violations
        - Hard constraints enforced on C_A, C_B, and T_K indices
        - Control effort penalized via delta_u in cost function
        """
        N = self.horizon
        gamma = self.discount_factor

        nx, nu = CSTRSystem.nx, CSTRSystem.nu
        a_bnd = CSTRSystem.a_bnd_mpc

        nlp = Nlp(sym_type="MX")
        super().__init__(
            nlp,
            N,
            tuning_parameters=self.pars_init,
            n_context=self.n_context,
            shooting="multi",
            neural=False
        )

        x_lb = self.parameter("x_lb", (nx, 1))
        x_ub = self.parameter("x_ub", (nx, 1))
        x_lb_f = self.parameter("x_lb_f", (nx, 1))
        x_ub_f = self.parameter("x_ub_f", (nx, 1))
        b = self.parameter("b", (nx, 1))
        Q = self.parameter("Q", (nx, nx))
        R = self.parameter("R", (nu, nu))
        SP = self.parameter("SP", (nx, 1))
        w = self.parameter("w", (nx, 1))
        x_scaling = self.parameter("x_scaling", (nx, 1))
        u_scaling = self.parameter("u_scaling", (nu, 1))

        x, _ = self.state("x", nx, bound_initial=False)
        u, u_exp, u0 = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s1, _, _ = self.variable("s1", (nx, N), lb=0)
        s2, _, _ = self.variable("s2", (nx, 1), lb=0)

        du = []
        du.append(u_exp[:, 0] - u0[:, -1])
        for t in range(1, N):
            du.append((u_exp[:, t] - u_exp[:, t - 1]) * u_scaling)
        du = cs.hcat(du)

        x_sym = cs.MX.sym("x_in", nx)
        u_sym = cs.MX.sym("u_in", nu)
        CA_p = (
            x_sym[0] * (Y_NORM_PARAMS["C_A"]["max"] - Y_NORM_PARAMS["C_A"]["min"])
            + Y_NORM_PARAMS["C_A"]["min"]
        )
        CB_p = (
            x_sym[1] * (Y_NORM_PARAMS["C_B"]["max"] - Y_NORM_PARAMS["C_B"]["min"])
            + Y_NORM_PARAMS["C_B"]["min"]
        )
        TR_p = (
            x_sym[2] * (Y_NORM_PARAMS["T_R"]["max"] - Y_NORM_PARAMS["T_R"]["min"])
            + Y_NORM_PARAMS["T_R"]["min"]
        )
        TK_p = (
            x_sym[3] * (Y_NORM_PARAMS["T_K"]["max"] - Y_NORM_PARAMS["T_K"]["min"])
            + Y_NORM_PARAMS["T_K"]["min"]
        )
        x_phys = cs.vertcat(CA_p, CB_p, TR_p, TK_p)

        F_p = (
            u_sym[0] * (U_NORM_PARAMS["F"]["max"] - U_NORM_PARAMS["F"]["min"])
            + U_NORM_PARAMS["F"]["min"]
        )
        Q_p = (
            u_sym[1] * (U_NORM_PARAMS["Q_dot"]["max"] - U_NORM_PARAMS["Q_dot"]["min"])
            + U_NORM_PARAMS["Q_dot"]["min"]
        )

        dt_hr = 0.005
        alpha, beta = 1.0, 1.0
        K0_ab, K0_bc, K0_ad = 1.287e12, 1.287e12, 9.043e9
        E_A_ab, E_A_bc, E_A_ad = 9758.3, 9758.3, 8560.0
        H_R_ab, H_R_bc, H_R_ad = 4.2, -11.0, -41.85
        rho, Cp, Kw, AR, VR = 0.9342, 3.01, 4032.0, 0.215, 10.01
        mk, Cpk = 5.0, 2.0
        CA0, Tin = (5.7 + 4.5) / 2.0, 130.0

        def get_derivatives(x_p, f_in, q_in):
            ca, cb, tr, tk = x_p[0], x_p[1], x_p[2], x_p[3]
            T_kelvin = tr + 273.15

            k1 = beta * K0_ab * cs.exp(-E_A_ab / T_kelvin)
            k2 = K0_bc * cs.exp(-E_A_bc / T_kelvin)
            k3 = K0_ad * cs.exp(-alpha * E_A_ad / T_kelvin)

            dCA = f_in * (CA0 - ca) - k1 * ca - k3 * ca**2
            dCB = -f_in * cb + k1 * ca - k2 * cb

            Q_react = k1 * ca * H_R_ab + k2 * cb * H_R_bc + k3 * ca**2 * H_R_ad
            Q_exchange = (Kw * AR * (tk - tr)) / (rho * Cp * VR)
            dTR = (Q_react / (-rho * Cp)) + f_in * (Tin - tr) + Q_exchange

            dTK = (q_in + Kw * AR * (tr - tk)) / (mk * Cpk)
            return cs.vertcat(dCA, dCB, dTR, dTK)

        k1 = get_derivatives(x_phys, F_p, Q_p)
        k2 = get_derivatives(x_phys + 0.5 * dt_hr * k1, F_p, Q_p)
        k3 = get_derivatives(x_phys + 0.5 * dt_hr * k2, F_p, Q_p)
        k4 = get_derivatives(x_phys + dt_hr * k3, F_p, Q_p)

        x_phys_next = x_phys + (dt_hr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        CA_n_next = (x_phys_next[0] - Y_NORM_PARAMS["C_A"]["min"]) / (
            Y_NORM_PARAMS["C_A"]["max"] - Y_NORM_PARAMS["C_A"]["min"]
        )
        CB_n_next = (x_phys_next[1] - Y_NORM_PARAMS["C_B"]["min"]) / (
            Y_NORM_PARAMS["C_B"]["max"] - Y_NORM_PARAMS["C_B"]["min"]
        )
        TR_n_next = (x_phys_next[2] - Y_NORM_PARAMS["T_R"]["min"]) / (
            Y_NORM_PARAMS["T_R"]["max"] - Y_NORM_PARAMS["T_R"]["min"]
        )
        TK_n_next = (x_phys_next[3] - Y_NORM_PARAMS["T_K"]["min"]) / (
            Y_NORM_PARAMS["T_K"]["max"] - Y_NORM_PARAMS["T_K"]["min"]
        )

        x_next_norm = cs.vertcat(CA_n_next, CB_n_next, TR_n_next, TK_n_next)

        F_casadi = cs.Function("F_model", [x_sym, u_sym], [x_next_norm])

        self.set_dynamics(F=F_casadi, n_in=nu, n_out=nx)

        xlb_rep = cs.repmat(x_lb, 1, N)
        xub_rep = cs.repmat(x_ub, 1, N)
        hard_indices = [0, 1, 3]
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
        S = (gamma**N) * (cs.bilin(Q, e_N) + w.T @ s2)

        Lt = 0.0

        for k in range(self.n_context - 1, self.n_context - 1 + N):
            e_k = x[:, k] - SP
            e_k = e_k * x_scaling
            k_abs = k - (self.n_context - 1)
            Lt += (gamma**k_abs) * (cs.bilin(Q, e_k))
            Lt += (gamma**k_abs) * (cs.bilin(R, du[:, k_abs]))
            Lt += (gamma**k_abs) * (w.T @ s1[:, k_abs])

        self.minimize(S + Lt)

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
        """Get the most recent setpoint for a given timestep.

        Parameters
        ----------
        timestep : int
            Current simulation timestep.

        Returns
        -------
        np.ndarray
            Setpoint vector [C_A, C_B, T_R, T_K] in physical units, shape (4,).

        Notes
        -----
        Uses a piecewise constant schedule. Returns the most recent setpoint
        whose timestamp does not exceed the current timestep.
        """
        idx = max(
            i
            for i in range(len(setpoint_timestamps))
            if setpoint_timestamps[i] <= timestep
        )
        return np.asarray(setpoint_values[idx])

    def normalize_vector(v_phys):
        """Normalize a state vector to [0, 1] using Y_NORM_PARAMS.

        Parameters
        ----------
        v_phys : np.ndarray
            Physical state [C_A, C_B, T_R, T_K], shape (4,) or (4, 1).

        Returns
        -------
        np.ndarray
            Normalized state in [0, 1], shape (4,) or (4, 1).

        Examples
        --------
        >>> v_phys = np.array([1.5, 1.0, 100.0, 100.0])
        >>> v_norm = normalize_vector(v_phys)
        >>> v_norm
        array([0.294, 0.196, 0.714, 0.714])
        """
        v_norm = np.zeros_like(v_phys)
        keys = ["C_A", "C_B", "T_R", "T_K"]
        for i, key in enumerate(keys):
            p_min = Y_NORM_PARAMS[key]["min"]
            p_max = Y_NORM_PARAMS[key]["max"]
            v_norm[i] = (v_phys[i] - p_min) / (p_max - p_min)
        return v_norm

    MAX_SEED = np.iinfo(np.uint32).max + 1

    def mk_seed(rng: np.random.Generator) -> int:
        """Generate a random seed from a NumPy random generator.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random generator instance.

        Returns
        -------
        int
            Random seed in the range [0, 2**32).

        Examples
        --------
        >>> rng = np.random.default_rng(42)
        >>> seed = mk_seed(rng)
        >>> 0 <= seed < 2**32
        True
        """
        return int(rng.integers(MAX_SEED))

    simulation_time = NUM_ITER
    mpc = NMPC()
    env = CSTRSystem()

    setpoint_values = [[[1.5], [1.0], [100.0], [100.0]]]
    setpoint_timestamps = [0]

    state_indices = [0, 1, 2, 3]

    rng = np.random.default_rng(69)
    state, _ = env.reset(seed=mk_seed(rng), options=None)

    X, U, SP, X_pred = [state], [], [], []
    if WARMUP_TYPE == "X0":
        state_context = np.tile(state.T, (mpc.n_context, 1))
    else:
        state_context = np.zeros((mpc.n_context, CSTRSystem.nx))
    action_context = np.zeros((mpc.n_context, CSTRSystem.nu))

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
                sp_phys = get_current_setpoint(timestep)
                sp = normalize_vector(sp_phys)

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
                state_context = np.vstack([state_context, obs.T])[-mpc.n_context :]
                action_context = np.vstack([action_context, np.asarray(u_opt).T])[
                    -mpc.n_context :
                ]

                if mpc._last_solution is not None:
                    X_pred.append(
                        np.asarray(mpc._last_solution.vals["x"][:, mpc._n_context])
                    )
                else:
                    X_pred.append(
                        np.asarray([np.nan, np.nan, np.nan, np.nan]).reshape(4, 1)
                    )

                X.append(obs.copy())
                U.append(u_opt)
                SP.append(sp.copy())
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
    save_dir = project_root / "examples" / "CSTR" / "data" / "NONLINEAR" / experiment_id
    save_dir.mkdir(parents=True, exist_ok=True)

    df_system = pd.DataFrame(
        {
            "step": np.arange(len(U)),
            "C_A": X[1:, 0],
            "C_B": X[1:, 1],
            "T_R": X[1:, 2],
            "T_K": X[1:, 3],
            "C_A_pred": X_pred[:, 0],
            "C_B_pred": X_pred[:, 1],
            "T_R_pred": X_pred[:, 2],
            "T_K_pred": X_pred[:, 3],
            "F": U[:, 0],
            "Q_dot": U[:, 1],
            "sp_C_A": SP[:, 0],
            "sp_C_B": SP[:, 1],
            "sp_T_R": SP[:, 2],
            "sp_T_K": SP[:, 3],
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

    keys = ["C_A", "C_B", "T_R", "T_K"]
    for i, key in enumerate(keys):
        p_min, p_max = Y_NORM_PARAMS[key]["min"], Y_NORM_PARAMS[key]["max"]
        X[:, i] = X[:, i] * (p_max - p_min) + p_min
        X_pred[:, i] = X_pred[:, i] * (p_max - p_min) + p_min
        SP[:, i] = SP[:, i] * (p_max - p_min) + p_min

    u_keys = ["F", "Q_dot"]
    for i, key in enumerate(u_keys):
        p_min, p_max = U_NORM_PARAMS[key]["min"], U_NORM_PARAMS[key]["max"]
        U[:, i] = U[:, i] * (p_max - p_min) + p_min

    fig, axs = plt.subplots(6, 1, constrained_layout=True, sharex=True)
    fig.suptitle("System Response")
    timesteps = np.arange(stop=X.shape[0] * 0.005, step=0.005)

    axs[0].plot(timesteps, X[:, 1], label=r"$C_B$")
    axs[0].plot(timesteps[1:], SP[:, 1], linestyle=":", label=r"SP $C_B$")
    axs[0].plot(timesteps[1:], X_pred[:, 1], linestyle=":", label=r"$C_B$ Pred.")
    axs[1].plot(timesteps, X[:, 0], label=r"$C_A$")
    axs[1].plot(timesteps[1:], SP[:, 0], linestyle=":", label=r"SP $C_A$")
    axs[1].plot(timesteps[1:], X_pred[:, 0], linestyle=":", label=r"$C_A$ Pred.")
    axs[2].plot(timesteps, X[:, 2], label=r"$T_R$")
    axs[2].plot(timesteps[1:], SP[:, 2], linestyle=":", label=r"SP $T_R$")
    axs[2].plot(timesteps[1:], X_pred[:, 2], linestyle=":", label=r"$T_R$ Pred.")
    axs[3].plot(timesteps, X[:, 3], label=r"$T_K$")
    axs[3].plot(timesteps[1:], SP[:, 3], linestyle=":", label=r"SP $T_K$")
    axs[3].plot(timesteps[1:], X_pred[:, 3], linestyle=":", label=r"$T_K$ Pred.")
    axs[4].step(timesteps[1:], U[:, 0], where="post", label=r"$F$")
    axs[5].step(timesteps[1:], U[:, 1], where="post", label=r"$\dot{Q}$")

    lb_states = [0.1, 0.1, 50.0, 50.0]
    ub_states = [2.0, 2.0, 135.0, 140.0]
    axs[0].axhline(lb_states[1], linestyle="--")
    axs[0].axhline(ub_states[1], linestyle="--")
    axs[1].axhline(lb_states[0], linestyle="--")
    axs[1].axhline(ub_states[0], linestyle="--")
    axs[2].axhline(lb_states[2], linestyle="--")
    axs[2].axhline(ub_states[2], linestyle="--")
    axs[3].axhline(lb_states[3], linestyle="--")
    axs[3].axhline(ub_states[3], linestyle="--")

    for ax, label in zip(
        axs,
        (
            r"$C_B$ [mol/L]",
            r"$C_A$ [mol/L]",
            r"$T_R$ [$^\circ$C]",
            r"$T_K$ [$^\circ$C]",
            r"$F$[$h^{-1}$]",
            r"$\dot{Q}$[kJ/h]",
        ),
    ):
        ax.set_ylabel(label)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("time [h]")

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
