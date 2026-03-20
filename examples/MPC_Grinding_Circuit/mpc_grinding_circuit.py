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

"""Model Predictive Control for Ball-Mill Grinding Circuit.

Implements a constrained MPC controller for a 4×4 MIMO grinding circuit
based on:
    Chen et al. (2007), "Application of model predictive control in ball mill
    grinding circuit", Minerals Engineering 20(11):1099-1108.
    doi:10.1016/j.mineng.2007.04.007

The system has:
- Controlled outputs: Sp (particle size), Dm (mill solids concentration),
  Fc (circulating load), Ls (sump level)
- Manipulated inputs: Ff (fresh ore feed), Fm (mill water flow),
  Fd (dilution water flow), Vp (pump speed)

Usage
-----
Run from the examples/MPC_Grinding_Circuit/ directory::

    python mpc_grinding_circuit.py

Output figures are saved to ``figures_grinding/``.
"""

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

import casadi as cs
import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

from neuralmpcx import Nlp
from neuralmpcx.util.control import TransferFunctionTerm, dlqr, mimo_tf2ss
from neuralmpcx.util.estimators import AugmentedKalmanFilter
from neuralmpcx.wrappers import Mpc

class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    handlers=[_TqdmLoggingHandler()],
)


class NtiSystem(gym.Env):
    """Gymnasium environment for MIMO 4×4 grinding circuit process model.

    Dynamics:  Y = G * ΔU, where deviations ΔY = Y - Yss, ΔU = U - Uss.

    Attributes
    ----------
    ny, nu : int
        Number of controlled / manipulated variables.
    y_offset, u_offset : np.ndarray
        Steady-state output / input values.
    """

    ny = 4
    nu = 4

    y_offset = np.array([71.3, 79, 151, 1.15], dtype=float).reshape(ny, 1)
    u_offset = np.array([70, 8.5, 25, 42.5],   dtype=float).reshape(nu, 1)

    Sp_max = 100;  Dm_max = 100;  Fc_max = 200
    Ls_min = 0.5;  Ls_max = 2.4

    dFf_lim_hi = 80  - u_offset[0, 0];  dFf_lim_lo = 65  - u_offset[0, 0]
    dFm_lim_hi = 15  - u_offset[1, 0];  dFm_lim_lo = 2   - u_offset[1, 0]
    dFd_lim_hi = 45  - u_offset[2, 0];  dFd_lim_lo = 5   - u_offset[2, 0]
    dVp_lim_hi = 50  - u_offset[3, 0];  dVp_lim_lo = 35  - u_offset[3, 0]

    a_bnd_mpc = (
        np.array([[dFf_lim_lo], [dFm_lim_lo], [dFd_lim_lo], [dVp_lim_lo]], dtype=np.float64),
        np.array([[dFf_lim_hi], [dFm_lim_hi], [dFd_lim_hi], [dVp_lim_hi]], dtype=np.float64),
    )
    a_bnd = (
        np.array([[65], [2],  [5],  [35]], dtype=np.float64),
        np.array([[80], [15], [45], [50]], dtype=np.float64),
    )

    action_space = Box(*a_bnd_mpc, (nu, 1), np.float64)

    output_noise_low  = np.array([-0.01 * Sp_max, -0.01 * Dm_max, -0.01 * Fc_max, -0.01 * Ls_max])
    output_noise_high = np.array([ 0.01 * Sp_max,  0.01 * Dm_max,  0.01 * Fc_max,  0.01 * Ls_max])
    use_meas_noise      = True
    meas_state_directly = True
    saturate_level      = True

    def __init__(self, Ad, Bd, Cd, Dd):
        self.Ad = np.asarray(Ad, dtype=np.float64)
        self.Bd = np.asarray(Bd, dtype=np.float64)
        self.Cd = np.asarray(Cd, dtype=np.float64)
        self.Dd = np.asarray(Dd, dtype=np.float64)
        self.nx = self.Ad.shape[0]
        self.x  = np.zeros((self.nx, 1), dtype=np.float64)
        self.Y  = np.zeros((self.ny, 1), dtype=np.float64)
        self.np_random = np.random.default_rng()

    def _project_state_to_level_bound(self, x_next, dev_u, dev_bound_value):
        C4  = self.Cd[3:4, :]
        dev_u = np.asarray(dev_u, dtype=np.float64).reshape(-1, 1)
        D4u   = float((self.Dd[3:4, :] @ dev_u).flatten()[0])
        y4    = float((C4 @ x_next + D4u).flatten()[0])
        err   = dev_bound_value - y4
        denom = float((C4 @ C4.T).flatten()[0])
        if denom > 0.0:
            x_next = x_next + C4.T * (err / denom)
        return x_next

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.x = np.zeros((self.nx, 1), dtype=np.float64)
        self.Y = self.y_offset.copy()
        return self.Y.copy(), {"x": self.x.copy()}

    def step(self, action):
        if hasattr(action, "full"):
            action = action.full()
        u     = np.clip(np.asarray(action, dtype=np.float64).reshape(self.nu, 1),
                        self.a_bnd[0], self.a_bnd[1])
        dev_u = (u - self.u_offset).reshape(self.nu, 1)

        x_next = self.Ad @ self.x + self.Bd @ dev_u
        y      = self.Cd @ x_next + self.Dd @ dev_u + self.y_offset

        if self.saturate_level:
            y4 = float(y[3, 0])
            if y4 < self.Ls_min:
                x_next = self._project_state_to_level_bound(
                    x_next, dev_u, self.Ls_min - self.y_offset[3, 0])
                y[3, 0] = self.Ls_min
            elif y4 > self.Ls_max:
                x_next = self._project_state_to_level_bound(
                    x_next, dev_u, self.Ls_max - self.y_offset[3, 0])
                y[3, 0] = self.Ls_max

        y = self.Cd @ x_next + self.Dd @ dev_u + self.y_offset

        if self.use_meas_noise:
            y = y + self.np_random.uniform(
                self.output_noise_low, self.output_noise_high).reshape(self.ny, 1)

        self.x = x_next.copy()
        self.Y = y
        return self.Y.copy(), 0.0, False, False, {"x": self.x.copy()}


class LinearMpc(Mpc[cs.MX]):
    """Linear MPC controller for the grinding circuit.

    Multi-shooting MPC with:
    - Linear discrete-time dynamics + augmented Kalman filter biases
    - Soft constraints on controlled variables (slacks s1, s2)
    - Rate-of-change constraints on manipulated variables
    - Terminal cost based on discrete LQR solution
    """

    pred_horizon   = 30
    contr_horizon  = 5
    discount_factor = 1.0
    n_context      = 1
    n_inputs       = NtiSystem.nu
    n_outputs      = NtiSystem.ny
    u_offset = np.array([70, 8.5, 25, 42.5], dtype=float).reshape(NtiSystem.nu, 1)
    y_offset = np.array([71.3, 79, 151, 1.15], dtype=float).reshape(NtiSystem.ny, 1)

    fixed_pars = {
        "Y_lb":   np.asarray([70, 78, 140, 0.5],  dtype=float),
        "Y_ub":   np.asarray([74, 84, 180, 2.4],  dtype=float),
        "Y_lb_f": np.asarray([70, 78, 140, 0.5],  dtype=float),
        "Y_ub_f": np.asarray([74, 84, 180, 2.4],  dtype=float),
        "Q": np.asarray([[5,0,0,0],[0,0.8,0,0],[0,0,0.7,0],[0,0,0,0.3]], dtype=float),
        "R": np.asarray([[1,0,0,0],[0,0.6,0,0],[0,0,0.5,0],[0,0,0,0.5]], dtype=float),
        "w": np.asarray(0, dtype=np.float64),
    }

    def __init__(self, A_d, B_d, C_d, D_d, Ts_mul=0.5):
        N     = self.pred_horizon
        M     = self.contr_horizon
        gamma = self.discount_factor

        A_d = np.asarray(A_d, dtype=np.float64)
        B_d = np.asarray(B_d, dtype=np.float64)
        C_d = np.asarray(C_d, dtype=np.float64)
        D_d = np.asarray(D_d, dtype=np.float64)
        nx  = A_d.shape[0]
        nu  = B_d.shape[1]
        ny  = C_d.shape[0]
        a_bnd = NtiSystem.a_bnd_mpc

        nlp = Nlp(sym_type="MX")
        super().__init__(
            nlp,
            prediction_horizon=N,
            control_horizon=M,
            tuning_parameters=self.fixed_pars,
            n_context=self.n_context,
            shooting="multi",
            neural=False,
        )

        Y_lb   = self.parameter("Y_lb",   (ny, 1))
        Y_ub   = self.parameter("Y_ub",   (ny, 1))
        Y_lb_f = self.parameter("Y_lb_f", (ny, 1))
        Y_ub_f = self.parameter("Y_ub_f", (ny, 1))
        Q      = self.parameter("Q",  (ny, ny))
        R      = self.parameter("R",  (nu, nu))
        SP     = self.parameter("SP", (ny, 1))
        w      = self.parameter("w",  (1, 1))

        du_bias = self.parameter("du_bias", (nu, 1))
        dy_bias = self.parameter("dy_bias", (ny, 1))

        x, _          = self.state("x", nx, bound_initial=False)
        dev_u, dev_u_exp, u0 = self.action("du", nu, lb=a_bnd[0], ub=a_bnd[1])

        ddev_u = [dev_u_exp[:, 0] - u0]
        for t in range(1, M):
            ddev_u.append(dev_u_exp[:, t] - dev_u_exp[:, t - 1])
        ddev_u = cs.hcat(ddev_u)

        s1, _, _ = self.variable("s1", (ny, N - 1), lb=0)
        s2, _, _ = self.variable("s2", (ny, 1),     lb=0)

        x_sym    = cs.MX.sym("x",    nx)
        dev_u_sym = cs.MX.sym("dev_u", nu)
        Ad_cs = cs.DM(A_d);  Bd_cs = cs.DM(B_d)
        Cd_cs = cs.DM(C_d);  Dd_cs = cs.DM(D_d)

        F = cs.Function("F", [x_sym, dev_u_sym],
                        [Ad_cs @ x_sym + Bd_cs @ dev_u_sym],
                        {"allow_free": True, "cse": True})
        self.set_dynamics(F)

        Y_cols = [Cd_cs @ x[:, t] + Dd_cs @ (dev_u_exp[:, t-1] + du_bias) + dy_bias
                  for t in range(1, N)]
        YN = Cd_cs @ x[:, N] + Dd_cs @ (dev_u_exp[:, N-1] + du_bias) + dy_bias
        Y  = cs.hcat(Y_cols + [YN]) + cs.repmat(cs.DM(self.y_offset), 1, N)

        self.constraint("Y_lb",   cs.repmat(Y_lb,   1, N-1) - s1, "<=", Y[:, :-1])
        self.constraint("Y_ub",   Y[:, :-1], "<=", cs.repmat(Y_ub,   1, N-1) + s1)
        self.constraint("Y_lb_f", Y_lb_f - s2,     "<=", Y[:, -1])
        self.constraint("Y_ub_f", Y[:, -1],         "<=", Y_ub_f + s2)

        self.constraint("delta_u1", cs.fabs(ddev_u[0, :]), "<=", 3 * Ts_mul)
        self.constraint("delta_u2", cs.fabs(ddev_u[1, :]), "<=", 3 * Ts_mul)
        self.constraint("delta_u3", cs.fabs(ddev_u[2, :]), "<=", 5 * Ts_mul)
        self.constraint("delta_u4", cs.fabs(ddev_u[3, :]), "<=", 4 * Ts_mul)

        auxY = cs.MX(1, N)
        for t in range(N - 1):
            auxY[0, t] = cs.bilin(Q, Y[:, t] - SP) + w * cs.sum1(s1[:, t])

        auxU = cs.MX(1, M)
        for t in range(M):
            auxU[0, t] = cs.bilin(R, ddev_u[:, t])

        gamN = cs.DM(gamma ** np.arange(N)).T
        gamM = cs.DM(gamma ** np.arange(M)).T

        Qy = self.fixed_pars["Q"]
        Rv = self.fixed_pars["R"]
        Qx = np.asarray(C_d).T @ Qy @ np.asarray(C_d)
        _, P = dlqr(A_d, B_d, Qx, Rv)
        H_s = cs.DM(C_d @ P @ C_d.T)
        S   = cs.DM(gamma**N) * (cs.bilin(H_s, Y[:, -1] - SP) + w * cs.sum1(s2))

        self.minimize(cs.sum2(gamN * auxY) + cs.sum2(gamM * auxU) + S)

        opts = {
            "print_time": False,
            "ipopt": {
                "linear_solver": "mumps",
                "mumps_pivtol": 1e-3,
                "mumps_pivtolmax": 0.5,
                # Uncomment below to use HSL solvers (faster for large problems)
                # "linear_solver": "ma27",  # or "ma57", "ma97"
                # "hsllib": os.path.expanduser("/home/coinhsl/lib/libcoinhsl.so"),
                "nlp_scaling_method": "gradient-based", # NLP scaling is necessary when mixing signals of very different magnitudes.
                "nlp_scaling_max_gradient": 100.0,
                "max_iter": 200,
                "sb": "yes",
                "print_level": 0,
                "tol": 1e-4,
                "acceptable_tol": 1e-2,
            },
        }
        self.init_solver(opts, solver="ipopt")


def get_current_setpoint(timestep, setpoint_values, setpoint_timestamps):
    """Return setpoint for the current timestep."""
    idx = max(i for i in range(len(setpoint_timestamps))
              if setpoint_timestamps[i] <= timestep)
    return setpoint_values[:, idx]


def mk_seed(rng):
    return int(rng.integers(np.iinfo(np.uint32).max + 1))


if __name__ == "__main__":
    simulation_time = np.int64(240)

    TF = TransferFunctionTerm
    G = {
        (0, 0): [TF(gain=-0.58, delay=41.0, time_constants=[83.0])],
        (0, 1): [TF(gain=0.97,  delay=40.0,  time_constants=[125.0, 195.0]),
                 TF(gain=-0.97 * 1.08, delay=272.0, time_constants=[125.0, 195.0])],
        (0, 2): [TF(gain=+0.67, delay=8.0,   time_constants=[20.0, 92.0]),
                 TF(gain=-0.67 * 1.07, delay=8.0 + 214.0, time_constants=[20.0, 92.0])],
        (0, 3): [TF(gain=0.50, delay=2.0,   time_constants=[18.0])],
        (1, 0): [TF(gain=0.62, time_constants=[123.0])],
        (1, 1): [TF(gain=-1.75, time_constants=[118.0])],
        (1, 2): [TF(gain=0.51, delay=87.0,  time_constants=[81.0, 182.0])],
        (1, 3): [TF(gain=0.64, delay=9.0,   time_constants=[137.0])],
        (2, 0): [TF(gain=2.61, delay=45.0,  time_constants=[110.0])],
        (2, 1): [TF(gain=9.52, delay=93.0,  time_constants=[98.0, 137.0])],
        (2, 2): [TF(gain=2.83, delay=8.0,   time_constants=[128.0])],
        (2, 3): [TF(gain=2.81, delay=5.0,   time_constants=[108.0])],
        (3, 0): [TF(gain=0.001, delay=30.0, time_constants=[150.0], has_integrator=True)],
        (3, 1): [TF(gain=0.011, delay=30.0, time_constants=[100.0], has_integrator=True)],
        (3, 2): [TF(gain=0.032, has_integrator=True)],
        (3, 3): [TF(gain=-0.031, has_integrator=True)],
    }
    Ts_mul = 0.5
    ss = mimo_tf2ss(G, ny=4, nu=4, Ts=Ts_mul * 60.0, pade_order=2)
    Ad, Bd, Cd, Dd = ss.Ad, ss.Bd, ss.Cd, ss.Dd
    Ts = ss.Ts

    mpc = LinearMpc(Ad, Bd, Cd, Dd)
    env = NtiSystem(Ad, Bd, Cd, Dd)
    env.use_meas_noise = True

    nx = Ad.shape[0];  ny = Cd.shape[0];  nu = Bd.shape[1]

    Q_x  = np.eye(nx) * 0.1
    Q_du = np.eye(nu) * 0.01
    Q_dy = np.eye(ny) * 0.01
    R_kal = np.eye(ny) * 1.0
    kf = AugmentedKalmanFilter(Ad, Bd, Cd, Dd,
                               Q_x=Q_x, Q_du=Q_du, Q_dy=Q_dy, R=R_kal)

    setpoint_values     = np.array([[72,73,72,71],[79,79,79,79],
                                    [151,151,151,151],[1.15,1.15,1.15,1.15]],
                                   dtype=float).reshape(4, 4)
    setpoint_timestamps = [0, 60, 120, 180]
    state_indices       = list(range(env.nx))

    rng = np.random.default_rng(69)
    y, info = env.reset(seed=mk_seed(rng))

    state = kf.x_est
    if NtiSystem.meas_state_directly:
        state_context = np.tile(state.T, (mpc.n_context, 1))
    else:
        state_context = np.zeros((mpc.n_context, NtiSystem.ny))
    action_context = np.zeros((mpc.n_context, NtiSystem.nu))

    X_true, X_est, Y, U, SP = [info["x"]], [state], [y], [], []
    DU_BIAS, DY_BIAS = [], []
    vals0         = None
    store_solution = True
    input_bias     = None
    timestep       = 0

    with tqdm(total=simulation_time, desc="MPC Simulation", unit="step", ncols=80, colour="green") as pbar:
        for t in range(simulation_time):
            setpoint     = get_current_setpoint(timestep, setpoint_values, setpoint_timestamps)
            dynamic_pars = kf.get_mpc_biases()

            dev_u_opt = mpc.solve_mpc(
                state, state_context, state_indices, action_context,
                setpoint, input_bias, vals0, store_solution, dynamic_pars)

            dev_u_opt_np = np.array(dev_u_opt.full(), dtype=np.float64).reshape(-1, 1)
            u_opt = dev_u_opt_np + LinearMpc.u_offset
            y, _, _, _, info = env.step(u_opt)

            kf.predict(dev_u_opt_np)
            kf.update(y - LinearMpc.y_offset, dev_u_opt_np)

            state         = kf.x_est
            state_context = state.T
            action_context = dev_u_opt_np.T

            X_true.append(info["x"])
            X_est.append(state)
            Y.append(y)
            U.append(u_opt)
            SP.append(setpoint)
            DU_BIAS.append(kf.du_bias_est.copy())
            DY_BIAS.append(kf.dy_bias_est.copy())
            timestep += 1
            pbar.update(1)

    X_true  = np.hstack(X_true)
    X_est   = np.hstack(X_est)
    Y       = np.hstack(Y)
    U       = np.hstack(U).reshape(mpc.n_inputs, -1)
    SP      = np.vstack(SP).T.reshape(mpc.n_outputs, -1)
    DU_BIAS = np.hstack(DU_BIAS)
    DY_BIAS = np.hstack(DY_BIAS)

    # ── Publication-quality figures ─────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif':  ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.format': 'svg',
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
        'lines.linewidth': 1.5, 'axes.linewidth': 0.8,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.top': True, 'ytick.right': True,
    })

    COLOR_PV = '#000000';  COLOR_SP = '#E69F00'
    COLOR_CONSTRAINT = '#D55E00';  COLOR_FEASIBLE = '#56B4E9'
    COLORS_BIAS = ['#000000', '#E69F00', '#56B4E9', '#009E73']

    OUTPUT_DIR = Path("figures_grinding")
    OUTPUT_DIR.mkdir(exist_ok=True)

    timesteps = np.arange(Y.shape[1]);  t_u = timesteps[:-1]
    Y_lb_vals = LinearMpc.fixed_pars["Y_lb"]
    Y_ub_vals = LinearMpc.fixed_pars["Y_ub"]
    U_lb_vals = NtiSystem.a_bnd[0].flatten()
    U_ub_vals = NtiSystem.a_bnd[1].flatten()

    y_labels = (r'$S_p$ [%-200mesh]', r'$D_m$ [% solids]', r'$F_c$ [t/h]', r'$L_s$ [m]')
    u_labels = (r'$F_f$ [t/h]', r'$F_m$ [m$^3$/h]', r'$F_d$ [m$^3$/h]', r'$V_p$ [Hz]')
    du_labels = (r'$\Delta F_f$', r'$\Delta F_m$', r'$\Delta F_d$', r'$\Delta V_p$')
    dy_labels = (r'$\Delta S_p$', r'$\Delta D_m$', r'$\Delta F_c$', r'$\Delta L_s$')

    # Figure 1: Controlled Variables
    fig_cv, axs_cv = plt.subplots(4, 1, figsize=(6.5, 8.5), sharex=True,
                                   constrained_layout=True)
    fig_cv.suptitle('MPC Grinding Circuit — Controlled Variables',
                    fontsize=11, fontweight='bold')
    for i, ax in enumerate(axs_cv):
        lb, ub = float(Y_lb_vals[i]), float(Y_ub_vals[i])
        ax.fill_between(timesteps, lb, ub, color=COLOR_FEASIBLE, alpha=0.15, label='Feasible region')
        ax.axhline(y=lb, color=COLOR_CONSTRAINT, linestyle='--', linewidth=1.2, label='Constraints')
        ax.axhline(y=ub, color=COLOR_CONSTRAINT, linestyle='--', linewidth=1.2)
        ax.step(t_u, SP[i], where='post', color=COLOR_SP, linewidth=1.5, label='Setpoint')
        ax.plot(timesteps, Y[i], color=COLOR_PV, linewidth=1.5, label='PV')
        ax.set_ylabel(y_labels[i])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True, length=4)
        ax.tick_params(which='minor', length=2)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    axs_cv[-1].set_xlabel('Step [–]')
    fig_cv.savefig(OUTPUT_DIR / 'grinding_cv_trajectories.svg')
    fig_cv.savefig(OUTPUT_DIR / 'grinding_cv_trajectories.pdf')
    plt.show()

    # Figure 2: Manipulated Variables
    fig_mv, axs_mv = plt.subplots(4, 1, figsize=(6.5, 8.5), sharex=True,
                                   constrained_layout=True)
    fig_mv.suptitle('MPC Grinding Circuit — Manipulated Variables',
                    fontsize=11, fontweight='bold')
    for i, ax in enumerate(axs_mv):
        lb, ub = float(U_lb_vals[i]), float(U_ub_vals[i])
        ax.fill_between(t_u, lb, ub, color=COLOR_FEASIBLE, alpha=0.15, label='Feasible region')
        ax.axhline(y=lb, color=COLOR_CONSTRAINT, linestyle='--', linewidth=1.2, label='Constraints')
        ax.axhline(y=ub, color=COLOR_CONSTRAINT, linestyle='--', linewidth=1.2)
        ax.step(t_u, U[i], where='post', color=COLOR_PV, linewidth=1.5, label='MV')
        ax.set_ylabel(u_labels[i])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True, length=4)
        ax.tick_params(which='minor', length=2)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    axs_mv[-1].set_xlabel('Step [–]')
    fig_mv.savefig(OUTPUT_DIR / 'grinding_mv_trajectories.svg')
    fig_mv.savefig(OUTPUT_DIR / 'grinding_mv_trajectories.pdf')
    plt.show()

    # Figure 3: Kalman Filter Bias Estimates
    fig_bias, axs_bias = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True,
                                       constrained_layout=True)
    fig_bias.suptitle('MPC Grinding Circuit — Kalman Filter Bias Estimates',
                      fontsize=11, fontweight='bold')
    ax_du = axs_bias[0]
    for i in range(DU_BIAS.shape[0]):
        ax_du.plot(t_u, DU_BIAS[i], color=COLORS_BIAS[i], linewidth=1.5, label=du_labels[i])
    ax_du.set_ylabel('Input bias [–]')
    ax_du.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    ax_du.xaxis.set_minor_locator(AutoMinorLocator()); ax_du.yaxis.set_minor_locator(AutoMinorLocator())
    ax_du.tick_params(which='both', direction='in', top=True, right=True, length=4)

    ax_dy = axs_bias[1]
    for i in range(DY_BIAS.shape[0]):
        ax_dy.plot(t_u, DY_BIAS[i], color=COLORS_BIAS[i], linewidth=1.5, label=dy_labels[i])
    ax_dy.set_ylabel('Output bias [–]'); ax_dy.set_xlabel('Step [–]')
    ax_dy.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    ax_dy.xaxis.set_minor_locator(AutoMinorLocator()); ax_dy.yaxis.set_minor_locator(AutoMinorLocator())
    ax_dy.tick_params(which='both', direction='in', top=True, right=True, length=4)

    fig_bias.savefig(OUTPUT_DIR / 'grinding_bias_estimates.svg')
    fig_bias.savefig(OUTPUT_DIR / 'grinding_bias_estimates.pdf')
    plt.show()

    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
