# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Helmholtz-Zentrum Dresden-Rossendorf e.V. (HZDR)
# Authors: 
# - Ênio Lopes Júnior
# - Sebastian Felix Reinecke
#
# Contains modifications of code from casadi-nlp
# (https://github.com/FilippoAiraldi/casadi-nlp),
# Copyright (c) 2024 Filippo Airaldi, licensed under the MIT License.

import logging
from collections.abc import Collection, Iterable
from itertools import chain
from math import ceil
from typing import Any, Callable, Literal, Optional, Tuple, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ...core.cache import invalidate_caches_of
from ...core.solutions import Solution
from ...core.warmstart import WarmStartStrategy
from ...util.math import repeat
from ..wrapper import Nlp, NonRetroactiveWrapper

logger = logging.getLogger(__name__)

SymType = TypeVar("SymType", cs.SX, cs.MX)


def _n(statename: str) -> str:
    """Internal utility for naming initial states."""
    return f"{statename}_0"


Sym = Union[cs.MX, cs.SX, cs.DM, float, int, np.ndarray]


def _smooth_clip(z: cs.MX, low: float, high: float, eps: float = 1e-4) -> cs.MX:
    """Smooth approximation of clipping z to [low, high] using sqrt-based softmax/min.

    Parameters
    ----------
    z : cs.MX
        Value to clip.
    low : float
        Lower bound.
    high : float
        Upper bound.
    eps : float, optional
        Smoothing parameter, default 1e-4.

    Returns
    -------
    cs.MX
        Smoothly clipped value.
    """

    def smax(a, b):
        return 0.5 * ((a + b) + cs.sqrt((a - b) ** 2 + eps**2))

    def smin(a, b):
        return 0.5 * ((a + b) - cs.sqrt((a - b) ** 2 + eps**2))

    return smin(smax(z, low), high)


def _to_mx(v: Sym) -> cs.MX:
    """Converts various types (SX, DM, float, int, ndarray) to MX.

    Parameters
    ----------
    v : Sym
        Value of type MX, SX, DM, float, int, or np.ndarray.

    Returns
    -------
    cs.MX
        Value as MX type.
    """
    if isinstance(v, cs.MX):
        return v
    if isinstance(v, cs.SX):
        return cs.MX(v)
    if isinstance(v, cs.DM):
        return cs.MX(v)
    if isinstance(v, (float, int)):
        return cs.MX(v)
    return cs.MX(cs.DM(v))


def _broadcast_input_bias(bias: Sym, nu: int, T: int) -> cs.MX:
    """Broadcasts input bias from scalar or (nu,1) to (nu, T).

    Parameters
    ----------
    bias : Sym
        Bias value, either scalar (1x1) or column vector (nu, 1).
    nu : int
        Number of inputs.
    T : int
        Time horizon length.

    Returns
    -------
    cs.MX
        Broadcast bias with shape (nu, T).

    Raises
    ------
    ValueError
        If bias shape is not (1x1) or (nu,1).
    """
    B = _to_mx(bias)
    r, c = int(B.size1()), int(B.size2())
    # scalar
    if r == 1 and c == 1:
        return cs.repmat(B, nu, T)
    # (nu,1)
    if r == nu and c in (1, 0):
        return cs.repmat(B, 1, T)
    raise ValueError("input_bias should be (1x1) scalar or (nu,1).")


def _broadcast_output_bias(bias: Sym, nx: int, T: int) -> cs.MX:
    """Broadcasts output bias from scalar or (nx,1) to (nx, T).

    Parameters
    ----------
    bias : Sym
        Bias value, either scalar (1x1) or column vector (nx, 1).
    nx : int
        Number of outputs.
    T : int
        Time horizon length.

    Returns
    -------
    cs.MX
        Broadcast bias with shape (nx, T).

    Raises
    ------
    ValueError
        If bias shape is not (1x1) or (nx,1).
    """
    B = _to_mx(bias)
    r, c = int(B.size1()), int(B.size2())
    # scalar
    if r == 1 and c == 1:
        return cs.repmat(B, nx, T)
    # (nx,1)
    if r == nx and c in (1, 0):
        return cs.repmat(B, 1, T)
    raise ValueError("output_bias should be (1x1) scalar or (nx,1).")


class Mpc(NonRetroactiveWrapper[SymType]):
    """A wrapper to easily turn an NLP scheme into an MPC controller. Most of the theory
    for MPC is taken from :cite:`rawlings_model_2017`.

    Parameters
    ----------
    nlp : Nlp
        NLP scheme to be wrapped
    prediction_horizon : int
        A positive integer for the prediction horizon of the MPC controller.
    tuning_parameters: None or dict[str, array_like] or collection of
        A None or a dict containing all tuning parameters (e.g. cost function weighting matrices, slack penalty matrices etc.)
    n_context: int
        A positive integer for the context inputs used to warmup the Neural Network Model.
        At every MPC computation, the LSTM hidden/cell states are estimated numerically by warm-up over 
        a context window of ``n_context`` past observations.
    control_horizon : int, optional
        A positive integer for the control horizon of the MPC controller. If not given,
        it is set equal to the control horizon.
    input_spacing : int, optional
        Spacing between independent input actions. This argument allows to reduce the
        number of free actions along the control horizon by allowing only the first
        action every ``n`` to be free, and the following ``n-1`` to be fixed equal to
        that action (where ``n`` is given by ``input_spacing``). By default, no spacing
        is allowed, i.e., ``1``.
    neural: bool, optional
        A flag to signal if the MPC is whether Neural or Non-Neural (i.e. "Conventional" Linear or Non-linear MPC)
    shooting : 'single' or 'multi', optional
        Type of approach in the direct shooting for parametrizing the control
        trajectory. See Section 8.5 in :cite:`rawlings_model_2017`. By default, direct
        shooting is used.

    Raises
    ------
    ValueError
        Raises if the shooting method is invalid; or if any of the horizons are invalid.
    """

    def __init__(
        self,
        nlp: Nlp[SymType],
        prediction_horizon: int,
        tuning_parameters: Union[None, dict[str, npt.ArrayLike]],
        n_context: int = 1,
        control_horizon: Optional[int] = None,
        input_spacing: int = 1,
        shooting: Literal["single", "multi"] = "multi",
        neural: bool = False,
    ) -> None:
        super().__init__(nlp)

        if not isinstance(prediction_horizon, int) or prediction_horizon <= 0:
            raise ValueError("Prediction horizon must be positive and > 0.")
        if shooting == "single":
            self._is_multishooting = False
        elif shooting == "multi":
            self._is_multishooting = True
        else:
            raise ValueError("Invalid shooting method.")

        self._tuning_pars = tuning_parameters
        self._prediction_horizon = prediction_horizon
        self._n_context = n_context
        self._neural = neural
        if not neural and n_context != 1:
            raise ValueError("For non-neural MPC: n_context must be equal to 1.")
        if control_horizon is None:
            self._control_horizon = self._prediction_horizon
        elif not isinstance(control_horizon, int) or control_horizon <= 0:
            raise ValueError("Control horizon must be positive and > 0.")
        else:
            self._control_horizon = control_horizon

        if not isinstance(input_spacing, int) or input_spacing <= 0:
            raise ValueError("Input spacing factor must be positive and > 0.")
        self._input_spacing = input_spacing

        self._states: dict[str, SymType] = {}
        self._initial_states: dict[str, SymType] = {}
        self._initial_actions: dict[str, SymType] = {}
        self._actions: dict[str, SymType] = {}
        self._actions_exp: dict[str, SymType] = {}
        self._slacks: dict[str, SymType] = {}
        self._disturbances: dict[str, SymType] = {}
        self._dynamics: cs.Function = None

        # Stateful neural-MPC bookkeeping (populated by set_neural_dynamics + solve_mpc)
        self._n_warmup: int = 1
        self._solve_count: int = 0
        self._lstm_h: Optional[list[np.ndarray]] = None
        self._lstm_c: Optional[list[np.ndarray]] = None
        self._lstm_model: object = None
        self._lstm_layers: int = 0
        self._lstm_h_out: int = 0
        self._lstm_hidden: int = 0
        self._h0_nlp_param: Optional[SymType] = None
        self._c0_nlp_param: Optional[SymType] = None

    @property
    def prediction_horizon(self) -> int:
        """Gets the prediction horizon of the MPC controller."""
        return self._prediction_horizon

    @property
    def control_horizon(self) -> int:
        """Gets the control horizon of the MPC controller."""
        return self._control_horizon

    @property
    def states(self) -> dict[str, SymType]:
        """Gets the states of the MPC controller."""
        return self._states

    @property
    def initial_states(self) -> dict[str, SymType]:
        """Gets the initial states (parameters) of the MPC controller."""
        return self._initial_states

    @property
    def initial_actions(self) -> dict[str, SymType]:
        """Gets the initial actions of the MPC controller."""
        return self._initial_actions

    @property
    def first_states(self) -> dict[str, SymType]:
        """Gets the first (along the prediction horizon) states of the controller."""
        return {n: s[:, 0] for n, s in self._states.items()}

    @property
    def first_actions(self) -> dict[str, SymType]:
        """Gets the first effective (along the prediction horizon) actions of the controller."""
        return {n: a[:, 0] for n, a in self._actions.items()}


    @property
    def ns(self) -> int:
        """Gets the number of states of the MPC controller."""
        return sum(x0.shape[0] for x0 in self._initial_states.values())

    @property
    def actions(self) -> dict[str, SymType]:
        """Gets the control actions of the MPC controller."""
        return self._actions

    @property
    def actions_expanded(self) -> dict[str, SymType]:
        """Gets the expanded control actions of the MPC controller."""
        return self._actions_exp

    @property
    def na(self) -> int:
        """Gets the number of actions of the MPC controller."""
        return sum(a.shape[0] for a in self._actions.values())

    @property
    def slacks(self) -> dict[str, SymType]:
        """Gets the slack variables of the MPC controller."""
        return self._slacks

    @property
    def nslacks(self) -> int:
        """Gets the number of slacks of the MPC controller."""
        return sum(s.shape[0] for s in self._slacks.values())

    @property
    def disturbances(self) -> dict[str, SymType]:
        """Gets the disturbance parameters of the MPC controller."""
        return self._disturbances

    @property
    def nd(self) -> int:
        """Gets the number of disturbances in the MPC controller."""
        return sum(d.shape[0] for d in self._disturbances.values())

    @property
    def dynamics(self) -> Optional[cs.Function]:
        """Dynamics of the controller's prediction model, i.e., a CasADi function of the
        form :math:`x_+ = F(x,u)` or :math:`x+ = F(x,u,d)`, where :math:`x,u,d` are the
        state, action, disturbances respectively, and :math:`x_+` is the next state. The
        function can have multiple outputs, in which case :math:`x_+` is assumed to be
        the first one."""
        return self._dynamics

    @property
    def is_warmed_up(self) -> bool:
        """True once the stateful LSTM has been executed for `n_warmup` solves.

        Only meaningful when the dynamics were registered via
        `set_neural_dynamics`. After warmup, `solve_mpc()` advances
        `_lstm_h`/`_lstm_c` incrementally instead of re-estimating from the
        context window.
        """
        return self._lstm_model is not None and self._solve_count >= self._n_warmup

    def reset_lstm_state(self) -> None:
        """Force the next `solve_mpc()` call to re-warmup the LSTM.

        Zeros the persisted hidden/cell buffers and resets the solve counter,
        so subsequent solves re-run the warmup phase from the context window
        (seeded with zeros, since previous h/c are cleared). Use this when
        you suspect the persisted state has drifted from the real system —
        the `state_context`/`action_context` arrays act as the recovery
        truth data. No-op unless dynamics were registered via
        `set_neural_dynamics`.
        """
        self._lstm_h = None
        self._lstm_c = None
        self._solve_count = 0

    def state(
        self,
        name: str,
        size: int = 1,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
        bound_initial: bool = True,
        bound_terminal: bool = True,
    ) -> tuple[Optional[SymType], SymType]:
        """Adds a state variable to the MPC controller along the whole prediction
        horizon. Automatically creates the constraint on the initial conditions for this
        state.

        Parameters
        ----------
        name : str
            Name of the state.
        size : int
            Size of the state (assumed to be a vector).
        lb : array_like, casadi.DM, optional
            Hard lower bound of the state, by default ``-np.inf``.
        ub : array_like, casadi.DM, optional
            Hard upper bound of the state, by default ``+np.inf``.
        bound_initial : bool, optional
            If ``False``, then the upper and lower bounds on the initial state are not
            imposed, i.e., set to ``+/- np.inf`` (since the initial state is constrained
            to be equal to the current state of the system, it is sometimes advantageous
            to remove its bounds). By default ``True``.
        bound_terminal : bool, optional
            Same as above, but for the terminal state. By default ``True``.

        Returns
        -------
        state : casadi.SX or MX or None
            The state symbolic variable. If ``shooting=single``, then ``None`` is
            returned since the state will only be available once the dynamics are set.
        initial state : casadi.SX or MX
            The initial state symbolic parameter.

        Raises
        ------
        ValueError
            Raises if there exists already a state with the same name.
        RuntimeError
            Raises in single shooting if lower or upper bounds have been specified,
            since these can only be set after the dynamics have been set via the
            :meth:`constraint` method.
        """
        x0_name = _n(name)
        if self._is_multishooting:
            # Neural MPC predicts every column from the persisted LSTM state, so it
            # uses a plain N-column layout. Conventional MPC keeps the extra anchor
            # column (x[:, 0] == x0), hence N + 1 columns.
            T = (
                self._prediction_horizon
                if self._neural
                else self._prediction_horizon + 1
            )
            shape = (size, T)
            lb = np.broadcast_to(lb, shape).astype(float)
            ub = np.broadcast_to(ub, shape).astype(float)
            if not bound_initial:
                if self._neural:
                    pass  # column 0 is a genuine prediction; keep its bounds
                else:
                    lb[:, : 1] = -np.inf
                    ub[:, : 1] = +np.inf
            if not bound_terminal:
                lb[:, -1] = -np.inf
                ub[:, -1] = +np.inf

            # create state variable and initial state constraint
            x = self.nlp.variable(name, shape, lb, ub)[0]
            x0 = self.nlp.parameter(x0_name, (size, 1))
            if not self._neural: #Only conventional state-space MPC needs this constraint
                self.nlp.constraint(x0_name, x[:, 0], "==", x0)
        else:
            if np.any(lb != -np.inf) or np.any(ub != +np.inf):
                raise RuntimeError(
                    "in single shooting, lower and upper state bounds can only be "
                    "created after the dynamics have been set"
                )
            x = None
            x0 = self.nlp.parameter(x0_name, (size, 1))
        self._states[name] = x
        self._initial_states[x0_name] = x0
        return x, x0

    def action(
        self,
        name: str,
        size: int = 1,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> tuple[SymType, SymType]:
        """Adds a control action variable to the MPC controller along the whole control
        horizon. Automatically expands this action to be of the same length of the
        prediction horizon by padding with the final action.

        Parameters
        ----------
        name : str
            Name of the control action.
        size : int, optional
            Size of the control action (assumed to be a vector). Defaults to ``1``.
        lb : array_like, casadi.DM, optional
            Hard lower bound of the control action, by default ``-np.inf``.
        ub : array_like, casadi.DM, optional
            Hard upper bound of the control action, by default ``+np.inf``.

        Returns
        -------
        action : casadi.SX or MX
            The control action symbolic variable.
        action_expanded : casadi.SX or MX
            The same control  action variable, but expanded to the same length of the
            prediction horizon.
        """
        nu_free = ceil(self._control_horizon / self._input_spacing)
        u = self.nlp.variable(name, (size, nu_free), lb, ub)[0]
        u0_name = _n(name)
        u0 = self.nlp.parameter(u0_name, (size, 1))
        u_exp: SymType = (
            u
            if self._input_spacing == 1
            else repeat(u, (1, self._input_spacing))[:, : self._control_horizon]
        )
        gap = self._prediction_horizon - u_exp.shape[1]
        u_last = u_exp[:, -1]
        u_exp = cs.horzcat(u_exp, *(u_last for _ in range(gap)))
        self._actions[name] = u
        self._actions_exp[name] = u_exp
        self._initial_actions[u0_name] = u0
        return u, u_exp, u0

    def disturbance(self, name: str, size: int = 1) -> SymType:
        """Adds a disturbance parameter to the MPC controller along the whole prediction
        horizon.

        Parameters
        ----------
        name : str
            Name of the disturbance.
        size : int, optional
            Size of the disturbance (assumed to be a vector). Defaults to ``1``.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new disturbance in the MPC controller.
        """
        d = self.nlp.parameter(name, (size, self._prediction_horizon))
        self._disturbances[name] = d
        return d

    def constraint(
        self,
        name: str,
        lhs: Union[SymType, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[SymType, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> tuple[SymType, ...]:
        """See :meth:`neuralmpcx.Nlp.constraint`."""
        out = self.nlp.constraint(name, lhs, op, rhs, soft, simplify)
        if soft:
            self._slacks[f"slack_{name}"] = out[2]
        return out

    def set_neural_dynamics(
        self,
        model: object,
        *,
        name: str = "F",
        input_order: str = "y_then_u",  # "y_then_u" or "u_then_y"
        input_bias: Optional[Sym] = None,  # scalar (1x1) or (nu,1)
        input_clip: Optional[Tuple[float, float]] = None,
        output_bias: Optional[Sym] = None,  # scalar (1x1) or (nx,1)
        output_clip: Optional[Tuple[float, float]] = None,
        allow_disturbances: bool = False,
        casadi_opts: Optional[dict] = None,
        warmstart: Union[
            Literal["last", "last-successful"], WarmStartStrategy
        ] = "last-successful",
        use_last_action_on_fail: bool = False,
        n_warmup: int = 1,
    ) -> None:
        """
        Build and register a CasADi dynamics function from a neural wrapper.

        Creates `casadi.Function name: F(x, u, h0, c0[, d]) -> y_next` from a
        callable model (e.g., `CasadiLSTM`) and registers it via
        `self.set_dynamics(...)`.

        The LSTM is stateful: its hidden/cell states are persisted across
        `solve_mpc()` calls. F is built with two NLP parameters ``h0``
        (shape ``(num_layers*h_out, 1)``) and ``c0`` (shape
        ``(num_layers*hidden_size, 1)``); the first `n_warmup` solves
        re-estimate `(h, c)` from the context window via teacher forcing
        (seeded with the previous solve's `(h, c)` — hybrid warmup), and
        subsequent solves advance them one numeric step using the latest
        measured `(u, y)`.

        Args
        ----
        model : callable
            CasADi-friendly wrapper with `__call__(inp)` or `forward(inp)` returning MX/SX/DM.
        name : str = "F"
        input_order : {"y_then_u","u_then_y"} = "y_then_u"
            How to stack inputs for the model: `[x;u]` or `[u;x]`.
        input_bias : Sym | None
            Additive bias on `u`. Scalar or (nu,1). Broadcast to (nu,T).
        input_clip : (float, float) | None
            Smoothly clip model inputs.
        output_bias : Sym | None
            Additive bias on output. Scalar or (nx,1). Broadcast to (nx,T).
        output_clip : (float, float) | None
            Smoothly clip model outputs.
        allow_disturbances : bool = False
            If True and disturbances exist, include `d ∈ R^{nd×T}`.
        casadi_opts : dict | None
            Options for `casadi.Function`. Default `{"allow_free": True, "cse": True}`.
        warmstart : {"last","last-successful"} | WarmStartStrategy = "last-successful"
        use_last_action_on_fail : bool = False
        n_warmup : int = 1
            Number of initial `solve_mpc()` calls during which the persisted
            LSTM hidden/cell buffers are re-estimated from the full context
            window (seeded with the previous solve's `(h, c)`). After warmup,
            the buffers are advanced one teacher-forced step per solve using
            the latest measured `(u, y)`.

        What it handles
        ---------------
        - Infers dimensions from `self._states` and `self._actions_exp` (require same T).
        - Applies input bias
        - Normalizes model output to shape `(nx, T)` (transpose/reshape if needed).
        - Smooth clipping for inputs/outputs.

        Raises
        ------
        RuntimeError
            Mismatched sequence lengths; disturbances T mismatch; `model` is a casadi.Function.
        TypeError
            `model` not callable / no `.forward`, or output not CasADi type.
        ValueError
            Invalid `input_order`.

        Side effects
        ------------
        - Registers `F` via `self.set_dynamics(..., warmstart=..., use_last_action_on_fail=...)`.
        """
        if isinstance(model, cs.Function):
            raise RuntimeError(
                "casadi.Function must be directly pass to set_dynamics() method; "
                "set_neural_dynamics is for callable models (e.g. CasadiLSTM)."
            )
        if casadi_opts is None:
            casadi_opts = {"allow_free": True, "cse": True}

        # Dimentions (full sequence T = N )
        X = cs.vcat(self._states.values())
        U = cs.vcat(self._actions_exp.values())
        nx, T = int(X.size1()), int(X.size2())
        nu, Tu = int(U.size1()), int(U.size2())
        if Tu != T:
                raise RuntimeError(
                    "Actions and states should have the same sequence length."
                )

        # Register h0/c0 NLP parameters for the persisted LSTM state.
        if not isinstance(n_warmup, int) or n_warmup < 1:
            raise ValueError("n_warmup must be a positive integer.")
        n_layers = int(getattr(model, "num_layers", 1))
        h_out = int(model.h_out)
        hidden = int(model.hidden_size)
        H = n_layers * h_out
        C = n_layers * hidden
        h0_par = self.nlp.parameter("h0", (H, 1))
        c0_par = self.nlp.parameter("c0", (C, 1))
        h0_sym = cs.MX.sym("h0", H, 1)
        c0_sym = cs.MX.sym("c0", C, 1)
        # Per-layer slicing for the model API
        h0_list = [h0_sym[l * h_out : (l + 1) * h_out, :] for l in range(n_layers)]
        c0_list = [c0_sym[l * hidden : (l + 1) * hidden, :] for l in range(n_layers)]
        # Persist layout so solve_mpc() knows the shapes
        self._lstm_model = model
        self._lstm_layers = n_layers
        self._lstm_h_out = h_out
        self._lstm_hidden = hidden
        self._n_warmup = n_warmup
        # Keep references to NLP-level params so the dynamics builders can wire them
        self._h0_nlp_param = h0_par
        self._c0_nlp_param = c0_par

        # Symbolic variables
        x_sym = cs.MX.sym("x", nx, T)
        u_sym = cs.MX.sym("u", nu, T)

        # Input BIAS (only allowed shapes; broadcast -> (nu,T))
        if input_bias is not None:
            B_full = _broadcast_input_bias(input_bias, nu, T)

            B = B_full
            u_biased = u_sym + B
        else:
            u_biased = u_sym

        # Stacking as expected by the wrapper
        if input_order == "y_then_u":
            stack = cs.vertcat(x_sym, u_biased)
        elif input_order == "u_then_y":
            stack = cs.vertcat(u_biased, x_sym)
        else:
            raise ValueError("input_order should be either 'y_then_u' or 'u_then_y'.")

        # Input smooth clipping
        if input_clip is not None:
            lo, hi = input_clip
            stack = _smooth_clip(stack, lo, hi)

        # model/wrapper call (duck-typing: __call__ or .forward)
        def _call_model(inp, **kwargs):
            if hasattr(model, "__call__"):
                return model(inp, **kwargs)
            if hasattr(model, "forward"):
                return model.forward(inp, **kwargs)
            raise TypeError(" *model* is not callable nor has .forward(...)")

        y_pred = _call_model(stack, h0=h0_list, c0=c0_list)
        if not isinstance(y_pred, (cs.MX, cs.SX, cs.DM)):
            raise TypeError("Model output should be CasADi (MX/SX/DM).")

        # Shape normalization (nx, T)
        if y_pred.size1() == nx and y_pred.size2() == T:
            y_norm = cs.MX(y_pred)
        elif y_pred.size1() == T and y_pred.size2() == nx:
            y_norm = cs.MX(y_pred).T
        elif y_pred.size1() == 1 and nx == 1 and y_pred.size2() == T:
            y_norm = cs.MX(y_pred)
        else:
            y_norm = cs.reshape(cs.MX(y_pred), nx, T)

        # Output BIAS (only allowed shapes; broadcast -> (nx,T))
        if output_bias is not None:
            YB = _broadcast_output_bias(output_bias, nx, T)
            y_norm = y_norm + YB

        # Output smooth clipping
        if output_clip is not None:
            lo, hi = output_clip
            y_norm = _smooth_clip(y_norm, lo, hi)

        # F formal inputs
        in_args = [x_sym, u_sym, h0_sym, c0_sym]
        in_names = ["x", "u", "h0", "c0"]
        if allow_disturbances and list(self._disturbances.values()):
            D = cs.vcat(self._disturbances.values())
            nd, Td = int(D.size1()), int(D.size2())
            if Td != T:
                raise RuntimeError(
                    "Disturbances should have the same sequence length as x&u."
                )
            d_sym = cs.MX.sym("d", nd, T)
            in_args.append(d_sym)
            in_names.append("d")

        F = cs.Function(name, in_args, [y_norm], in_names, ["y_next"], casadi_opts)
        self.set_dynamics(
            F, warmstart=warmstart, use_last_action_on_fail=use_last_action_on_fail
        )

    def set_dynamics(
        self,
        F: Union[
            cs.Function,
            Callable[[tuple[npt.ArrayLike, ...]], tuple[npt.ArrayLike, ...]],
        ],
        n_in: Optional[int] = None,
        n_out: Optional[int] = None,
        warmstart: Union[
            Literal["last", "last-successful"], WarmStartStrategy
        ] = "last-successful",
        use_last_action_on_fail: bool = False,
    ) -> None:
        """Sets the dynamics of the controller's prediction model and creates the
        dynamics constraints.

        Parameters
        ----------
        F : casadi.Function or callable
            A CasADi function of the form :math:`x_+ = F(x,u)` or :math:`x+ = F(x,u,d)`,
            where :math:`x,u,d` are the state, action, disturbances respectively, and
            :math:`x_+` is the next state. The function can have multiple outputs, in
            which case :math:`x_+` is assumed to be the first one.
        n_in : int, optional
            In case a callable is passed instead of a casadi.Function, then the number
            of inputs must be manually specified via this argument.
        n_out : int, optional
            Same as above, for outputs.

        Raises
        ------
        ValueError
            When setting, raises if the dynamics do not accept 2 or 3 input
            arguments (2 to 5 for neural MPC, where F may also take the
            ``h0``/``c0`` LSTM state parameters).
        RuntimeError
            When setting, raises if the dynamics have been already set; or if the
            function ``F`` does not take accept the expected input sizes.
        """
        if self._dynamics is not None:
            raise RuntimeError("Dynamics were already set.")
        if isinstance(F, cs.Function):
            n_in = F.n_in()
            n_out = F.n_out()
        elif n_in is None or n_out is None:
            raise ValueError(
                "Args `n_in` and `n_out` must be manually specified when F is not a "
                "casadi function."
            )
        if self._neural:
            # Neural F may have: (x, u), (x, u, d), (x, u, h0, c0), or (x, u, h0, c0, d).
            if n_in is None or n_in < 2 or n_in > 5 or n_out is None or n_out < 1:
                raise ValueError(
                    "The dynamics function must accept between 2 and 5 arguments "
                    f"and return at least 1 output; got {n_in} inputs and {n_out} "
                    "outputs instead."
                )
        else:
            if n_in is None or n_in < 2 or n_in > 3 or n_out is None or n_out < 1:
                raise ValueError(
                    "The dynamics function must accepted 2 or 3 arguments and return at "
                    f"at least 1 output; got {n_in} inputs and {n_out} outputs instead."
                )

        if self._is_multishooting:
            if self._neural:
                self._neural_multishooting_dynamics(F, n_in)
            else:
                self._multishooting_dynamics(F, n_in, n_out)
        else:
            if self._neural:
                self._neural_singleshooting_dynamics(F, n_in)
            else:
                self._singleshooting_dynamics(F, n_in, n_out)
        self._dynamics = F
        self._last_action_on_fail = use_last_action_on_fail
        self._last_solution: Optional[Solution[SymType]] = None
        self._last_action: Optional[cs.DM] = None
        self._warmstart = WarmStartStrategy(warmstart)

    def solve_mpc(
        self,
        state: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        state_context: npt.ArrayLike,
        state_indices: npt.ArrayLike,
        action_context: npt.ArrayLike,
        setpoint: npt.ArrayLike = None,  # Added the dynamic setpoint functionality
        input_bias: npt.ArrayLike = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        store_solution: bool = True,
        dynamic_pars: Union[None, dict[str, npt.ArrayLike]] = None,
    ) -> cs.DM:
        """
        Solve the agent's MPC and return the first control move.

        This assembles the parameter dictionary (state/context, action context,
        set-point, optional input bias, and tuning parameters), warm-starts the
        underlying NLP when possible, calls `self.solve(...)`, and returns the
        first control input as a `casadi.DM`.

        Behavior by mode
        ----------------
        - Neural mode (`self._neural is True`):
          * Builds state parameter `x0` from the **last** step of 
          `state_context[:, state_indices]`.
          * Provides the first action parameter `u0` from the last
            step of `action_context`.
          * Updates the persisted LSTM `(h, c)` numerically using the **last**
            `self._n_context` rows of `state_context`/`action_context` (the
            warmup window), and passes them as the `h0`/`c0` parameters.
          * Sets the set-point parameter `SP` to `setpoint[state_indices, 0]`.
          * Returns the control at column 0 of the solved action
            trajectory (the receding-horizon action applied now).
        - Non-neural mode (`self._neural is False`):
          * Builds initial state blocks directly from `state[state_indices]`
            (accepts array-like or dict-of-arrays).
          * Sets the set-point parameter `SP` to `setpoint` as given.
          * Returns the control at column `0` of the solved action trajectory.

        Parameters
        ----------
        state : array_like or dict[str, array_like]
            Current full state used **only in non-neural mode**. If an array,
            it must index with `state_indices`. If a dict, values are stacked in
            the order of `self.initial_states.keys()`.
        state_context : array_like, shape (T_ctx, N_full)
            Rolling window of measured states. The last `self._n_context` rows feed
            the numeric `(h, c)` warmup; the last row builds the
            `x0` parameter. Columns are filtered with `state_indices` to match
            the internal MPC state ordering (sum of block sizes in
            `self.initial_states`).
        state_indices : array_like of int, shape (nx_sel,)
            Indices mapping from the **full** state vector to the subset/order
            used by the MPC (`self.initial_states`). Length must equal the total
            state size expected by the MPC.
        action_context : array_like
            Recent control history. Accepted shapes:
            - `(T_ctx, na)` or `(na, T_ctx)` for multi-input,
            - `(T_ctx,)` for single-input.
            The last `self._n_context` steps feed the numeric `(h, c)` warmup; the
            last step build the `u0` parameter with shape
            `(na, 1)`.
        setpoint : array_like, optional
            Target output/set-point. In neural mode, only
            `setpoint[state_indices, 0]` is consumed (i.e., first column for the
            selected states). In non-neural mode, the full array is passed
            through to the MPC as `SP`. Provide shapes consistent with those
            expectations.
        input_bias : array_like, optional
            Additive bias for the control input passed as parameter
            `"input_bias"`. Must have shape `(na, 1)` or be broadcastable to it.
        vals0 : dict[str, array_like] or iterable of dict, optional
            Initial values for the NLP variables (warm start). If `None` and a
            previous solution exists, the solver is warm-started from
            `self._last_solution`. If an **iterable** is provided, it bypasses
            the internal warm-start strategy and is used directly (e.g.,
            multi-start).
        store_solution : bool, optional
            If `True` (default), store the solution according to the warm-start
            policy; otherwise do not update `self._last_solution`.
        dynamic_pars : dict[str, array_like] or iterable of dict, optional
            A None or a dict containing all dynamic parameters (e.g. initial state measurement, biases etc.)

        Returns
        -------
        casadi.DM
            The first optimal control input, shape `(na, 1)`.

        Notes
        -----
        - Tuning parameters from `self._get_tuning_parameters()` are merged with
          the assembled parameters from contexts/bias/set-point.
        - If `self.is_multi` and `vals0` is `None` or a single dict, additional
          initial guesses are generated via `self._warmstart.generate(...)`.
        - On solver success, `self._last_action` is updated. If the solve fails
          and `self._last_action_on_fail` is `True`, the previously successful
          action is returned instead.

        Raises
        ------
        ValueError
            If `input_bias` does not have shape `(na, 1)` (or broadcastable).
        """

        if self._neural:
            mpcstates = self.initial_states
            mpcactions = self.initial_actions
            # First parameters use only the most recent  measurement;
            # the full `_n_context` window feeds the numeric h/c warmup below.
            selected_states_contexts = state_context[
                -1 :, state_indices
            ]  # Shape: (len(state_indices), 1)

            if (
                selected_states_contexts.shape[1] == 1
            ):  # If the array has just one column
                selected_states_contexts = selected_states_contexts.flatten()[
                    np.newaxis, :
                ]  # Convert to a single row
            else:  # If two-column array
                selected_states_contexts = (
                    selected_states_contexts.T
                )  # Transpose to get len(state_indices) rows and multiple columns. Now i have an array of len(state_indices) rows (1 for each state) and multiple columns (1 for each timestep)
            if len(mpcstates) == 1:
                states = (selected_states_contexts,)
            else:
                cumsizes = np.cumsum([s.shape[0] for s in mpcstates.values()][:-1])
                states = np.split(
                    selected_states_contexts.reshape(len(state_indices), -1), cumsizes
                )
            x0_dict = dict(zip(mpcstates.keys(), states))

            _action_context = np.array(action_context)[-1 :].copy()

            if _action_context.shape[1] == 1:  # Case: Lists with single elements
                _action_context = _action_context.flatten()[
                    np.newaxis, :
                ]  # Convert to a single row
            else:  # Case: Two-element or multi-element lists
                _action_context = (
                    _action_context.T
                )  # Transpose to get 2 rows and multiple columns
            if len(mpcactions) == 1:
                actions = (_action_context,)
            else:
                cumsizes = np.cumsum([a.shape[0] for a in mpcactions.values()][:-1])
                actions = np.split(_action_context, cumsizes)
            additional_pars = x0_dict
            u0_dict = dict(zip(mpcactions.keys(), actions))
            additional_pars.update(u0_dict)
            setpoint = np.array(setpoint)
            additional_pars["SP"] = setpoint[state_indices, 0]

            if self._lstm_model is not None:
                # Numerical state update outside F: warmup re-estimates the
                # context window seeded from previous (h, c); post-warmup
                # advances one step using the latest measured (u, y).
                y_ctx = np.asarray(state_context)[
                    -self._n_context :, state_indices
                ]  # (n_ctx, ny)
                u_ctx = np.asarray(action_context)[-self._n_context :, :]  # (n_ctx, nu)
                if self._solve_count < self._n_warmup:
                    self._lstm_h, self._lstm_c = self._lstm_model.estimate_numeric(
                        u_ctx,
                        y_ctx,
                        h_seed=self._lstm_h,
                        c_seed=self._lstm_c,
                    )
                else:
                    self._lstm_h, self._lstm_c = self._lstm_model.step_numeric(
                        u_ctx[-1], y_ctx[-1], self._lstm_h, self._lstm_c
                    )
                additional_pars["h0"] = np.concatenate(
                    [np.asarray(h).ravel() for h in self._lstm_h]
                )[:, None]
                additional_pars["c0"] = np.concatenate(
                    [np.asarray(c).ravel() for c in self._lstm_c]
                )[:, None]
                self._solve_count += 1
        else:
            mpcstates = self.initial_states
            mpcactions = self.initial_actions
            if isinstance(state, dict):
                state = np.array(
                    list(state.values())
                )  # Convert dict values to NumPy array
                selected_states = state[state_indices]
                if len(mpcstates) == 1:
                    states = (selected_states,)
                else:
                    cumsizes = np.cumsum([s.shape[0] for s in mpcstates.values()][:-1])
                    states = np.split(selected_states, cumsizes)
                x0_dict = dict(zip(mpcstates.keys(), states))
            else:
                selected_states = state[state_indices]
                if len(mpcstates) == 1:
                    states = (selected_states,)
                else:
                    cumsizes = np.cumsum([s.shape[0] for s in mpcstates.values()][:-1])
                    states = np.split(selected_states, cumsizes)
                x0_dict = dict(zip(mpcstates.keys(), states))

            _action_context = np.array(action_context).copy()

            if _action_context.shape[1] == 1:  # Case: Lists with single elements
                _action_context = _action_context.flatten()[
                    np.newaxis, :
                ]  # Convert to a single row
            else:  # Case: Two-element or multi-element lists
                _action_context = (
                    _action_context.T
                )  # Transpose to get 2 rows and multiple columns
            if len(mpcactions) == 1:
                actions = (_action_context.reshape(1, -1),)
            else:
                cumsizes = np.cumsum([a.shape[0] for a in mpcactions.values()][:-1])
                actions = np.split(_action_context, cumsizes)
            additional_pars = x0_dict
            u0_dict = dict(zip(mpcactions.keys(), actions))
            additional_pars.update(u0_dict)
            setpoint = np.array(setpoint)
            additional_pars["SP"] = setpoint

        if input_bias is not None:
            r, c = input_bias.shape
            if r == self.na and c in (1, 0):
                input_bias = np.array(input_bias)
                additional_pars["input_bias"] = input_bias
            else:
                raise ValueError("input_bias should be (nu,1).")

        pars = self._get_tuning_parameters()
        pars.update(additional_pars)

        if dynamic_pars is not None:
            pars.update(dynamic_pars)

        if vals0 is None and self._last_solution is not None:
            vals0 = self._last_solution.vals

        # use the warmstart strategy to generate multiple initial points for the NLP if
        # the NLP supports multi and `vals0` is not already an iterable of dict
        if self.is_multi and (vals0 is None or isinstance(vals0, dict)):
            more_vals0s = self._warmstart.generate(vals0)
            if self.nlp.starts > self._warmstart.n_points:
                # the difference between these two has been checked to be at most one,
                # meaning we can include `vals0` itself
                more_vals0s = chain((vals0,), more_vals0s)
            vals0 = more_vals0s

        sol = self.solve(pars, vals0)

        if store_solution and (self._warmstart.store_always or sol.success):
            self._last_solution = sol

        # times.append(sol.stats["t_wall_solver"])
        u_opt = cs.vertcat(*(sol.vals[u][:, 0] for u in self.actions.keys()))
        if sol.success:
            self._last_action = u_opt
        elif self._last_action_on_fail and self._last_action is not None:
            u_opt = self._last_action
            logger.warning(
                "Solver failed (using last action as fallback): %s", sol.status
            )
        else:
            logger.warning("Solver failed: %s", sol.status)

        return u_opt

    def _multishooting_dynamics(self, F: cs.Function, n_in: int, n_out: int) -> None:
        """Creates step-wise dynamics equality constraints for multi-shooting.

        Enforces x[:, k+1] == F(x[:, k], u[:, k]) for k=0..N-1.
        """
        X = cs.vcat(self._states.values())
        U = cs.vcat(self._actions_exp.values())
        if n_in < 3:
            args_at = lambda k: (X[:, k], U[:, k])
        else:
            D = cs.vcat(self._disturbances.values())
            args_at = lambda k: (X[:, k], U[:, k], D[:, k])
        xs_next = []
        for k in range(self._prediction_horizon):
            x_next = F(*args_at(k))
            if n_out != 1:
                x_next = x_next[0]
            xs_next.append(x_next)
        self.constraint("dyn", cs.hcat(xs_next), "==", X[:, 1:])

    def _neural_multishooting_dynamics(self, F: cs.Function, n_in: int) -> None:
        """Creates vectorized dynamics constraints for neural MPC multi-shooting.

        Calls F(X, U[, h0, c0][, D]) over the full horizon. Constraints enforce
        x[:, :] == F(...)[:, :], so every column (including the first) is a genuine
        prediction rolled forward from the persisted LSTM state `h0`/`c0` — no
        column is pinned to the measurement. When the dynamics were built by
        `set_neural_dynamics`, the `h0`, `c0` NLP parameters are threaded as formal
        inputs to F.
        """
        X = cs.vcat(self._states.values())
        U = cs.vcat(self._actions_exp.values())

        args_at: tuple[Any, ...] = (X, U)
        if self._h0_nlp_param is not None:
            args_at = args_at + (self._h0_nlp_param, self._c0_nlp_param)
        if list(self._disturbances.values()):
            D = cs.vcat(self._disturbances.values())
            args_at = args_at + (D,)

        xs_next = F(*args_at)
        self.constraint(
            "dyn", xs_next[:, :], "==", X[:, :]
        )

    def _singleshooting_dynamics(self, F: cs.Function, n_in: int, n_out: int) -> None:
        """Creates state trajectory by iterative forward simulation from x0.

        Builds the state trajectory X by recursively applying the dynamics function F
        starting from the initial state. Each state is computed as x[:, k+1] = F(x[:, k], u[:, k]).
        """
        Xk = cs.vcat(self._initial_states.values())
        U = cs.vcat(self._actions_exp.values())
        if n_in < 3:
            args_at = lambda k: (U[:, k],)
        else:
            D = cs.vcat(self._disturbances.values())
            args_at = lambda k: (U[:, k], D[:, k])
        X = [Xk]
        for k in range(self._prediction_horizon):
            Xk = F(Xk, *args_at(k))
            if n_out != 1:
                Xk = Xk[0]
            X.append(Xk)
        X = cs.hcat(X)
        cumsizes = np.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        self._states = dict(zip(self._states.keys(), cs.vertsplit(X, cumsizes)))

    def _neural_singleshooting_dynamics(self, F: cs.Function, n_in: int) -> None:
        """Creates state trajectory using neural dynamics with first column padding.

        Constructs the full state trajectory
        to the full sequence length T (= N), then calling the vectorized
        neural dynamics F(X0, U). When the dynamics were built by
        `set_neural_dynamics`, the `h0`/`c0` NLP parameters are passed to F as well.
        """
        U = cs.vcat(self._actions_exp.values())
        X0 = cs.vcat(self._initial_states.values())

        args_at: tuple[Any, ...]
        if list(self._disturbances.values()):
            D = cs.vcat(self._disturbances.values())
            args_at = (X0, U)
        if self._h0_nlp_param is not None:
            args_at = args_at + (self._h0_nlp_param, self._c0_nlp_param)
            args_at = args_at + (D,)
        else:
            args_at = (X0, U)
            if self._h0_nlp_param is not None:
                args_at = args_at + (self._h0_nlp_param, self._c0_nlp_param)

        X_next = F(*args_at)
        X = X_next
        cumsizes = np.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        self._states = dict(zip(self._states.keys(), cs.vertsplit(X, cumsizes)))


    def _get_tuning_parameters(
        self,
    ) -> Union[None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]]:
        """Internal utility to retrieve parameters of the MPC in order to solve it."""
        return self._tuning_pars
