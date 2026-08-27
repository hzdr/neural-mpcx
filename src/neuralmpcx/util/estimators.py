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

"""State estimation utilities for MPC applications.

This module provides Kalman filter implementations for state and bias
estimation in Model Predictive Control applications.

Classes
-------
KalmanFilter
    Standard discrete-time linear Kalman filter.
ExtendedKalmanFilter
    Discrete-time extended Kalman filter for nonlinear systems with
    CasADi-derived Jacobians.
AugmentedKalmanFilter
    Augmented linear Kalman filter for joint state and bias estimation.
AugmentedExtendedKalmanFilter
    Augmented extended Kalman filter for joint state and bias estimation on
    nonlinear systems.
MovingHorizonEstimator
    Constrained moving-horizon estimator for joint state and bias estimation,
    solving a small least-squares problem over a window of past measurements so
    that bounds on the estimate are enforced exactly.

Examples
--------
>>> from neuralmpcx.util.estimators import AugmentedKalmanFilter
>>> import numpy as np
>>> Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
>>> Bd = np.array([[0.0], [0.1]])
>>> Cd = np.array([[1.0, 0.0]])
>>> kf = AugmentedKalmanFilter(Ad, Bd, Cd)
>>> kf.predict(u=np.array([[1.0]]))
>>> kf.update(y=np.array([[0.5]]))
>>> biases = kf.get_mpc_biases()
"""

import time
import warnings
from collections import deque
from typing import Optional, Sequence, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_triangular

__all__ = [
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "AugmentedKalmanFilter",
    "AugmentedExtendedKalmanFilter",
    "MovingHorizonEstimator",
    "bias_detectability",
]


def _ensure_array(
    x: npt.ArrayLike, name: str, dtype: type = np.float64
) -> npt.NDArray[np.floating]:
    """Convert input to numpy array with validation.

    Parameters
    ----------
    x : array_like
        Input to convert.
    name : str
        Name of the parameter for error messages.
    dtype : type, optional
        Data type for the array. Default is np.float64.

    Returns
    -------
    np.ndarray
        Converted array.
    """
    if hasattr(x, "full"):  # CasADi DM
        x = x.full()
    return np.asarray(x, dtype=dtype)


def _ensure_column_vector(
    x: npt.ArrayLike, n: int, name: str
) -> npt.NDArray[np.floating]:
    """Convert input to column vector with dimension validation.

    Parameters
    ----------
    x : array_like
        Input to convert.
    n : int
        Expected number of elements.
    name : str
        Name of the parameter for error messages.

    Returns
    -------
    np.ndarray
        Column vector of shape (n, 1).

    Raises
    ------
    ValueError
        If the input does not have exactly n elements.
    """
    arr = _ensure_array(x, name)
    arr = arr.reshape(-1, 1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have {n} elements, got {arr.shape[0]}")
    return arr


def _ensure_channel_index(
    index: Optional[Sequence[int]], n: int, name: str, default_all: bool
) -> tuple[int, ...]:
    """Validate a channel selection against a signal width.

    Parameters
    ----------
    index : sequence of int or None
        Selected channel positions. None falls back to the default (either all
        channels or none, per ``default_all``).
    n : int
        Width of the signal the indices point into.
    name : str
        Name of the parameter for error messages.
    default_all : bool
        Whether None means "every channel" (True) or "no channel" (False).

    Returns
    -------
    tuple of int
        Validated, order-preserving channel positions.

    Raises
    ------
    ValueError
        If an index is out of range or repeated.
    """
    if index is None:
        return tuple(range(n)) if default_all else ()
    idx = tuple(int(i) for i in index)
    for i in idx:
        if not 0 <= i < n:
            raise ValueError(
                f"{name} contains out-of-range channel {i}; valid range is "
                f"0..{n - 1}"
            )
    if len(set(idx)) != len(idx):
        raise ValueError(f"{name} contains repeated channels: {idx}")
    return idx


def _scatter_matrix(index: Sequence[int], n: int) -> npt.NDArray[np.floating]:
    """Build the (n, len(index)) matrix scattering selected channels to full width.

    Parameters
    ----------
    index : sequence of int
        Selected channel positions.
    n : int
        Full signal width.

    Returns
    -------
    np.ndarray
        Matrix S with ``S[index[j], j] == 1`` and zeros elsewhere, so that
        ``S @ v`` places the compact vector ``v`` into the full-width signal.
    """
    S = np.zeros((n, len(index)), dtype=np.float64)
    for j, i in enumerate(index):
        S[i, j] = 1.0
    return S


def bias_detectability(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
    C: npt.ArrayLike,
    du_index: Sequence[int] = (),
    dy_index: Sequence[int] = (),
) -> dict[str, Union[bool, int, tuple[int, ...]]]:
    """Whether a bias augmentation can be estimated from the measurements.

    The textbook condition (Muske & Badgwell; Pannocchia & Rawlings) is that the
    augmented system is detectable iff the plant is detectable *and*::

        rank([[I - A, -B_d], [C, C_d]]) == nx + n_bias

    That matrix has ``nx + ny`` rows, so no more than ``ny`` bias states can ever
    be estimated, however they are distributed.

    This function tests the same matrix **incrementally**: the bias columns must
    *add* ``n_bias`` to the rank of the plant's own ``[[I - A], [C]]``, rather
    than reach the absolute ``nx + n_bias``. Where the plant is detectable the
    baseline *is* ``nx`` and the two statements are identical. Where it is not,
    the incremental form separates two failures a single number conflates:

    * the **selection** is unidentifiable -- a bias that the measurements cannot
      distinguish from something the model already does. An output bias on a
      channel the model already integrates is the canonical case: both are free
      integrators feeding the same output, so their split is set by the ratio of
      the drift covariances rather than by the data. Reported through
      ``detectable`` / ``deficiency``, and the offending channels are named.
    * the **realization** carries modes at ``z = 1`` that no measurement sees.
      That is a property of the model, true of the plain Kalman filter on the
      same plant, and unrelated to the bias selection. Reported separately
      through ``plant_rank_deficiency`` so a caller can warn about it instead of
      refusing an augmentation that is perfectly well posed on top of it.

    Parameters
    ----------
    A : array_like, shape (nx, nx)
        Discrete-time state transition matrix (a Jacobian, for a nonlinear
        model at the operating point).
    B : array_like, shape (nx, nu)
        Discrete-time input matrix.
    C : array_like, shape (ny, nx)
        Output/measurement matrix.
    du_index : sequence of int, optional
        Input channels carrying a bias. Default is none.
    dy_index : sequence of int, optional
        Output channels carrying a bias. Default is none.

    Returns
    -------
    dict
        ``n_bias`` and ``max_bias_states`` (the ``ny`` cap); ``baseline_rank``
        and ``rank`` (without and with the bias columns) and their difference
        ``gained``; ``required`` (``nx + n_bias``, the absolute test's target)
        and ``deficiency`` (``n_bias - gained``, how many bias states the data
        cannot separate); ``plant_rank_deficiency`` (``nx - baseline_rank``);
        ``detectable``; and ``redundant_du`` / ``redundant_dy``, the selected
        channels that add no rank at all.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0.9]]); B = np.array([[0.1]]); C = np.array([[1.0]])
    >>> bias_detectability(A, B, C, dy_index=[0])["detectable"]
    True
    >>> bias_detectability(A, B, C, du_index=[0], dy_index=[0])["detectable"]
    False
    """
    A_arr = _ensure_array(A, "A")
    B_arr = _ensure_array(B, "B")
    C_arr = _ensure_array(C, "C")
    nx = A_arr.shape[0]
    ny = C_arr.shape[0]
    du = tuple(int(i) for i in du_index)
    dy = tuple(int(i) for i in dy_index)
    n_bias = len(du) + len(dy)

    def rank_of(du_sel: Sequence[int], dy_sel: Sequence[int]) -> int:
        n = len(du_sel) + len(dy_sel)
        B_d = np.zeros((nx, n), dtype=np.float64)
        C_d = np.zeros((ny, n), dtype=np.float64)
        for j, i in enumerate(du_sel):
            B_d[:, j] = B_arr[:, i]
        for j, i in enumerate(dy_sel):
            C_d[i, len(du_sel) + j] = 1.0
        M = np.block(
            [
                [np.eye(nx, dtype=np.float64) - A_arr, -B_d],
                [C_arr, C_d],
            ]
        )
        return int(np.linalg.matrix_rank(M))

    baseline = rank_of((), ())
    rank = baseline if n_bias == 0 else rank_of(du, dy)
    gained = rank - baseline

    # Name the channels that carry no information of their own: dropping one and
    # finding the rank unchanged means the measurements never separated it. Only
    # worth computing when something is actually deficient, and cheap when it is
    # -- the whole selection is at most `ny` channels wide.
    redundant_du: tuple[int, ...] = ()
    redundant_dy: tuple[int, ...] = ()
    if n_bias and gained < n_bias:
        redundant_du = tuple(
            c
            for c in du
            if rank_of([i for i in du if i != c], dy) == rank
        )
        redundant_dy = tuple(
            c
            for c in dy
            if rank_of(du, [i for i in dy if i != c]) == rank
        )

    return {
        "n_bias": n_bias,
        "max_bias_states": ny,
        "baseline_rank": baseline,
        "rank": rank,
        "gained": gained,
        "required": nx + n_bias,
        "deficiency": n_bias - gained,
        "plant_rank_deficiency": nx - baseline,
        "detectable": n_bias <= ny and gained == n_bias,
        "redundant_du": redundant_du,
        "redundant_dy": redundant_dy,
    }


def _detectability_error(
    report: dict, du_index: Sequence[int], dy_index: Sequence[int]
) -> Optional[str]:
    """The message explaining a failed :func:`bias_detectability`, or None.

    Shared by the two augmented filters so that they refuse the same selections
    for the same stated reason.
    """
    n_bias = int(report["n_bias"])
    if n_bias == 0:
        return None
    ny = int(report["max_bias_states"])
    if n_bias > ny:
        return (
            f"{n_bias} bias states were requested ({len(du_index)} input + "
            f"{len(dy_index)} output) but only {ny} measured output(s) are "
            "available: an augmented model is never detectable with more bias "
            f"states than measurements. Keep at most {ny} channel(s) in "
            "du_index and dy_index combined -- beyond that the split between "
            "input and output bias is set by the ratio of Q_du to Q_dy rather "
            "than by the data."
        )
    if report["detectable"]:
        return None
    named = ", ".join(
        [f"input channel {c}" for c in report["redundant_du"]]
        + [f"output channel {c}" for c in report["redundant_dy"]]
    )
    culprit = (
        f" The measurements never separate {named} -- an output bias on a "
        "channel the model already integrates is the usual cause, since the "
        "bias and the integrator are both free and feed the same output."
        if named
        else ""
    )
    return (
        f"the requested bias states are not detectable: they add "
        f"{report['gained']} to the rank of [[I - A, -B_d], [C, C_d]] but "
        f"{n_bias} is required (deficiency {report['deficiency']}).{culprit} "
        f"Drop {report['deficiency']} channel(s) from du_index/dy_index, or "
        "move a bias onto a channel the measurements can actually separate. "
        "Pass check_detectability=False to build it anyway."
    )


def _warn_plant_rank_deficiency(report: dict, stacklevel: int = 3) -> None:
    """Warn when the *realization* has modes at ``z = 1`` no measurement sees.

    Not an error: it is equally true of the plain Kalman filter on the same
    model, so refusing only the augmented one would be incoherent. It does mean
    the estimate of those modes never converges, which is worth saying out loud.
    """
    deficiency = int(report["plant_rank_deficiency"])
    if deficiency <= 0:
        return
    warnings.warn(
        f"the model itself is rank-deficient in [[I - A], [C]] by {deficiency} "
        "-- it carries modes at z = 1 (integrators, or a non-minimal "
        "realization of them) that no measurement observes. Their estimate "
        "never converges. This is a property of the model, not of the bias "
        "selection, and applies to the unaugmented filter just as much; the "
        "bias states below are checked against this baseline rather than "
        "against the full state dimension.",
        UserWarning,
        stacklevel=stacklevel,
    )


#: Condition number past which a covariance earns a warning rather than an error.
_COND_WARN = 1e10


def _inv_chol(A: npt.ArrayLike, name: str) -> npt.NDArray[np.floating]:
    """Return the weight factor ``W`` with ``W.T @ W == inv(A)`` exactly.

    The moving-horizon cost weights residuals by the *inverse* of a covariance.
    Forming that inverse explicitly is the obvious route and the wrong one: a
    numerically computed ``inv(A)`` can come back very slightly indefinite, and
    an interior-point solver will happily chase a spurious descent direction
    along the offending eigenvector. Writing the term as ``sumsqr(W @ r)``
    instead is exactly positive semi-definite under roundoff regardless, and it
    hands the solver a least-squares structure it can exploit.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Symmetric positive-definite covariance.
    name : str
        Name of the matrix, for error messages.

    Returns
    -------
    np.ndarray
        The lower-triangular Cholesky factor's inverse, ``inv(L)`` where
        ``A == L @ L.T``, so that ``W.T @ W == inv(A)``.

    Raises
    ------
    ValueError
        If ``A`` is not square, not symmetric, or not positive definite.

    Warns
    -----
    UserWarning
        If ``A`` is ill-conditioned (``cond(A) > 1e10``).
    """
    arr = _ensure_array(A, name)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)

    asym = float(np.max(np.abs(arr - arr.T))) if arr.size else 0.0
    scale = max(1.0, float(np.max(np.abs(arr)))) if arr.size else 1.0
    if asym > 1e-8 * scale:
        raise ValueError(
            f"{name} must be symmetric; the largest asymmetry is {asym:.3g}. A "
            "covariance is symmetric by definition, and the moving-horizon "
            "weight is built from a Cholesky factor, which reads only the "
            "lower triangle -- so an asymmetric entry would be silently "
            "ignored rather than used."
        )

    try:
        L = np.linalg.cholesky(arr)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{name} must be positive definite: the moving-horizon cost weights "
            "residuals by its inverse, so a zero or negative direction means "
            "infinite weight. The Kalman filters tolerate a singular "
            f"{name} but the moving-horizon estimator cannot -- to say 'trust "
            "the model here', use a small positive value rather than zero. "
            "(Note that Q_x/Q_du/Q_dy/R/P0 are *covariances* here, as they are "
            "for every filter in this module, not the inverse-covariance "
            "weights that the moving-horizon literature usually writes.)"
        ) from exc

    if arr.size:
        cond = float(np.linalg.cond(arr))
        if cond > _COND_WARN:
            warnings.warn(
                f"{name} is ill-conditioned (cond = {cond:.3g}); the "
                "moving-horizon cost weights by its inverse, so the solver may "
                "report Restoration_Failed or stall. Rescale the states to O(1) "
                "or raise the smallest diagonal entry.",
                UserWarning,
                stacklevel=3,
            )

    return solve_triangular(L, np.eye(arr.shape[0]), lower=True)


def _inv_chol_blocks(
    blocks: Sequence[tuple[npt.ArrayLike, str]],
) -> npt.NDArray[np.floating]:
    """Weight factor for a block-diagonal covariance, validated block by block.

    The augmented process covariance is ``blkdiag(Q_x, Q_du, Q_dy)``, and the
    Cholesky factor of a block-diagonal matrix is the block diagonal of the
    factors -- so factoring per block is exact, and it lets a rejected block be
    reported under the keyword the caller actually passed (``Q_du``) rather than
    under the assembled matrix's internal name.

    Parameters
    ----------
    blocks : sequence of (array_like, str)
        The diagonal blocks with their argument names, in order.

    Returns
    -------
    np.ndarray
        Block-diagonal ``W`` with ``W.T @ W == inv(blkdiag(...))``.
    """
    factors = [(_inv_chol(block, name), name) for block, name in blocks]
    n = sum(W.shape[0] for W, _ in factors)
    out = np.zeros((n, n), dtype=np.float64)
    off = 0
    for W, _ in factors:
        k = W.shape[0]
        out[off : off + k, off : off + k] = W
        off += k
    return out


def _augmented_model(
    f_plain: cs.Function,
    h_plain: cs.Function,
    S_du: npt.NDArray[np.floating],
    S_dy: npt.NDArray[np.floating],
    nx: int,
    n_du: int,
    n_dy: int,
) -> tuple[cs.Function, cs.Function]:
    """Build the bias-augmented process and measurement maps.

    The augmented state is ``z = [x; du_bias; dy_bias]`` with the bias entries
    *compact* -- only the selected channels -- and the model is::

        z_next = [f(x, u + S_du @ du) ; du ; dy]
        y      = h(x) + S_dy @ dy

    i.e. the input bias biases the control that drives ``f`` (which is what an
    actuator offset physically is), the output bias sits on the measurement, and
    both are random walks whose noise enters through the augmented ``Q``.

    Shared by :class:`AugmentedExtendedKalmanFilter` and
    :class:`MovingHorizonEstimator` so that the two provably describe the same
    augmented system -- that shared model is what makes them agree exactly on a
    linear plant.

    Parameters
    ----------
    f_plain : casadi.Function
        Normalized single-output plant dynamics ``f(x, u) -> x_next``.
    h_plain : casadi.Function
        Normalized measurement map ``h(x) -> y``.
    S_du, S_dy : np.ndarray
        Scatter matrices placing the selected bias channels into the full input
        and output widths, from :func:`_scatter_matrix`.
    nx, n_du, n_dy : int
        Plant state dimension and the counts of estimated bias states.

    Returns
    -------
    tuple of casadi.Function
        ``(f_aug, h_aug)`` with signatures ``f_aug(z, u) -> z_next`` and
        ``h_aug(z) -> y``.
    """
    nu = S_du.shape[0]
    n_aug = nx + n_du + n_dy

    z_sym = cs.MX.sym("z", n_aug)
    u_sym = cs.MX.sym("u", nu)
    x_part = z_sym[:nx]
    du_part = z_sym[nx : nx + n_du]
    dy_part = z_sym[nx + n_du :]

    u_biased = u_sym
    if n_du:
        u_biased = u_sym + cs.DM(S_du) @ du_part
    z_next = cs.vertcat(
        *(
            [f_plain(x_part, u_biased)]
            + ([du_part] if n_du else [])
            + ([dy_part] if n_dy else [])
        )
    )
    f_aug = cs.Function("aekf_f", [z_sym, u_sym], [z_next])

    y_aug = h_plain(x_part)
    if n_dy:
        y_aug = y_aug + cs.DM(S_dy) @ dy_part
    h_aug = cs.Function("aekf_h", [z_sym], [y_aug])

    return f_aug, h_aug


class KalmanFilter:
    """Discrete-time linear Kalman filter for state estimation.

    Estimates state x from noisy measurements y and control inputs u using
    the standard predict-update (time-measurement) recursion.

    System model::

        x[k+1] = Ad @ x[k] + Bd @ u[k] + w[k]    (process)
        y[k]   = Cd @ x[k] + Dd @ u[k] + v[k]    (measurement)

    where w ~ N(0, Q) and v ~ N(0, R).

    Parameters
    ----------
    Ad : array_like, shape (nx, nx)
        Discrete-time state transition matrix.
    Bd : array_like, shape (nx, nu)
        Discrete-time input matrix.
    Cd : array_like, shape (ny, nx)
        Output/measurement matrix.
    Dd : array_like, shape (ny, nu), optional
        Feedthrough matrix. Default is zero matrix.
    Q : array_like, shape (nx, nx), optional
        Process noise covariance matrix. Default is 0.1 * I.
    R : array_like, shape (ny, ny), optional
        Measurement noise covariance matrix. Default is 1.0 * I.
    x0 : array_like, shape (nx,) or (nx, 1), optional
        Initial state estimate. Default is zero vector.
    P0 : array_like, shape (nx, nx), optional
        Initial error covariance. Default is identity matrix.

    Attributes
    ----------
    nx : int
        State dimension.
    nu : int
        Input dimension.
    ny : int
        Output dimension.

    Examples
    --------
    >>> import numpy as np
    >>> Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
    >>> Bd = np.array([[0.0], [0.1]])
    >>> Cd = np.array([[1.0, 0.0]])
    >>> Q = np.eye(2) * 0.01
    >>> R = np.eye(1) * 0.1
    >>> kf = KalmanFilter(Ad, Bd, Cd, Q=Q, R=R)
    >>> kf.predict(u=np.array([[1.0]]))
    >>> kf.update(y=np.array([[0.5]]))
    >>> print(kf.x_est.flatten())

    Notes
    -----
    The filter operates in deviation form: all states and outputs are
    deviations from a nominal operating point. Users should handle
    offset subtraction/addition externally.
    """

    def __init__(
        self,
        Ad: npt.ArrayLike,
        Bd: npt.ArrayLike,
        Cd: npt.ArrayLike,
        Dd: Optional[npt.ArrayLike] = None,
        Q: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        x0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
    ) -> None:
        # Convert and validate system matrices
        self._Ad = _ensure_array(Ad, "Ad")
        self._Bd = _ensure_array(Bd, "Bd")
        self._Cd = _ensure_array(Cd, "Cd")

        # Validate dimensions
        if self._Ad.ndim != 2 or self._Ad.shape[0] != self._Ad.shape[1]:
            raise ValueError(f"Ad must be square, got shape {self._Ad.shape}")

        self._nx = self._Ad.shape[0]
        self._nu = self._Bd.shape[1]
        self._ny = self._Cd.shape[0]

        if self._Bd.shape[0] != self._nx:
            raise ValueError(
                f"Bd must have {self._nx} rows to match Ad, got {self._Bd.shape[0]}"
            )
        if self._Cd.shape[1] != self._nx:
            raise ValueError(
                f"Cd must have {self._nx} columns to match Ad, got {self._Cd.shape[1]}"
            )

        # Feedthrough matrix
        if Dd is None:
            self._Dd = np.zeros((self._ny, self._nu), dtype=np.float64)
        else:
            self._Dd = _ensure_array(Dd, "Dd")
            if self._Dd.shape != (self._ny, self._nu):
                raise ValueError(
                    f"Dd must have shape ({self._ny}, {self._nu}), "
                    f"got {self._Dd.shape}"
                )

        # Process noise covariance
        if Q is None:
            self._Q = np.eye(self._nx, dtype=np.float64) * 0.1
        else:
            self._Q = _ensure_array(Q, "Q")
            if self._Q.shape != (self._nx, self._nx):
                raise ValueError(
                    f"Q must have shape ({self._nx}, {self._nx}), got {self._Q.shape}"
                )

        # Measurement noise covariance
        if R is None:
            self._R = np.eye(self._ny, dtype=np.float64) * 1.0
        else:
            self._R = _ensure_array(R, "R")
            if self._R.shape != (self._ny, self._ny):
                raise ValueError(
                    f"R must have shape ({self._ny}, {self._ny}), got {self._R.shape}"
                )

        # Initialize state estimate
        if x0 is None:
            self._x_est = np.zeros((self._nx, 1), dtype=np.float64)
        else:
            self._x_est = _ensure_column_vector(x0, self._nx, "x0")

        # Initialize error covariance
        if P0 is None:
            self._P = np.eye(self._nx, dtype=np.float64)
        else:
            self._P = _ensure_array(P0, "P0")
            if self._P.shape != (self._nx, self._nx):
                raise ValueError(
                    f"P0 must have shape ({self._nx}, {self._nx}), got {self._P.shape}"
                )

    @property
    def nx(self) -> int:
        """State dimension."""
        return self._nx

    @property
    def nu(self) -> int:
        """Input dimension."""
        return self._nu

    @property
    def ny(self) -> int:
        """Output dimension."""
        return self._ny

    @property
    def x_est(self) -> npt.NDArray[np.floating]:
        """Current state estimate, shape (nx, 1)."""
        return self._x_est.copy()

    @property
    def P(self) -> npt.NDArray[np.floating]:
        """Current error covariance, shape (nx, nx)."""
        return self._P.copy()

    def predict(self, u: npt.ArrayLike) -> None:
        """Time update (prediction step).

        Propagates state estimate and covariance forward one time step
        using the system dynamics.

        Parameters
        ----------
        u : array_like, shape (nu,) or (nu, 1)
            Control input applied at the current time step.

        Notes
        -----
        Updates::

            x_pred = Ad @ x_est + Bd @ u
            P_pred = Ad @ P @ Ad.T + Q
        """
        u_vec = _ensure_column_vector(u, self._nu, "u")

        # State prediction
        self._x_est = self._Ad @ self._x_est + self._Bd @ u_vec

        # Covariance prediction
        self._P = self._Ad @ self._P @ self._Ad.T + self._Q

    def update(self, y: npt.ArrayLike, u: Optional[npt.ArrayLike] = None) -> None:
        """Measurement update (correction step).

        Corrects the predicted state using the measurement.

        Parameters
        ----------
        y : array_like, shape (ny,) or (ny, 1)
            Measured output (in deviation form).
        u : array_like, shape (nu,) or (nu, 1), optional
            Control input. Required if Dd is nonzero.

        Notes
        -----
        Updates::

            y_pred = Cd @ x_pred + Dd @ u
            K = P_pred @ Cd.T @ inv(Cd @ P_pred @ Cd.T + R)
            x_est = x_pred + K @ (y - y_pred)
            P = (I - K @ Cd) @ P_pred
        """
        y_vec = _ensure_column_vector(y, self._ny, "y")

        # Predicted measurement
        y_pred = self._Cd @ self._x_est
        if u is not None:
            u_vec = _ensure_column_vector(u, self._nu, "u")
            y_pred = y_pred + self._Dd @ u_vec

        # Measurement residual (innovation)
        y_res = y_vec - y_pred

        # Innovation covariance
        S = self._Cd @ self._P @ self._Cd.T + self._R

        # Kalman gain
        K = self._P @ self._Cd.T @ np.linalg.inv(S)

        # State estimate update
        self._x_est = self._x_est + K @ y_res

        # Error covariance update
        I = np.eye(self._nx, dtype=np.float64)
        self._P = (I - K @ self._Cd) @ self._P

    def reset(
        self,
        x0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Reset filter state to initial conditions.

        Parameters
        ----------
        x0 : array_like, optional
            New initial state estimate. If None, resets to zeros.
        P0 : array_like, optional
            New initial covariance. If None, resets to identity.
        """
        if x0 is None:
            self._x_est = np.zeros((self._nx, 1), dtype=np.float64)
        else:
            self._x_est = _ensure_column_vector(x0, self._nx, "x0")

        if P0 is None:
            self._P = np.eye(self._nx, dtype=np.float64)
        else:
            self._P = _ensure_array(P0, "P0")
            if self._P.shape != (self._nx, self._nx):
                raise ValueError(
                    f"P0 must have shape ({self._nx}, {self._nx}), got {self._P.shape}"
                )


class ExtendedKalmanFilter:
    """Discrete-time extended Kalman filter (EKF) for nonlinear systems.

    Estimates the state x of a nonlinear plant from noisy measurements y and
    control inputs u using the standard EKF predict-update recursion. The
    nonlinear dynamics are provided as a CasADi function, and the Jacobians
    required for the covariance propagation are derived automatically via
    CasADi algorithmic differentiation (no finite differences).

    System model::

        x[k+1] = f(x[k], u[k]) + w[k]    (process)
        y[k]   = h(x[k]) + v[k]          (measurement)

    where w ~ N(0, Q) and v ~ N(0, R).

    Parameters
    ----------
    f : casadi.Function
        Discrete-time state transition map with signature ``f(x, u) -> x_next``,
        where ``x`` is the state column vector (nx, 1) and ``u`` the input
        column vector (nu, 1). If the function has multiple outputs, the next
        state is assumed to be the first one (same convention as
        ``Mpc.set_dynamics``), so the dynamics registered on an ``Mpc``
        instance (``mpc.dynamics``) can be passed directly.
    h : casadi.Function or array_like
        Measurement map. Either a CasADi function with signature
        ``h(x) -> y``, or a measurement matrix ``C`` of shape (ny, nx) for the
        common linear-measurement case ``y = C @ x``.
    Q : array_like, shape (nx, nx), optional
        Process noise covariance matrix. Default is 0.1 * I.
    R : array_like, shape (ny, ny), optional
        Measurement noise covariance matrix. Default is 1.0 * I.
    x0 : array_like, shape (nx,) or (nx, 1), optional
        Initial state estimate. Default is zero vector.
    P0 : array_like, shape (nx, nx), optional
        Initial error covariance. Default is identity matrix.

    Attributes
    ----------
    nx : int
        State dimension.
    nu : int
        Input dimension.
    ny : int
        Output dimension.

    Examples
    --------
    >>> import casadi as cs
    >>> import numpy as np
    >>> x = cs.MX.sym("x", 2)
    >>> u = cs.MX.sym("u", 1)
    >>> dt = 0.1
    >>> x_next = x + dt * cs.vertcat(x[1], -cs.sin(x[0]) + u)
    >>> f = cs.Function("f", [x, u], [x_next])
    >>> C = np.array([[1.0, 0.0]])  # only the first state is measured
    >>> ekf = ExtendedKalmanFilter(f, C, Q=np.eye(2) * 0.01, R=np.eye(1) * 0.1)
    >>> ekf.predict(u=np.array([[0.5]]))
    >>> ekf.update(y=np.array([[0.2]]))
    >>> print(ekf.x_est.flatten())

    Notes
    -----
    The state and measurement dimensions are inferred from ``f`` and ``h``.
    Jacobians are evaluated numerically at the current estimate at each
    ``predict``/``update`` call.
    """

    def __init__(
        self,
        f: cs.Function,
        h: Union[cs.Function, npt.ArrayLike],
        Q: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        x0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
    ) -> None:
        # Validate the state transition function
        if not isinstance(f, cs.Function):
            raise TypeError(f"f must be a casadi.Function, got {type(f).__name__}")
        if f.n_in() != 2:
            raise ValueError(f"f must take exactly 2 inputs (x, u), got {f.n_in()}")
        if f.size2_in(0) != 1 or f.size2_in(1) != 1:
            raise ValueError(
                "f inputs must be column vectors, got shapes "
                f"{f.size_in(0)} and {f.size_in(1)}"
            )

        self._nx = int(f.size1_in(0))
        self._nu = int(f.size1_in(1))

        if f.size1_out(0) != self._nx:
            raise ValueError(
                f"first output of f must have {self._nx} rows to match the "
                f"state, got {f.size1_out(0)}"
            )

        # Rewrap f with fresh symbols so that multi-output functions are
        # normalized (x_next is the first output) and build the Jacobian
        # dF/dx via CasADi algorithmic differentiation.
        x_sym = cs.MX.sym("x", self._nx)
        u_sym = cs.MX.sym("u", self._nu)
        x_next = f(x_sym, u_sym)
        if isinstance(x_next, (list, tuple)):
            x_next = x_next[0]
        self._f = cs.Function("ekf_f", [x_sym, u_sym], [x_next])
        self._F_jac = cs.Function(
            "ekf_F_jac", [x_sym, u_sym], [cs.jacobian(x_next, x_sym)]
        )

        # Process the measurement map: CasADi function or linear matrix C
        if isinstance(h, cs.Function):
            if h.n_in() != 1:
                raise ValueError(
                    f"h must take exactly 1 input (x), got {h.n_in()}"
                )
            if h.size2_in(0) != 1:
                raise ValueError(
                    f"h input must be a column vector, got shape {h.size_in(0)}"
                )
            if h.size1_in(0) != self._nx:
                raise ValueError(
                    f"h input must have {self._nx} rows to match f, "
                    f"got {h.size1_in(0)}"
                )
            self._ny = int(h.size1_out(0))
            y_expr = h(x_sym)
            if isinstance(y_expr, (list, tuple)):
                y_expr = y_expr[0]
        else:
            C = _ensure_array(h, "h")
            if C.ndim != 2:
                raise ValueError(f"h must be a 2D matrix, got {C.ndim} dimensions")
            if C.shape[1] != self._nx:
                raise ValueError(
                    f"h must have {self._nx} columns to match f, got {C.shape[1]}"
                )
            self._ny = int(C.shape[0])
            y_expr = cs.mtimes(cs.DM(C), x_sym)
        self._h = cs.Function("ekf_h", [x_sym], [y_expr])
        self._H_jac = cs.Function("ekf_H_jac", [x_sym], [cs.jacobian(y_expr, x_sym)])

        # Process noise covariance
        if Q is None:
            self._Q = np.eye(self._nx, dtype=np.float64) * 0.1
        else:
            self._Q = _ensure_array(Q, "Q")
            if self._Q.shape != (self._nx, self._nx):
                raise ValueError(
                    f"Q must have shape ({self._nx}, {self._nx}), got {self._Q.shape}"
                )

        # Measurement noise covariance
        if R is None:
            self._R = np.eye(self._ny, dtype=np.float64) * 1.0
        else:
            self._R = _ensure_array(R, "R")
            if self._R.shape != (self._ny, self._ny):
                raise ValueError(
                    f"R must have shape ({self._ny}, {self._ny}), got {self._R.shape}"
                )

        # Initialize state estimate
        self._x_est: npt.NDArray[np.floating]
        if x0 is None:
            self._x_est = np.zeros((self._nx, 1), dtype=np.float64)
        else:
            self._x_est = _ensure_column_vector(x0, self._nx, "x0")

        # Initialize error covariance
        if P0 is None:
            self._P = np.eye(self._nx, dtype=np.float64)
        else:
            self._P = _ensure_array(P0, "P0")
            if self._P.shape != (self._nx, self._nx):
                raise ValueError(
                    f"P0 must have shape ({self._nx}, {self._nx}), got {self._P.shape}"
                )

    @property
    def nx(self) -> int:
        """State dimension."""
        return self._nx

    @property
    def nu(self) -> int:
        """Input dimension."""
        return self._nu

    @property
    def ny(self) -> int:
        """Output dimension."""
        return self._ny

    @property
    def x_est(self) -> npt.NDArray[np.floating]:
        """Current state estimate, shape (nx, 1)."""
        return self._x_est.copy()

    @property
    def P(self) -> npt.NDArray[np.floating]:
        """Current error covariance, shape (nx, nx)."""
        return self._P.copy()

    def predict(self, u: npt.ArrayLike) -> None:
        """Time update (prediction step).

        Propagates state estimate and covariance forward one time step using
        the nonlinear dynamics and their Jacobian linearization.

        Parameters
        ----------
        u : array_like, shape (nu,) or (nu, 1)
            Control input applied at the current time step.

        Notes
        -----
        Updates::

            F_k    = df/dx evaluated at (x_est, u)
            x_pred = f(x_est, u)
            P_pred = F_k @ P @ F_k.T + Q
        """
        u_vec = _ensure_column_vector(u, self._nu, "u")

        # Jacobian must be evaluated at the prior estimate
        F_k = self._F_jac(self._x_est, u_vec).full()

        # State prediction
        self._x_est = self._f(self._x_est, u_vec).full()

        # Covariance prediction
        self._P = F_k @ self._P @ F_k.T + self._Q

    def update(self, y: npt.ArrayLike, u: Optional[npt.ArrayLike] = None) -> None:
        """Measurement update (correction step).

        Corrects the predicted state using the measurement.

        Parameters
        ----------
        y : array_like, shape (ny,) or (ny, 1)
            Measured output.
        u : array_like, shape (nu,) or (nu, 1), optional
            Unused; kept for API symmetry with ``KalmanFilter.update``. The
            EKF measurement model ``h(x)`` has no feedthrough.

        Notes
        -----
        Updates::

            H_k    = dh/dx evaluated at x_pred
            y_pred = h(x_pred)
            K = P_pred @ H_k.T @ inv(H_k @ P_pred @ H_k.T + R)
            x_est = x_pred + K @ (y - y_pred)
            P = (I - K @ H_k) @ P_pred
        """
        y_vec = _ensure_column_vector(y, self._ny, "y")

        # Measurement Jacobian and predicted measurement at current estimate
        H_k = self._H_jac(self._x_est).full()
        y_pred = self._h(self._x_est).full()

        # Measurement residual (innovation)
        y_res = y_vec - y_pred

        # Innovation covariance
        S = H_k @ self._P @ H_k.T + self._R

        # Kalman gain
        K = self._P @ H_k.T @ np.linalg.inv(S)

        # State estimate update
        self._x_est = self._x_est + K @ y_res

        # Error covariance update
        I = np.eye(self._nx, dtype=np.float64)
        self._P = (I - K @ H_k) @ self._P

    def reset(
        self,
        x0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Reset filter state to initial conditions.

        Parameters
        ----------
        x0 : array_like, optional
            New initial state estimate. If None, resets to zeros.
        P0 : array_like, optional
            New initial covariance. If None, resets to identity.
        """
        if x0 is None:
            self._x_est = np.zeros((self._nx, 1), dtype=np.float64)
        else:
            self._x_est = _ensure_column_vector(x0, self._nx, "x0")

        if P0 is None:
            self._P = np.eye(self._nx, dtype=np.float64)
        else:
            self._P = _ensure_array(P0, "P0")
            if self._P.shape != (self._nx, self._nx):
                raise ValueError(
                    f"P0 must have shape ({self._nx}, {self._nx}), got {self._P.shape}"
                )


class AugmentedKalmanFilter:
    """Augmented Kalman filter for joint state and bias estimation.

    Extends the standard Kalman filter to simultaneously estimate:

    - Plant state x (nx dimensions)
    - Input bias du_bias on the channels named by ``du_index``
    - Output bias dy_bias on the channels named by ``dy_index``

    Biases are modeled as random walks (integrated white noise).

    Augmented state vector, with the bias entries *compact* -- one state per
    selected channel, not one per signal::

        z = [x; du_bias[du_index]; dy_bias[dy_index]]

    Augmented system::

        z[k+1] = A_aug @ z[k] + B_aug @ u[k]
        y[k]   = C_aug @ z[k] + D_aug @ u[k]

    where::

        A_aug = [Ad   Bd@S_du  0    ]     B_aug = [Bd]
                [0    I        0    ]             [0 ]
                [0    0        I    ]             [0 ]

        C_aug = [Cd   Dd@S_du  S_dy ]     D_aug = Dd

    The ``Bd @ S_du`` cross-block is what makes the input bias an *actuator*
    offset: the augmented prediction is ``x⁺ = Ad·x + Bd·(u + S_du·δu)``, the
    linear form of :class:`AugmentedExtendedKalmanFilter`'s
    ``x⁺ = f(x, u + S_du·δu)``. Before this block existed the input bias reached
    the model only through ``Dd``, so on a plant with no feedthrough it was a
    dead state -- it random-walked under ``Q_du``, corrected nothing, and was
    unobservable while still being reported as an estimate.

    **The channels are a budget, not a checkbox list.** The augmented system is
    detectable only if the bias states add rank to ``[[I - A, -B_d], [C, C_d]]``,
    a matrix with ``nx + ny`` rows -- so at most ``ny`` bias states exist, and a
    bias the measurements cannot separate from something the model already does
    (an output bias on a channel the model integrates, say) does not count. The
    constructor checks this through :func:`bias_detectability` and raises;
    :meth:`detectability_report` exposes the numbers.

    Parameters
    ----------
    Ad : array_like, shape (nx, nx)
        Discrete-time state transition matrix.
    Bd : array_like, shape (nx, nu)
        Discrete-time input matrix.
    Cd : array_like, shape (ny, nx)
        Output/measurement matrix.
    Dd : array_like, shape (ny, nu), optional
        Feedthrough matrix. Default is zero matrix.
    du_index : sequence of int, optional
        Input channels carrying a bias. Default is **none** -- an input bias
        enters the dynamics and competes with the state for the same
        innovation, so it is opt-in.
    dy_index : sequence of int, optional
        Output channels carrying a bias. Default is every output, the textbook
        offset-free choice.
    Q_x : array_like, shape (nx, nx), optional
        Process noise covariance for plant states. Default is 0.1 * I.
    Q_du : array_like, shape (n_du, n_du), optional
        Process noise covariance for input bias (random walk intensity), over
        the *selected* channels. Default is 0.01 * I.
    Q_dy : array_like, shape (n_dy, n_dy), optional
        Process noise covariance for output bias (random walk intensity), over
        the *selected* channels. Default is 0.01 * I.
    R : array_like, shape (ny, ny), optional
        Measurement noise covariance. Default is 1.0 * I.
    x0 : array_like, shape (nx,) or (nx, 1), optional
        Initial plant state estimate. Default is zeros.
    du_bias0 : array_like, shape (nu,) or (nu, 1), optional
        Initial input bias estimate, at **full signal width**; channels outside
        ``du_index`` are dropped. Default is zeros.
    dy_bias0 : array_like, shape (ny,) or (ny, 1), optional
        Initial output bias estimate, at full signal width. Default is zeros.
    P0 : array_like, shape (n_aug, n_aug), optional
        Initial augmented error covariance. Default is identity.
    check_detectability : bool, optional
        Whether to reject an unidentifiable bias selection at construction.
        Default True.

    Attributes
    ----------
    nx : int
        Plant state dimension.
    nu : int
        Input dimension.
    ny : int
        Output dimension.
    n_du, n_dy : int
        Number of estimated input / output bias states.
    n_aug : int
        Total augmented state dimension (nx + n_du + n_dy).

    Examples
    --------
    >>> from neuralmpcx.util.estimators import AugmentedKalmanFilter
    >>> from neuralmpcx.util.control import mimo_tf2ss
    >>> import numpy as np
    >>>
    >>> # Simple example with explicit matrices
    >>> Ad = np.array([[0.9, 0.1], [0.0, 0.95]])
    >>> Bd = np.array([[0.0], [0.1]])
    >>> Cd = np.array([[1.0, 0.0]])
    >>>
    >>> kf = AugmentedKalmanFilter(
    ...     Ad=Ad, Bd=Bd, Cd=Cd,
    ...     dy_index=[0],
    ...     Q_x=np.eye(2) * 0.1,
    ...     Q_dy=np.eye(1) * 0.01,
    ...     R=np.eye(1) * 1.0,
    ... )
    >>>
    >>> # In MPC loop
    >>> kf.predict(u=np.array([[1.0]]))
    >>> kf.update(y=np.array([[0.5]]))
    >>> biases = kf.get_mpc_biases()

    Notes
    -----
    This filter is designed for MPC applications where plant-model mismatch
    causes offset errors. The estimated biases can be passed directly to
    the MPC solver via the ``dynamic_pars`` argument -- ``du_bias_est`` and
    ``dy_bias_est`` come back at full signal width (zero off the selection), so
    a consumer indexing by signal position needs no knowledge of the selection.

    The controller's prediction model must apply the input bias the same way the
    filter does (``x⁺ = A·x + B·(u + du_bias)``); a filter and a controller that
    disagree about where the bias enters are worse than neither having one.

    References
    ----------
    .. [1] Maciejowski, J. M. (2002). "Predictive Control with Constraints",
           Chapter 4. Prentice Hall.
    .. [2] Pannocchia, G. and Rawlings, J. B. (2003). "Disturbance models for
           offset-free model-predictive control". AIChE Journal 49(2).
    """

    def __init__(
        self,
        Ad: npt.ArrayLike,
        Bd: npt.ArrayLike,
        Cd: npt.ArrayLike,
        Dd: Optional[npt.ArrayLike] = None,
        du_index: Optional[Sequence[int]] = None,
        dy_index: Optional[Sequence[int]] = None,
        Q_x: Optional[npt.ArrayLike] = None,
        Q_du: Optional[npt.ArrayLike] = None,
        Q_dy: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        x0: Optional[npt.ArrayLike] = None,
        du_bias0: Optional[npt.ArrayLike] = None,
        dy_bias0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
        check_detectability: bool = True,
    ) -> None:
        # Convert and validate system matrices
        Ad_arr = _ensure_array(Ad, "Ad")
        Bd_arr = _ensure_array(Bd, "Bd")
        Cd_arr = _ensure_array(Cd, "Cd")

        # Validate dimensions
        if Ad_arr.ndim != 2 or Ad_arr.shape[0] != Ad_arr.shape[1]:
            raise ValueError(f"Ad must be square, got shape {Ad_arr.shape}")

        self._nx = Ad_arr.shape[0]
        self._nu = Bd_arr.shape[1]
        self._ny = Cd_arr.shape[0]

        # Bias channel selection, with the same defaults as the augmented EKF:
        # no input bias (it enters the dynamics and competes with the state for
        # the innovation, so it is opt-in), every output.
        self._du_index = _ensure_channel_index(
            du_index, self._nu, "du_index", default_all=False
        )
        self._dy_index = _ensure_channel_index(
            dy_index, self._ny, "dy_index", default_all=True
        )
        self._n_du = len(self._du_index)
        self._n_dy = len(self._dy_index)
        self._n_aug = self._nx + self._n_du + self._n_dy
        self._S_du = _scatter_matrix(self._du_index, self._nu)
        self._S_dy = _scatter_matrix(self._dy_index, self._ny)

        if Bd_arr.shape[0] != self._nx:
            raise ValueError(
                f"Bd must have {self._nx} rows to match Ad, got {Bd_arr.shape[0]}"
            )
        if Cd_arr.shape[1] != self._nx:
            raise ValueError(
                f"Cd must have {self._nx} columns to match Ad, got {Cd_arr.shape[1]}"
            )

        # Feedthrough matrix
        if Dd is None:
            Dd_arr = np.zeros((self._ny, self._nu), dtype=np.float64)
        else:
            Dd_arr = _ensure_array(Dd, "Dd")
            if Dd_arr.shape != (self._ny, self._nu):
                raise ValueError(
                    f"Dd must have shape ({self._ny}, {self._nu}), "
                    f"got {Dd_arr.shape}"
                )

        # Store original matrices
        self._Ad = Ad_arr
        self._Bd = Bd_arr
        self._Cd = Cd_arr
        self._Dd = Dd_arr

        # Construct augmented dynamics matrix A_aug. The Bd @ S_du cross-block
        # is what makes the input bias an actuator offset rather than a
        # feedthrough one: x⁺ = Ad·x + Bd·(u + S_du·δu).
        n_du, n_dy = self._n_du, self._n_dy
        self._A_aug = np.block(
            [
                [
                    Ad_arr,
                    Bd_arr @ self._S_du,
                    np.zeros((self._nx, n_dy)),
                ],
                [
                    np.zeros((n_du, self._nx)),
                    np.eye(n_du),
                    np.zeros((n_du, n_dy)),
                ],
                [
                    np.zeros((n_dy, self._nx)),
                    np.zeros((n_dy, n_du)),
                    np.eye(n_dy),
                ],
            ]
        )

        # Construct augmented input matrix B_aug
        self._B_aug = np.vstack(
            [
                Bd_arr,
                np.zeros((n_du, self._nu)),
                np.zeros((n_dy, self._nu)),
            ]
        )

        # Construct augmented output matrix C_aug
        self._C_aug = np.hstack(
            [Cd_arr, Dd_arr @ self._S_du, self._S_dy]
        )
        self._D_aug = Dd_arr

        # Process noise covariances
        if Q_x is None:
            Q_x_arr = np.eye(self._nx, dtype=np.float64) * 0.1
        else:
            Q_x_arr = _ensure_array(Q_x, "Q_x")
            if Q_x_arr.shape != (self._nx, self._nx):
                raise ValueError(
                    f"Q_x must have shape ({self._nx}, {self._nx}), "
                    f"got {Q_x_arr.shape}"
                )

        # The bias covariances are over the *selected* channels, not the full
        # signal width -- the augmented state is compact.
        if Q_du is None:
            Q_du_arr = np.eye(n_du, dtype=np.float64) * 0.01
        else:
            Q_du_arr = _ensure_array(Q_du, "Q_du")
            if Q_du_arr.shape != (n_du, n_du):
                raise ValueError(
                    f"Q_du must have shape ({n_du}, {n_du}), "
                    f"got {Q_du_arr.shape} (the covariance is over the "
                    "selected bias channels, not the full signal width)"
                )

        if Q_dy is None:
            Q_dy_arr = np.eye(n_dy, dtype=np.float64) * 0.01
        else:
            Q_dy_arr = _ensure_array(Q_dy, "Q_dy")
            if Q_dy_arr.shape != (n_dy, n_dy):
                raise ValueError(
                    f"Q_dy must have shape ({n_dy}, {n_dy}), "
                    f"got {Q_dy_arr.shape} (the covariance is over the "
                    "selected bias channels, not the full signal width)"
                )

        # Construct augmented process noise covariance Q_aug
        self._Q_aug = np.block(
            [
                [
                    Q_x_arr,
                    np.zeros((self._nx, n_du)),
                    np.zeros((self._nx, n_dy)),
                ],
                [
                    np.zeros((n_du, self._nx)),
                    Q_du_arr,
                    np.zeros((n_du, n_dy)),
                ],
                [
                    np.zeros((n_dy, self._nx)),
                    np.zeros((n_dy, n_du)),
                    Q_dy_arr,
                ],
            ]
        )

        # Measurement noise covariance
        if R is None:
            self._R = np.eye(self._ny, dtype=np.float64) * 1.0
        else:
            self._R = _ensure_array(R, "R")
            if self._R.shape != (self._ny, self._ny):
                raise ValueError(
                    f"R must have shape ({self._ny}, {self._ny}), got {self._R.shape}"
                )

        # Initialize augmented state estimate
        self._z_est = self._augmented_state(x0, du_bias0, dy_bias0)

        # Initialize error covariance
        if P0 is None:
            self._P = np.eye(self._n_aug, dtype=np.float64)
        else:
            self._P = _ensure_array(P0, "P0")
            if self._P.shape != (self._n_aug, self._n_aug):
                raise ValueError(
                    f"P0 must have shape ({self._n_aug}, {self._n_aug}), "
                    f"got {self._P.shape}"
                )

        if check_detectability:
            self._assert_detectable()

    def _augmented_state(
        self,
        x0: Optional[npt.ArrayLike],
        du_bias0: Optional[npt.ArrayLike],
        dy_bias0: Optional[npt.ArrayLike],
    ) -> npt.NDArray[np.floating]:
        """Stack the initial estimates into ``z``, gathering the bias seeds.

        The bias seeds come in at full signal width so that
        :attr:`du_bias_est` / :attr:`dy_bias_est` can be handed straight back;
        channels outside the selection are dropped here.
        """
        if x0 is None:
            x0_arr = np.zeros((self._nx, 1), dtype=np.float64)
        else:
            x0_arr = _ensure_column_vector(x0, self._nx, "x0")

        if du_bias0 is None:
            du_arr = np.zeros((self._n_du, 1), dtype=np.float64)
        else:
            full = _ensure_column_vector(du_bias0, self._nu, "du_bias0")
            du_arr = full[list(self._du_index)].reshape(self._n_du, 1)

        if dy_bias0 is None:
            dy_arr = np.zeros((self._n_dy, 1), dtype=np.float64)
        else:
            full = _ensure_column_vector(dy_bias0, self._ny, "dy_bias0")
            dy_arr = full[list(self._dy_index)].reshape(self._n_dy, 1)

        return np.vstack([x0_arr, du_arr, dy_arr])

    # ------------------------------------------------------------------
    # detectability
    # ------------------------------------------------------------------

    def detectability_report(self) -> dict[str, Union[bool, int, tuple[int, ...]]]:
        """Whether the requested bias augmentation can be estimated.

        Thin wrapper over :func:`bias_detectability` at this filter's matrices;
        see it for what each key means.
        """
        return bias_detectability(
            self._Ad, self._Bd, self._Cd, self._du_index, self._dy_index
        )

    def _assert_detectable(self) -> None:
        """Raise if the requested bias augmentation cannot be estimated."""
        if not (self._n_du or self._n_dy):
            return
        report = self.detectability_report()
        _warn_plant_rank_deficiency(report, stacklevel=4)
        message = _detectability_error(report, self._du_index, self._dy_index)
        if message is not None:
            raise ValueError(message)

    @property
    def nx(self) -> int:
        """Plant state dimension."""
        return self._nx

    @property
    def nu(self) -> int:
        """Input dimension."""
        return self._nu

    @property
    def ny(self) -> int:
        """Output dimension."""
        return self._ny

    @property
    def n_du(self) -> int:
        """Number of estimated input-bias states."""
        return self._n_du

    @property
    def n_dy(self) -> int:
        """Number of estimated output-bias states."""
        return self._n_dy

    @property
    def n_aug(self) -> int:
        """Total augmented state dimension (nx + n_du + n_dy)."""
        return self._n_aug

    @property
    def du_index(self) -> tuple[int, ...]:
        """Input channels carrying a bias state."""
        return self._du_index

    @property
    def dy_index(self) -> tuple[int, ...]:
        """Output channels carrying a bias state."""
        return self._dy_index

    @property
    def x_est(self) -> npt.NDArray[np.floating]:
        """Plant state estimate, shape (nx, 1)."""
        return self._z_est[: self._nx].copy()

    @property
    def du_bias_est(self) -> npt.NDArray[np.floating]:
        """Input bias estimate at **full** signal width, shape (nu, 1).

        Scattered back out of the compact augmented state, so an unselected
        channel reads a hard zero and a consumer indexing by signal position
        needs no knowledge of the selection.
        """
        return self._S_du @ self._z_est[self._nx : self._nx + self._n_du]

    @property
    def dy_bias_est(self) -> npt.NDArray[np.floating]:
        """Output bias estimate at **full** signal width, shape (ny, 1)."""
        return self._S_dy @ self._z_est[self._nx + self._n_du :]

    @property
    def z_est(self) -> npt.NDArray[np.floating]:
        """Full augmented state estimate, shape (n_aug, 1) -- biases compact."""
        return self._z_est.copy()

    @property
    def P(self) -> npt.NDArray[np.floating]:
        """Current error covariance, shape (n_aug, n_aug)."""
        return self._P.copy()

    def predict(self, u: npt.ArrayLike) -> None:
        """Time update (prediction step).

        Propagates augmented state estimate and covariance forward one time
        step using the augmented system dynamics.

        Parameters
        ----------
        u : array_like, shape (nu,) or (nu, 1)
            Control input applied at the current time step.

        Notes
        -----
        Updates::

            z_pred = A_aug @ z_est + B_aug @ u
            P_pred = A_aug @ P @ A_aug.T + Q_aug

        The bias states use identity dynamics (random walk model), so they
        remain constant during prediction with noise added through Q_aug. The
        input bias does move the *state*, through ``A_aug``'s ``Bd @ S_du``
        block: the plant row is ``x⁺ = Ad·x + Bd·(u + S_du·δu)``.
        """
        u_vec = _ensure_column_vector(u, self._nu, "u")

        # Augmented state prediction
        self._z_est = self._A_aug @ self._z_est + self._B_aug @ u_vec

        # Covariance prediction
        self._P = self._A_aug @ self._P @ self._A_aug.T + self._Q_aug

    def update(self, y: npt.ArrayLike, u: Optional[npt.ArrayLike] = None) -> None:
        """Measurement update (correction step).

        Corrects the predicted augmented state using the measurement.

        Parameters
        ----------
        y : array_like, shape (ny,) or (ny, 1)
            Measured output (in deviation form).
        u : array_like, shape (nu,) or (nu, 1), optional
            Control input. Required if Dd is nonzero.

        Notes
        -----
        Updates::

            y_pred = C_aug @ z_pred + D_aug @ u
            K = P_pred @ C_aug.T @ inv(C_aug @ P_pred @ C_aug.T + R)
            z_est = z_pred + K @ (y - y_pred)
            P = (I - K @ C_aug) @ P_pred
        """
        y_vec = _ensure_column_vector(y, self._ny, "y")

        # Predicted measurement
        y_pred = self._C_aug @ self._z_est
        if u is not None:
            u_vec = _ensure_column_vector(u, self._nu, "u")
            y_pred = y_pred + self._D_aug @ u_vec

        # Measurement residual (innovation)
        y_res = y_vec - y_pred

        # Innovation covariance
        S = self._C_aug @ self._P @ self._C_aug.T + self._R

        # Kalman gain
        K = self._P @ self._C_aug.T @ np.linalg.inv(S)

        # Augmented state estimate update
        self._z_est = self._z_est + K @ y_res

        # Error covariance update
        I = np.eye(self._n_aug, dtype=np.float64)
        self._P = (I - K @ self._C_aug) @ self._P

    def get_mpc_biases(self) -> dict[str, npt.NDArray[np.floating]]:
        """Get estimated biases in MPC-compatible format.

        Returns a dictionary ready to be passed to ``Mpc.solve_mpc()``
        via the ``dynamic_pars`` argument.

        Returns
        -------
        dict
            Dictionary with keys ``"du_bias"`` and ``"dy_bias"``,
            each mapping to the corresponding bias estimate as
            a column vector.

        Examples
        --------
        >>> biases = kf.get_mpc_biases()
        >>> u_opt = mpc.solve_mpc(
        ...     state, state_context, state_indices, action_context,
        ...     setpoint, dynamic_pars=biases
        ... )
        """
        return {
            "du_bias": self.du_bias_est,
            "dy_bias": self.dy_bias_est,
        }

    def reset(
        self,
        x0: Optional[npt.ArrayLike] = None,
        du_bias0: Optional[npt.ArrayLike] = None,
        dy_bias0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Reset filter to initial conditions.

        Parameters
        ----------
        x0 : array_like, optional
            New plant state estimate. If None, resets to zeros.
        du_bias0 : array_like, shape (nu,), optional
            New input bias estimate at **full** signal width; channels outside
            ``du_index`` are dropped. If None, resets to zeros.
        dy_bias0 : array_like, shape (ny,), optional
            New output bias estimate at full signal width. If None, resets to
            zeros.
        P0 : array_like, optional
            New error covariance. If None, resets to identity.
        """
        self._z_est = self._augmented_state(x0, du_bias0, dy_bias0)

        if P0 is None:
            self._P = np.eye(self._n_aug, dtype=np.float64)
        else:
            self._P = _ensure_array(P0, "P0")
            if self._P.shape != (self._n_aug, self._n_aug):
                raise ValueError(
                    f"P0 must have shape ({self._n_aug}, {self._n_aug}), "
                    f"got {self._P.shape}"
                )


class AugmentedExtendedKalmanFilter:
    """Augmented extended Kalman filter for joint state and bias estimation.

    The nonlinear counterpart of :class:`AugmentedKalmanFilter`: it estimates
    the plant state of a nonlinear system together with constant-in-expectation
    input and output biases, so that an MPC built on the same model tracks
    without steady-state offset under plant-model mismatch.

    Augmented state vector::

        z = [x; du_bias; dy_bias]

    Augmented system::

        x[k+1]       = f(x[k], u[k] + S_du @ du_bias[k]) + w_x[k]
        du_bias[k+1] = du_bias[k] + w_du[k]
        dy_bias[k+1] = dy_bias[k] + w_dy[k]
        y[k]         = h(x[k]) + S_dy @ dy_bias[k] + v[k]

    where the biases are random walks and ``S_du`` / ``S_dy`` scatter the
    selected bias channels into the full input / output width.

    The augmented Jacobians are derived from ``f`` and ``h`` by CasADi
    algorithmic differentiation, so the ``df/du @ S_du`` cross-block through
    which the input bias reaches the state is exact.

    Two things differ from :class:`AugmentedKalmanFilter`, both deliberate:

    1. **The input bias enters the dynamics.** In the linear filter ``du_bias``
       only reaches the output equation through the feedthrough matrix ``Dd``,
       so it does nothing at all when ``Dd == 0``. Here it biases the control
       that drives ``f``, which is what an actuator offset physically is.
    2. **Bias channels are selected, not assumed.** With ``n_bias`` integrating
       bias states the augmented system is detectable only if
       ``rank([[I - A, -B_d], [C, C_d]]) == nx + n_bias``, and that matrix has
       only ``nx + ny`` rows, so **at most ``ny`` bias states can ever be
       estimated** -- biasing every input *and* every output is undetectable
       for any plant with at least one input. Beyond the limit the filter still
       returns numbers, but the split between input and output bias is decided
       by the ratio of ``Q_du`` to ``Q_dy`` rather than by the measurements.
       The condition is checked at construction; see ``check_detectability``
       and :meth:`detectability_report`.

    Parameters
    ----------
    f : casadi.Function
        Discrete-time state transition map with signature ``f(x, u) -> x_next``.
        Multi-output functions are accepted and the next state is taken to be
        the first output, so ``mpc.dynamics`` can be passed directly (same
        convention as :class:`ExtendedKalmanFilter`).
    h : casadi.Function or array_like
        Measurement map: either ``h(x) -> y`` or a measurement matrix ``C`` of
        shape (ny, nx). ``ny`` counts *measured* channels, which is what the
        bias budget is limited by.
    du_index : sequence of int, optional
        Input channels carrying a bias, as positions in ``range(nu)``. Default
        is no input bias -- input bias is opt-in because every channel spends
        from the same budget as the output biases.
    dy_index : sequence of int, optional
        Output channels carrying a bias, as positions in ``range(ny)``. Default
        is every output channel, the textbook offset-free choice
        (``n_bias == ny``).
    Q_x : array_like, shape (nx, nx), optional
        Process noise covariance for the plant states. Default is 0.1 * I.
    Q_du : array_like, shape (n_du, n_du), optional
        Random-walk intensity of the input bias, over the *selected* channels
        only. Default is 0.01 * I.
    Q_dy : array_like, shape (n_dy, n_dy), optional
        Random-walk intensity of the output bias, over the *selected* channels
        only. Default is 0.01 * I.
    R : array_like, shape (ny, ny), optional
        Measurement noise covariance. Default is 1.0 * I.
    x0 : array_like, shape (nx,) or (nx, 1), optional
        Initial plant state estimate. Default is zeros.
    du_bias0 : array_like, shape (nu,) or (nu, 1), optional
        Initial input bias, given at **full input width**; entries outside
        ``du_index`` are ignored. Full width keeps it symmetric with
        :attr:`du_bias_est`, so a filter's estimate can be handed straight back
        to :meth:`reset`. Default is zeros.
    dy_bias0 : array_like, shape (ny,) or (ny, 1), optional
        Initial output bias at **full output width**, ignoring entries outside
        ``dy_index``. Default is zeros.
    P0 : array_like, shape (n_aug, n_aug), optional
        Initial augmented error covariance. Default is identity.
    u_lin : array_like, shape (nu,) or (nu, 1), optional
        Input the detectability linearization is taken at. Only ``df/du``
        depends on it. Default is zeros; pass the commissioning operating point
        when the plant is strongly nonlinear in ``u``.
    check_detectability : bool, optional
        Whether to run the rank test at construction and raise when it fails.
        Default is True. Set False only when you know the augmentation is
        unobservable and want it anyway.

    Attributes
    ----------
    nx : int
        Plant state dimension.
    nu : int
        Input dimension.
    ny : int
        Output dimension.
    n_du : int
        Number of input-bias states actually estimated.
    n_dy : int
        Number of output-bias states actually estimated.
    n_aug : int
        Total augmented state dimension (nx + n_du + n_dy).

    Examples
    --------
    >>> import casadi as cs
    >>> import numpy as np
    >>> x = cs.MX.sym("x", 2)
    >>> u = cs.MX.sym("u", 1)
    >>> f = cs.Function("f", [x, u], [x + 0.1 * cs.vertcat(x[1], -cs.sin(x[0]) + u)])
    >>> C = np.eye(2)
    >>> # both outputs measured -> a budget of 2 bias states; spend one on the
    >>> # actuator offset and one on the first output
    >>> aekf = AugmentedExtendedKalmanFilter(
    ...     f, C, du_index=[0], dy_index=[0], R=np.eye(2) * 0.1
    ... )
    >>> aekf.predict(u=np.array([[0.5]]))
    >>> aekf.update(y=np.array([[0.2], [0.1]]))
    >>> biases = aekf.get_mpc_biases()

    Notes
    -----
    The recursion itself is :class:`ExtendedKalmanFilter` running on the
    augmented model, so the two filters agree exactly wherever they describe
    the same system.

    The detectability test is a linearization at ``(x0, u_lin)`` and is
    therefore a local statement; the filter re-evaluates the true Jacobians at
    the current estimate on every step. Re-run :meth:`detectability_report` to
    check another operating point.

    References
    ----------
    .. [1] Muske, K. R. and Badgwell, T. A. (2002). "Disturbance modeling for
           offset-free linear model predictive control", Journal of Process
           Control, 12(5), 617-632.
    .. [2] Pannocchia, G. and Rawlings, J. B. (2003). "Disturbance models for
           offset-free model-predictive control", AIChE Journal, 49(2), 426-437.
    """

    def __init__(
        self,
        f: cs.Function,
        h: Union[cs.Function, npt.ArrayLike],
        du_index: Optional[Sequence[int]] = None,
        dy_index: Optional[Sequence[int]] = None,
        Q_x: Optional[npt.ArrayLike] = None,
        Q_du: Optional[npt.ArrayLike] = None,
        Q_dy: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        x0: Optional[npt.ArrayLike] = None,
        du_bias0: Optional[npt.ArrayLike] = None,
        dy_bias0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
        u_lin: Optional[npt.ArrayLike] = None,
        check_detectability: bool = True,
    ) -> None:
        # Reuse the EKF's own validation and normalization of (f, h): it
        # rejects the same malformed maps with the same messages, and hands
        # back single-output functions with x_next as the first output.
        base = ExtendedKalmanFilter(f, h)
        self._nx, self._nu, self._ny = base.nx, base.nu, base.ny
        f_plain, h_plain = base._f, base._h

        self._du_index = _ensure_channel_index(
            du_index, self._nu, "du_index", default_all=False
        )
        self._dy_index = _ensure_channel_index(
            dy_index, self._ny, "dy_index", default_all=True
        )
        self._n_du = len(self._du_index)
        self._n_dy = len(self._dy_index)
        self._n_aug = self._nx + self._n_du + self._n_dy
        self._S_du = _scatter_matrix(self._du_index, self._nu)
        self._S_dy = _scatter_matrix(self._dy_index, self._ny)

        # Plant-level Jacobians, kept for the detectability test. The filter
        # itself never uses them -- it differentiates the augmented model.
        x_sym = cs.MX.sym("x", self._nx)
        u_sym = cs.MX.sym("u", self._nu)
        x_next_plain = f_plain(x_sym, u_sym)
        y_plain = h_plain(x_sym)
        self._A_jac = cs.Function(
            "aekf_A", [x_sym, u_sym], [cs.jacobian(x_next_plain, x_sym)]
        )
        self._B_jac = cs.Function(
            "aekf_B", [x_sym, u_sym], [cs.jacobian(x_next_plain, u_sym)]
        )
        self._C_jac = cs.Function("aekf_C", [x_sym], [cs.jacobian(y_plain, x_sym)])

        # Augmented model: biased control into f, biased output out of h, and
        # random-walk dynamics for the bias states themselves. Shared with
        # MovingHorizonEstimator so the two describe the same system.
        f_aug, h_aug = _augmented_model(
            f_plain,
            h_plain,
            self._S_du,
            self._S_dy,
            self._nx,
            self._n_du,
            self._n_dy,
        )
        self._f_aug, self._h_aug = f_aug, h_aug

        Q_aug = self._Q_aug = self._augmented_process_noise(Q_x, Q_du, Q_dy)
        z0 = self._augmented_state(x0, du_bias0, dy_bias0)

        self._u_lin = (
            np.zeros((self._nu, 1), dtype=np.float64)
            if u_lin is None
            else _ensure_column_vector(u_lin, self._nu, "u_lin")
        )

        self._ekf = ExtendedKalmanFilter(f=f_aug, h=h_aug, Q=Q_aug, R=R, x0=z0, P0=P0)

        if check_detectability:
            self._assert_detectable()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _augmented_process_noise(
        self,
        Q_x: Optional[npt.ArrayLike],
        Q_du: Optional[npt.ArrayLike],
        Q_dy: Optional[npt.ArrayLike],
    ) -> npt.NDArray[np.floating]:
        """Assemble ``blkdiag(Q_x, Q_du, Q_dy)`` with validation and defaults."""
        blocks = []
        for value, n, default, name in (
            (Q_x, self._nx, 0.1, "Q_x"),
            (Q_du, self._n_du, 0.01, "Q_du"),
            (Q_dy, self._n_dy, 0.01, "Q_dy"),
        ):
            if value is None:
                blocks.append(np.eye(n, dtype=np.float64) * default)
                continue
            arr = _ensure_array(value, name)
            if arr.shape != (n, n):
                extra = (
                    " (the covariance is over the selected bias channels, not "
                    "the full signal width)"
                    if name in ("Q_du", "Q_dy")
                    else ""
                )
                raise ValueError(
                    f"{name} must have shape ({n}, {n}), got {arr.shape}{extra}"
                )
            blocks.append(arr)

        Q_aug = np.zeros((self._n_aug, self._n_aug), dtype=np.float64)
        off = 0
        for block in blocks:
            n = block.shape[0]
            Q_aug[off : off + n, off : off + n] = block
            off += n
        return Q_aug

    def _augmented_state(
        self,
        x0: Optional[npt.ArrayLike],
        du_bias0: Optional[npt.ArrayLike],
        dy_bias0: Optional[npt.ArrayLike],
    ) -> npt.NDArray[np.floating]:
        """Stack the initial estimates into ``z``, gathering the bias seeds.

        The bias seeds come in at full signal width so that
        :attr:`du_bias_est` / :attr:`dy_bias_est` can be handed straight back;
        channels outside the selection are dropped here.
        """
        if x0 is None:
            x0_arr = np.zeros((self._nx, 1), dtype=np.float64)
        else:
            x0_arr = _ensure_column_vector(x0, self._nx, "x0")

        if du_bias0 is None:
            du_arr = np.zeros((self._n_du, 1), dtype=np.float64)
        else:
            full = _ensure_column_vector(du_bias0, self._nu, "du_bias0")
            du_arr = full[list(self._du_index)].reshape(self._n_du, 1)

        if dy_bias0 is None:
            dy_arr = np.zeros((self._n_dy, 1), dtype=np.float64)
        else:
            full = _ensure_column_vector(dy_bias0, self._ny, "dy_bias0")
            dy_arr = full[list(self._dy_index)].reshape(self._n_dy, 1)

        return np.vstack([x0_arr, du_arr, dy_arr])

    # ------------------------------------------------------------------
    # detectability
    # ------------------------------------------------------------------

    def detectability_report(self) -> dict[str, Union[bool, int, tuple[int, ...]]]:
        """Check whether the bias augmentation is detectable at the current estimate.

        Linearizes the plant and hands the Jacobians to
        :func:`bias_detectability`, which is also what the linear
        :class:`AugmentedKalmanFilter` uses -- the two therefore accept and
        refuse exactly the same selections.

        Returns
        -------
        dict
            See :func:`bias_detectability`.

        Notes
        -----
        The Jacobians are taken at the current state estimate and at the
        ``u_lin`` given to the constructor, so this is a local statement about
        one operating point.
        """
        x, u = self.x_est, self._u_lin
        return bias_detectability(
            self._A_jac(x, u).full(),
            self._B_jac(x, u).full(),
            self._C_jac(x).full(),
            self._du_index,
            self._dy_index,
        )

    def _assert_detectable(self) -> None:
        """Raise if the requested bias augmentation cannot be estimated."""
        if not (self._n_du or self._n_dy):
            return
        report = self.detectability_report()
        _warn_plant_rank_deficiency(report, stacklevel=4)
        message = _detectability_error(report, self._du_index, self._dy_index)
        if message is not None:
            raise ValueError(f"{message} (evaluated at x0/u_lin)")

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def nx(self) -> int:
        """Plant state dimension."""
        return self._nx

    @property
    def nu(self) -> int:
        """Input dimension."""
        return self._nu

    @property
    def ny(self) -> int:
        """Output dimension."""
        return self._ny

    @property
    def n_du(self) -> int:
        """Number of estimated input-bias states."""
        return self._n_du

    @property
    def n_dy(self) -> int:
        """Number of estimated output-bias states."""
        return self._n_dy

    @property
    def n_aug(self) -> int:
        """Total augmented state dimension (nx + n_du + n_dy)."""
        return self._n_aug

    @property
    def du_index(self) -> tuple[int, ...]:
        """Input channels carrying a bias."""
        return self._du_index

    @property
    def dy_index(self) -> tuple[int, ...]:
        """Output channels carrying a bias."""
        return self._dy_index

    @property
    def x_est(self) -> npt.NDArray[np.floating]:
        """Plant state estimate, shape (nx, 1)."""
        return self._ekf.x_est[: self._nx]

    @property
    def du_bias_est(self) -> npt.NDArray[np.floating]:
        """Input bias estimate at full input width, shape (nu, 1).

        Channels outside :attr:`du_index` are zero, so the estimate can be
        added to a control vector or handed to an MPC without the consumer
        having to know which channels are estimated.
        """
        return self._S_du @ self._ekf.x_est[self._nx : self._nx + self._n_du]

    @property
    def dy_bias_est(self) -> npt.NDArray[np.floating]:
        """Output bias estimate at full output width, shape (ny, 1).

        Channels outside :attr:`dy_index` are zero.
        """
        return self._S_dy @ self._ekf.x_est[self._nx + self._n_du :]

    @property
    def z_est(self) -> npt.NDArray[np.floating]:
        """Full augmented state estimate, shape (n_aug, 1).

        Unlike :attr:`du_bias_est` / :attr:`dy_bias_est` the bias entries here
        are compact -- only the estimated channels, in ``du_index`` /
        ``dy_index`` order, matching the rows and columns of :attr:`P`.
        """
        return self._ekf.x_est

    @property
    def P(self) -> npt.NDArray[np.floating]:
        """Current augmented error covariance, shape (n_aug, n_aug)."""
        return self._ekf.P

    # ------------------------------------------------------------------
    # recursion
    # ------------------------------------------------------------------

    def predict(self, u: npt.ArrayLike) -> None:
        """Time update (prediction step).

        Parameters
        ----------
        u : array_like, shape (nu,) or (nu, 1)
            Control input applied at the current time step, *without* the
            estimated bias -- the filter adds its own estimate before calling
            ``f``.

        Notes
        -----
        The biases hold across the prediction (random walk) and gain
        uncertainty through ``Q_du`` / ``Q_dy``; the plant state is rolled
        through ``f(x, u + S_du @ du_bias)``.
        """
        self._ekf.predict(u)

    def update(self, y: npt.ArrayLike, u: Optional[npt.ArrayLike] = None) -> None:
        """Measurement update (correction step).

        Parameters
        ----------
        y : array_like, shape (ny,) or (ny, 1)
            Measured output.
        u : array_like, optional
            Unused; kept for API symmetry with the other filters. The augmented
            measurement map ``h(x) + S_dy @ dy_bias`` has no feedthrough.
        """
        self._ekf.update(y)

    def get_mpc_biases(self) -> dict[str, npt.NDArray[np.floating]]:
        """Get estimated biases in MPC-compatible format.

        Returns a dictionary ready to be passed to ``Mpc.solve_mpc()`` via the
        ``dynamic_pars`` argument, with the same keys and full-width shapes as
        :meth:`AugmentedKalmanFilter.get_mpc_biases`.

        Returns
        -------
        dict
            ``"du_bias"`` of shape (nu, 1) and ``"dy_bias"`` of shape (ny, 1).

        Examples
        --------
        >>> u_opt = mpc.solve_mpc(
        ...     state=aekf.x_est, state_indices=state_indices, setpoint=sp,
        ...     dynamic_pars=aekf.get_mpc_biases(),
        ... )
        """
        return {
            "du_bias": self.du_bias_est,
            "dy_bias": self.dy_bias_est,
        }

    def reset(
        self,
        x0: Optional[npt.ArrayLike] = None,
        du_bias0: Optional[npt.ArrayLike] = None,
        dy_bias0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Reset filter to initial conditions.

        Parameters
        ----------
        x0 : array_like, optional
            New plant state estimate. If None, resets to zeros.
        du_bias0 : array_like, shape (nu,) or (nu, 1), optional
            New input bias at full input width. If None, resets to zeros.
        dy_bias0 : array_like, shape (ny,) or (ny, 1), optional
            New output bias at full output width. If None, resets to zeros.
        P0 : array_like, shape (n_aug, n_aug), optional
            New augmented covariance. If None, resets to identity.
        """
        self._ekf.reset(x0=self._augmented_state(x0, du_bias0, dy_bias0), P0=P0)


def _bound_vector(
    value: Optional[npt.ArrayLike], n: int, fill: float, name: str
) -> npt.NDArray[np.floating]:
    """Resolve an optional bound to a length-``n`` column vector."""
    if value is None:
        return np.full((n, 1), fill, dtype=np.float64)
    return _ensure_column_vector(value, n, name)


def _check_bounds(
    lb: npt.NDArray[np.floating],
    ub: npt.NDArray[np.floating],
    lb_name: str,
    ub_name: str,
) -> None:
    """Raise if any channel has lower bound above upper bound."""
    crossed = np.nonzero((lb > ub).ravel())[0]
    if crossed.size:
        raise ValueError(
            f"{lb_name} must be <= {ub_name} on every channel; "
            f"channel(s) {crossed.tolist()} are crossed"
        )


class MovingHorizonEstimator:
    """Constrained moving-horizon estimator (MHE) for state and bias estimation.

    The constrained counterpart of :class:`AugmentedExtendedKalmanFilter`. Where
    a Kalman filter corrects the state with a single unconstrained linear
    update, this estimator re-solves a small least-squares problem over the last
    ``horizon`` measurements at every step. Two things follow:

    1. **Bounds are enforced exactly.** An EKF will cheerfully report a negative
       concentration or a bias larger than the actuator range, because nothing
       in its update knows about them. Here ``x_lb`` / ``x_ub`` and the bias
       bounds are constraints on the estimate itself.
    2. **The nonlinear model is used as written across the whole window**, rather
       than linearized once per step about a single point.

    Like the augmented filters it estimates the plant state together with
    user-selected input and output biases, so an MPC built on the same model
    tracks without steady-state offset under plant-model mismatch.

    Augmented state vector (biases *compact* -- only the selected channels)::

        z = [x; du_bias; dy_bias]

    Estimation problem solved at each step, over the window states
    ``Z = [z_0 ... z_N]``::

        min  ||z_a - z_arr||^2_{P_arr^-1}                        (arrival cost)
         Z   + sum_j ||z_{j+1} - f_aug(z_j, u_j)||^2_{Q_aug^-1}  (process)
             + sum_j ||y_j - h_aug(z_j)||^2_{R^-1}               (measurement)
        s.t. lb <= z_j <= ub

    where ``f_aug(z, u) = [f(x, u + S_du @ du); du; dy]`` and
    ``h_aug(z) = h(x) + S_dy @ dy`` are exactly the augmented maps
    :class:`AugmentedExtendedKalmanFilter` uses, and ``a`` is the oldest filled
    slot.

    The process and measurement noises are *eliminated* rather than declared:
    since ``w_j`` and ``v_j`` are free, substituting them into the cost is an
    exact reformulation that removes every equality constraint. What remains is
    a bound-constrained least-squares problem, which is both much smaller and
    **never infeasible** -- a textbook multiple-shooting MHE with hard dynamics
    equalities can be, and then a control loop has nothing to return.

    Parameters
    ----------
    f : casadi.Function
        Discrete-time state transition map ``f(x, u) -> x_next``. Multi-output
        functions are accepted and the next state is taken to be the first
        output, so ``mpc.dynamics`` can be passed directly (same convention as
        :class:`ExtendedKalmanFilter`).
    h : casadi.Function or array_like
        Measurement map: either ``h(x) -> y`` or a measurement matrix ``C`` of
        shape (ny, nx).
    horizon : int, optional
        Number of past intervals ``N`` in the window; the problem carries
        ``N + 1`` states. Default is 10. See Notes on choosing it.
    du_index : sequence of int, optional
        Input channels carrying a bias. Default is none.
    dy_index : sequence of int, optional
        Output channels carrying a bias. Default is every output channel.
    Q_x, Q_du, Q_dy, R, P0 : array_like, optional
        **Covariances**, with the same shapes, meanings and defaults as
        :class:`AugmentedExtendedKalmanFilter`. See the warning below.
    x0, du_bias0, dy_bias0 : array_like, optional
        Initial estimates; the bias seeds are at full signal width.
    x_lb, x_ub : array_like, shape (nx,), optional
        Bounds on the plant state. Default is unbounded.
    du_bias_lb, du_bias_ub : array_like, shape (nu,), optional
        Bounds on the input bias, at **full input width**; entries outside
        ``du_index`` are ignored, so ``du_bias_lb=mhe.du_bias_est`` round-trips
        like the seeds do. Default is unbounded.
    dy_bias_lb, dy_bias_ub : array_like, shape (ny,), optional
        Bounds on the output bias, at full output width. Default is unbounded.
    arrival_cost : {'ekf', 'constant'}, optional
        How the window's left edge is anchored. Default is ``'ekf'``; see Notes.
    on_solver_failure : {'fallback', 'raise', 'accept'}, optional
        What to do when the solve does not converge. ``'fallback'`` (default)
        uses the companion filter's estimate, ``'raise'`` raises, ``'accept'``
        takes the unconverged iterate. A non-finite iterate is never accepted
        under any setting.
    solver : str, optional
        CasADi ``nlpsol`` plugin. Default ``'ipopt'``.
    solver_opts : dict, optional
        Options merged over the defaults (which silence the solver and set
        ``bound_relax_factor`` to 0; see Notes).
    expand : bool, optional
        Expand the problem to SX for speed. Default True, with an automatic
        fallback to MX when the graph cannot be expanded.
    u_lin : array_like, optional
        Input the detectability linearization is taken at.
    check_detectability : bool, optional
        Run the rank test at construction. Default True.

    Attributes
    ----------
    nx, nu, ny : int
        Plant state, input and output dimensions.
    n_du, n_dy, n_aug : int
        Bias-state counts and the total augmented dimension.

    Warnings
    --------
    ``Q_x``, ``Q_du``, ``Q_dy``, ``R`` and ``P0`` are **covariances**, matching
    every other estimator in this module. The moving-horizon literature (and
    do-mpc's ``P_x`` / ``P_v`` / ``P_w``) states the same cost with *inverse*
    covariance weights, so tuning copied from a paper must be inverted before it
    is passed here. Keeping the filter convention means an
    :class:`AugmentedExtendedKalmanFilter`'s tuning transfers unchanged.

    Unlike the filters, the covariances must be **positive definite**, not
    merely positive semi-definite: the cost weights by their inverse. In
    particular ``Q_du=np.zeros((1, 1))`` is a legal augmented-filter input and
    an illegal one here.

    Examples
    --------
    >>> import casadi as cs
    >>> import numpy as np
    >>> x = cs.MX.sym("x", 2)
    >>> u = cs.MX.sym("u", 1)
    >>> f = cs.Function("f", [x, u], [x + 0.1 * cs.vertcat(x[1], -cs.sin(x[0]) + u)])
    >>> C = np.eye(2)
    >>> mhe = MovingHorizonEstimator(
    ...     f, C, horizon=5, du_index=[0], dy_index=[0],
    ...     R=np.eye(2) * 0.1, x_lb=[-np.inf, -2.0], x_ub=[np.inf, 2.0],
    ... )
    >>> mhe.predict(u=np.array([[0.5]]))
    >>> mhe.update(y=np.array([[0.2], [0.1]]))
    >>> biases = mhe.get_mpc_biases()

    Notes
    -----
    **Runtime contract.** ``predict(u)`` then ``update(y)`` on every cycle, in
    that order, exactly like the filters -- but here the order is mandatory
    rather than conventional, because the window needs the input that drove the
    plant into the measurement. ``update`` without a preceding ``predict``
    raises.

    **Arrival cost.** The window has to be told what it inherits from the data
    that has already scrolled off it. With ``arrival_cost='ekf'`` a companion
    :class:`AugmentedExtendedKalmanFilter` runs in lockstep, and the anchor is
    its *one-step-predicted* mean and covariance from the cycle the window's
    oldest slot belongs to -- conditioned on data strictly older than the
    window. That lag is the whole trick: anchoring on a filter that has already
    seen the in-window measurements counts every one of them twice, which
    silently collapses the estimator onto the EKF while making :attr:`P` look
    far better than it is. With this anchoring, and no active bounds, the
    estimator reproduces the companion filter *exactly* on a linear model for
    any ``horizon``. ``arrival_cost='constant'`` pins the weight at ``P0`` and
    anchors on the previous solution; it is a heuristic, and it needs a longer
    horizon so the frozen weight does not dominate.

    **Choosing the horizon.** With ``arrival_cost='ekf'`` correctness does not
    require a long window -- ``horizon=1`` already reproduces the Kalman filter
    on a linear model. The horizon buys constraint handling and robustness to
    nonlinearity, not consistency. A practical starting point is the dominant
    time constant divided by the sample time, clipped to roughly 5..30.

    **Cost per cycle.** A filter evaluates the model once and its Jacobian once.
    This estimator evaluates the model ``horizon`` times per solver iteration,
    with second derivatives, so budget two orders of magnitude more model work.
    If that does not fit the control cycle, pass
    ``solver_opts={'ipopt': {'hessian_approximation': 'limited-memory'}}``, or
    reduce the horizon. Note also that in a typical control loop the solve lands
    in the same cycle as the MPC solve.

    **Scaling matters more than it does for a filter.** A Kalman filter is
    invariant to a diagonal rescaling of the state; an interior-point method is
    emphatically not. States of order 1 are a requirement here, not a nicety.

    **Warm starting.** The previous solution, shifted by one step, is used as
    the initial guess. Interior-point methods do not exploit that the way an SQP
    would -- expect roughly a factor of two, not ten. The opt-in recipe is
    ``{'ipopt': {'warm_start_init_point': 'yes', 'mu_init': 1e-4,
    'warm_start_bound_push': 1e-6, 'warm_start_mult_bound_push': 1e-6}}``, with
    the caveat that a *bad* warm start under those options is worse than a cold
    solve. ``solver='fatrop'`` or ``'sqpmethod'`` are alternatives.

    **Local minima.** With a nonlinear ``f`` the solver returns a local
    minimum and can hop between minima from cycle to cycle, which shows up as
    chatter in the estimate that looks like measurement noise but is not.
    :attr:`last_cost` reveals it: a good objective followed by a worse one on
    similar data is a hop, not noise.

    **Monitoring.** :attr:`last_cost` is a better divergence alarm than the
    innovation. Under model mismatch this estimator distributes the error across
    the window instead of concentrating it in one innovation, and active bounds
    clip the residual, so the innovation can look healthy while the estimate is
    wrong.

    **The detectability budget still binds.** At most ``ny`` bias states can be
    estimated, exactly as for :class:`AugmentedExtendedKalmanFilter`: the
    condition is about the augmented model's structure, and a longer window does
    not buy observability.

    References
    ----------
    .. [1] Rao, C. V., Rawlings, J. B. and Lee, J. H. (2001). "Constrained
           linear state estimation -- a moving horizon approach", Automatica,
           37(10), 1619-1628.
    .. [2] Rawlings, J. B., Mayne, D. Q. and Diehl, M. M. (2017). "Model
           Predictive Control: Theory, Computation, and Design", Chapter 4.
           Nob Hill Publishing.
    """

    _ARRIVAL_MODES = ("ekf", "constant")
    _FAILURE_MODES = ("fallback", "raise", "accept")

    def __init__(
        self,
        f: cs.Function,
        h: Union[cs.Function, npt.ArrayLike],
        horizon: int = 10,
        du_index: Optional[Sequence[int]] = None,
        dy_index: Optional[Sequence[int]] = None,
        Q_x: Optional[npt.ArrayLike] = None,
        Q_du: Optional[npt.ArrayLike] = None,
        Q_dy: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        x0: Optional[npt.ArrayLike] = None,
        du_bias0: Optional[npt.ArrayLike] = None,
        dy_bias0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
        x_lb: Optional[npt.ArrayLike] = None,
        x_ub: Optional[npt.ArrayLike] = None,
        du_bias_lb: Optional[npt.ArrayLike] = None,
        du_bias_ub: Optional[npt.ArrayLike] = None,
        dy_bias_lb: Optional[npt.ArrayLike] = None,
        dy_bias_ub: Optional[npt.ArrayLike] = None,
        arrival_cost: str = "ekf",
        on_solver_failure: str = "fallback",
        solver: str = "ipopt",
        solver_opts: Optional[dict] = None,
        expand: bool = True,
        u_lin: Optional[npt.ArrayLike] = None,
        check_detectability: bool = True,
    ) -> None:
        # The companion filter is built first and does double duty: it validates
        # (f, h), the channel selection, every covariance shape and the
        # detectability budget -- with exactly the messages the augmented filter
        # raises, so the two stay in step -- and at runtime it supplies the
        # arrival anchor, the covariance and the solver-failure fallback.
        self._aekf = AugmentedExtendedKalmanFilter(
            f=f,
            h=h,
            du_index=du_index,
            dy_index=dy_index,
            Q_x=Q_x,
            Q_du=Q_du,
            Q_dy=Q_dy,
            R=R,
            x0=x0,
            du_bias0=du_bias0,
            dy_bias0=dy_bias0,
            P0=P0,
            u_lin=u_lin,
            check_detectability=check_detectability,
        )
        self._f_orig, self._h_orig = f, h
        self._u_lin_arg = u_lin

        self._nx = self._aekf.nx
        self._nu = self._aekf.nu
        self._ny = self._aekf.ny
        self._n_du = self._aekf.n_du
        self._n_dy = self._aekf.n_dy
        self._n_aug = self._aekf.n_aug
        self._S_du = self._aekf._S_du
        self._S_dy = self._aekf._S_dy
        self._f_aug = self._aekf._f_aug
        self._h_aug = self._aekf._h_aug

        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool):
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
        if horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")
        self._N = int(horizon)

        if arrival_cost not in self._ARRIVAL_MODES:
            raise ValueError(
                f"arrival_cost must be 'ekf' or 'constant', got {arrival_cost!r}"
            )
        self._arrival_cost = arrival_cost

        if on_solver_failure not in self._FAILURE_MODES:
            raise ValueError(
                "on_solver_failure must be 'fallback', 'raise' or 'accept', got "
                f"{on_solver_failure!r}"
            )
        self._on_failure = on_solver_failure

        if not cs.has_nlpsol(solver):
            raise ValueError(
                f"unknown NLP solver {solver!r}; casadi.has_nlpsol({solver!r}) "
                "is False"
            )

        # The covariances the companion already validated; reused verbatim so
        # the two estimators are weighted by the very same numbers.
        self._Q_aug = self._aekf._Q_aug.copy()
        self._R = self._aekf._ekf._R.copy()
        self._P0 = self._aekf.P
        self._W_Q = _inv_chol_blocks(self._q_blocks(self._Q_aug))
        self._W_R = _inv_chol(self._R, "R")
        self._W_P0 = _inv_chol(self._P0, "P0")

        self._lbz, self._ubz = self._assemble_state_bounds(
            x_lb, x_ub, du_bias_lb, du_bias_ub, dy_bias_lb, dy_bias_ub
        )

        self._build_nlp(solver, solver_opts, expand)

        # Rolling data. `_u_buf` and `_snap_buf` are pushed in predict(),
        # `_y_buf` in update(), so all three carry `window_fill` entries by the
        # time the solve runs.
        self._u_buf: deque = deque(maxlen=self._N + 1)
        self._y_buf: deque = deque(maxlen=self._N + 1)
        self._snap_buf: deque = deque(maxlen=self._N + 1)

        self._z0 = self._aekf.z_est
        self._z_filtered = self._z0.copy()
        self._z_est = self._z0.copy()
        self._Z: Optional[npt.NDArray[np.floating]] = None
        self._lam_x: Optional[npt.NDArray[np.floating]] = None
        self._predicted = False
        self._last_stats: dict = {}
        self._last_cost: Optional[float] = None
        self._last_solve_time_s: Optional[float] = None
        self._n_solver_failures = 0
        self._n_arrival_repairs = 0

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _q_blocks(
        self, Q_aug: npt.NDArray[np.floating]
    ) -> list[tuple[npt.NDArray[np.floating], str]]:
        """Split the augmented process covariance back into its named blocks."""
        nx, n_du = self._nx, self._n_du
        return [
            (Q_aug[:nx, :nx], "Q_x"),
            (Q_aug[nx : nx + n_du, nx : nx + n_du], "Q_du"),
            (Q_aug[nx + n_du :, nx + n_du :], "Q_dy"),
        ]

    def _assemble_state_bounds(
        self,
        x_lb: Optional[npt.ArrayLike],
        x_ub: Optional[npt.ArrayLike],
        du_bias_lb: Optional[npt.ArrayLike],
        du_bias_ub: Optional[npt.ArrayLike],
        dy_bias_lb: Optional[npt.ArrayLike],
        dy_bias_ub: Optional[npt.ArrayLike],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Stack the per-slot augmented bounds, gathering the bias channels.

        The bias bounds arrive at full signal width, like the bias seeds, and
        are gathered down to the selected channels here.
        """
        lb_x = _bound_vector(x_lb, self._nx, -np.inf, "x_lb")
        ub_x = _bound_vector(x_ub, self._nx, np.inf, "x_ub")
        _check_bounds(lb_x, ub_x, "x_lb", "x_ub")

        lb_du = _bound_vector(du_bias_lb, self._nu, -np.inf, "du_bias_lb")
        ub_du = _bound_vector(du_bias_ub, self._nu, np.inf, "du_bias_ub")
        _check_bounds(lb_du, ub_du, "du_bias_lb", "du_bias_ub")

        lb_dy = _bound_vector(dy_bias_lb, self._ny, -np.inf, "dy_bias_lb")
        ub_dy = _bound_vector(dy_bias_ub, self._ny, np.inf, "dy_bias_ub")
        _check_bounds(lb_dy, ub_dy, "dy_bias_lb", "dy_bias_ub")

        du_sel = list(self._aekf.du_index)
        dy_sel = list(self._aekf.dy_index)
        lbz = np.vstack(
            [
                lb_x,
                lb_du[du_sel].reshape(self._n_du, 1),
                lb_dy[dy_sel].reshape(self._n_dy, 1),
            ]
        )
        ubz = np.vstack(
            [
                ub_x,
                ub_du[du_sel].reshape(self._n_du, 1),
                ub_dy[dy_sel].reshape(self._n_dy, 1),
            ]
        )
        return lbz, ubz

    def _build_nlp(
        self, solver: str, solver_opts: Optional[dict], expand: bool
    ) -> None:
        """Assemble the fixed-size NLP and its parameter layout.

        Everything that changes from cycle to cycle -- the measurements, the
        inputs, which slots are filled, where the arrival cost attaches and all
        three weight factors -- is an NLP *parameter*, so the problem is built
        and compiled exactly once. Keeping the weights out of the graph is what
        lets :meth:`retune` change the tuning without paying for a rebuild.

        Parameter vector layout (offsets are instance attributes)::

            U      (nu,    N)      inputs; column j drives slot j -> slot j+1
            Y      (ny,    N+1)    measurements, one per slot
            m_w    (N,)            1 on filled intervals, else 0
            m_v    (N+1,)          1 on filled slots, else 0
            a      (N+1,)          one-hot: which slot carries the arrival cost
            z_arr  (n_aug,)        arrival anchor
            W_arr  (n_aug, n_aug)  arrival weight factor
            W_Q    (n_aug, n_aug)  process weight factor
            W_R    (ny,    ny)     measurement weight factor
        """
        N, n_aug, nu, ny = self._N, self._n_aug, self._nu, self._ny

        sizes = [
            ("U", nu * N),
            ("Y", ny * (N + 1)),
            ("m_w", N),
            ("m_v", N + 1),
            ("a", N + 1),
            ("z_arr", n_aug),
            ("W_arr", n_aug * n_aug),
            ("W_Q", n_aug * n_aug),
            ("W_R", ny * ny),
        ]
        self._p_slice = {}
        off = 0
        for name, size in sizes:
            self._p_slice[name] = slice(off, off + size)
            off += size
        self._n_p = off

        Z_flat = cs.MX.sym("Z", n_aug * (N + 1))
        Z = cs.reshape(Z_flat, n_aug, N + 1)
        p = cs.MX.sym("p", self._n_p)

        def block(name, rows, cols):
            part = p[self._p_slice[name]]
            return cs.reshape(part, rows, cols) if cols != 1 else part

        U = block("U", nu, N)
        Y = block("Y", ny, N + 1)
        m_w = block("m_w", N, 1)
        m_v = block("m_v", N + 1, 1)
        a = block("a", N + 1, 1)
        z_arr = block("z_arr", n_aug, 1)
        W_arr = block("W_arr", n_aug, n_aug)
        W_Q = block("W_Q", n_aug, n_aug)
        W_R = block("W_R", ny, ny)

        # Arrival cost. `Z @ a` picks the arrival slot by one-hot rather than by
        # Python indexing, so the expression graph is the same whatever the
        # window fill is.
        J = cs.sumsqr(W_arr @ (Z @ a - z_arr))

        # The mask multiplies the residual *inside* sumsqr (it is 0/1, so
        # squaring is a no-op), which keeps the least-squares structure intact
        # and makes a masked term contribute exactly zero to the gradient.
        for j in range(N):
            w_j = Z[:, j + 1] - self._f_aug(Z[:, j], U[:, j])
            J = J + cs.sumsqr(m_w[j] * (W_Q @ w_j))
        for j in range(N + 1):
            v_j = Y[:, j] - self._h_aug(Z[:, j])
            J = J + cs.sumsqr(m_v[j] * (W_R @ v_j))

        nlp = {"x": Z_flat, "p": p, "f": J}
        opts = {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 100,
                "tol": 1e-8,
                "acceptable_tol": 1e-6,
                # IPOPT relaxes bounds by ~1e-8 by default, which would let a
                # state bounded below by zero come back very slightly negative
                # and blow up the next sqrt() in the plant model.
                "bound_relax_factor": 0.0,
            },
        }
        for key, value in (solver_opts or {}).items():
            if isinstance(value, dict) and isinstance(opts.get(key), dict):
                opts[key] = {**opts[key], **value}
            else:
                opts[key] = value

        try:
            self._solver = cs.nlpsol("mhe", solver, nlp, {**opts, "expand": expand})
        except Exception:
            if not expand:
                raise
            # A model that cannot be expanded to SX (an interpolant, or a
            # network graph) is a legitimate input; it just runs as MX.
            warnings.warn(
                "the moving-horizon problem could not be expanded to SX "
                "(usually because the dynamics contain a node with no SX "
                "equivalent); falling back to MX, which solves the same "
                "problem more slowly. Pass expand=False to silence this.",
                UserWarning,
                stacklevel=3,
            )
            self._solver = cs.nlpsol("mhe", solver, nlp, {**opts, "expand": False})

        self._p_buf = np.zeros(self._n_p, dtype=np.float64)

    # ------------------------------------------------------------------
    # per-cycle assembly
    # ------------------------------------------------------------------

    def _arrival_anchor(self, n_fill: int) -> npt.NDArray[np.floating]:
        """The point the arrival cost pulls the window's oldest slot towards."""
        if self._arrival_cost == "constant" and self._Z is not None:
            # The slot that is now the arrival slot held the same time index one
            # cycle ago at position a + 1, because the window shifts left by
            # one. This re-uses a smoothed value and so is a heuristic, which is
            # the price of a frozen arrival weight.
            a_prev = self._N - n_fill + 2
            if 0 <= a_prev <= self._N:
                return self._Z[:, a_prev : a_prev + 1].copy()
        z_snap, _ = self._snap_buf[0]
        return z_snap

    def _arrival_weight(self) -> npt.NDArray[np.floating]:
        """Weight factor for the arrival cost, repaired if the prior went sour.

        With ``arrival_cost='ekf'`` this factors the companion filter's
        covariance every cycle. That covariance comes out of the non-Joseph
        update ``(I - K H) P``, which is known to drift indefinite under strong
        nonlinearity or a very confident ``R`` -- so on a long run the
        factorization *will* eventually fail. Symmetrize, then add growing
        jitter, then fall back to the construction-time prior.
        """
        if self._arrival_cost == "constant":
            return self._W_P0

        _, P = self._snap_buf[0]
        P = 0.5 * (P + P.T)
        n = P.shape[0]
        base = max(float(np.trace(P)) / n, 1.0) if n else 1.0
        for k in range(4):
            candidate = P if k == 0 else P + (10.0**k) * np.finfo(float).eps * base * np.eye(n)
            try:
                with warnings.catch_warnings():
                    # A marginally conditioned prior is normal mid-run; warning
                    # once per cycle would drown the log.
                    warnings.simplefilter("ignore", UserWarning)
                    W = _inv_chol(candidate, "the arrival covariance")
            except ValueError:
                continue
            if k:
                self._n_arrival_repairs += 1
            return W

        self._n_arrival_repairs += 1
        warnings.warn(
            "the arrival covariance lost positive definiteness and could not be "
            "repaired with jitter; falling back to the initial P0 for this "
            "step. Persistent repairs mean the companion filter is diverging -- "
            "raise Q_x, loosen R, or reset().",
            UserWarning,
            stacklevel=3,
        )
        return self._W_P0

    def _assemble_parameters(self, n_fill: int) -> npt.NDArray[np.floating]:
        """Write this cycle's data into the preallocated parameter buffer.

        Index algebra, with cycle counter ``t`` and ``a = N - n_fill + 1`` the
        oldest filled slot. Slot ``j`` holds time ``t - (N - j)``, so the newest
        estimate is always slot ``N``::

            slot j filled       iff j >= a           -> Y[:, j] = y_buf[-1-(N-j)]
            interval j filled   iff j >= a           -> U[:, j] = u_buf[-(N-j)]
            arrival slot        a                    -> one-hot
            arrival anchor      snapshot from cycle t - n_fill + 1, which is the
                                leftmost entry of the snapshot ring

        Interval ``j`` carries the input that drove the plant from slot ``j`` to
        slot ``j + 1``, i.e. the ``u`` of the *later* slot's cycle -- which is
        the pairing ``predict(u); update(y)`` establishes.
        """
        N, nu, ny, n_aug = self._N, self._nu, self._ny, self._n_aug
        buf = self._p_buf
        buf.fill(0.0)
        a = N - n_fill + 1

        U = np.zeros((nu, N), dtype=np.float64)
        Y = np.zeros((ny, N + 1), dtype=np.float64)
        m_w = np.zeros(N, dtype=np.float64)
        m_v = np.zeros(N + 1, dtype=np.float64)

        for j in range(a, N + 1):
            Y[:, j] = self._y_buf[-1 - (N - j)].ravel()
            m_v[j] = 1.0
        for j in range(a, N):
            U[:, j] = self._u_buf[-(N - j)].ravel()
            m_w[j] = 1.0

        one_hot = np.zeros(N + 1, dtype=np.float64)
        one_hot[a] = 1.0

        z_arr = self._arrival_anchor(n_fill)
        W_arr = self._arrival_weight()
        self._z_arr = z_arr

        buf[self._p_slice["U"]] = U.ravel(order="F")
        buf[self._p_slice["Y"]] = Y.ravel(order="F")
        buf[self._p_slice["m_w"]] = m_w
        buf[self._p_slice["m_v"]] = m_v
        buf[self._p_slice["a"]] = one_hot
        buf[self._p_slice["z_arr"]] = z_arr.ravel()
        buf[self._p_slice["W_arr"]] = W_arr.ravel(order="F")
        buf[self._p_slice["W_Q"]] = self._W_Q.ravel(order="F")
        buf[self._p_slice["W_R"]] = self._W_R.ravel(order="F")
        assert n_aug == z_arr.shape[0]  # layout invariant
        return buf

    def _pin_value(self) -> npt.NDArray[np.floating]:
        """Value the unfilled slots are pinned to.

        Not zero. A slot whose cost terms are masked out is a free variable in a
        singular problem, so it has to be pinned -- but pinning it at the origin
        would evaluate ``f`` and ``h`` there anyway (the terms are multiplied by
        the mask *after* being computed), and a model containing ``sqrt`` or a
        division returns NaN, which then poisons the whole objective. The
        arrival anchor is a point the model is known to be defined at; clipping
        it into the bounds also keeps ``lbx <= ubx`` when a seed violates them.
        """
        return np.clip(self._z_arr, self._lbz, self._ubz)

    def _assemble_bounds(
        self, n_fill: int
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Per-slot box bounds, with the unfilled prefix pinned."""
        N, n_aug = self._N, self._n_aug
        lbx = np.tile(self._lbz, (N + 1, 1)).ravel()
        ubx = np.tile(self._ubz, (N + 1, 1)).ravel()
        pin = self._pin_value().ravel()
        for j in range(N - n_fill + 1):
            lbx[j * n_aug : (j + 1) * n_aug] = pin
            ubx[j * n_aug : (j + 1) * n_aug] = pin
        return lbx, ubx

    def _warm_start(self) -> npt.NDArray[np.floating]:
        """Previous solution shifted one step, or the anchor broadcast."""
        N, n_aug = self._N, self._n_aug
        if self._Z is None:
            guess = np.tile(self._z_arr, (1, N + 1))
        else:
            guess = np.empty((n_aug, N + 1), dtype=np.float64)
            guess[:, :N] = self._Z[:, 1:]
            guess[:, N] = np.asarray(
                self._f_aug(self._Z[:, N], self._u_buf[-1])
            ).ravel()
        guess = np.clip(guess, self._lbz, self._ubz)
        return guess.ravel(order="F")

    def _shift_forward(self, z_new: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Roll the stored window one slot left and append ``z_new``."""
        N, n_aug = self._N, self._n_aug
        if self._Z is None:
            return np.tile(z_new.reshape(n_aug, 1), (1, N + 1))
        shifted = np.empty((n_aug, N + 1), dtype=np.float64)
        shifted[:, :N] = self._Z[:, 1:]
        shifted[:, N] = z_new.ravel()
        return shifted

    # ------------------------------------------------------------------
    # recursion
    # ------------------------------------------------------------------

    def predict(self, u: npt.ArrayLike) -> None:
        """Time update: register the input and roll the estimate forward.

        Parameters
        ----------
        u : array_like, shape (nu,) or (nu, 1)
            Control input applied at the current time step, *without* the
            estimated bias -- the model adds its own estimate before calling
            ``f``.

        Notes
        -----
        No optimization happens here; the window is solved in :meth:`update`.
        What this does is buffer ``u``, advance the companion filter, take the
        arrival snapshot for this cycle (after the companion's prediction and
        before its correction, so it is conditioned on strictly older data), and
        expose a one-step-ahead estimate so that an innovation formed between
        ``predict`` and ``update`` is a genuine moving-horizon residual.
        """
        u_vec = _ensure_column_vector(u, self._nu, "u")
        self._u_buf.append(u_vec)

        self._aekf.predict(u_vec)
        # Post-predict, pre-update: this is the correctly-conditioned prior for
        # this cycle's time index, read back n_fill cycles later when this slot
        # becomes the window's oldest.
        self._snap_buf.append((self._aekf.z_est, self._aekf.P))

        self._z_est = np.asarray(
            self._f_aug(self._z_filtered, u_vec), dtype=np.float64
        ).reshape(self._n_aug, 1)
        self._predicted = True

    def update(self, y: npt.ArrayLike, u: Optional[npt.ArrayLike] = None) -> None:
        """Measurement update: append the measurement and solve the window.

        Parameters
        ----------
        y : array_like, shape (ny,) or (ny, 1)
            Measured output.
        u : array_like, optional
            Unused; kept for API symmetry with the filters. The input for this
            step was already supplied to :meth:`predict`.

        Raises
        ------
        RuntimeError
            If :meth:`predict` was not called first, or if the solve failed and
            ``on_solver_failure='raise'``.
        """
        y_vec = _ensure_column_vector(y, self._ny, "y")
        if not self._predicted:
            raise RuntimeError(
                "update() called without a preceding predict(); the "
                "moving-horizon window needs the input that drove the plant "
                "into this measurement, so predict(u) must be called first on "
                "every cycle. (This is stricter than the Kalman filters, which "
                "tolerate a measurement-only update.)"
            )

        self._y_buf.append(y_vec)
        self._aekf.update(y_vec)

        n_fill = len(self._y_buf)
        p = self._assemble_parameters(n_fill)
        lbx, ubx = self._assemble_bounds(n_fill)
        x0 = self._warm_start()

        call = {"x0": x0, "p": p, "lbx": lbx, "ubx": ubx}
        if self._lam_x is not None:
            call["lam_x0"] = self._lam_x

        t_start = time.perf_counter()
        sol = self._solver(**call)
        self._last_solve_time_s = time.perf_counter() - t_start
        # stats() describes only the most recent call, so snapshot it now.
        stats = dict(self._solver.stats())
        self._last_stats = stats

        Z = np.asarray(sol["x"], dtype=np.float64).reshape(
            self._n_aug, self._N + 1, order="F"
        )
        finite = bool(np.all(np.isfinite(Z)))
        converged = bool(stats.get("success", False)) or stats.get(
            "return_status"
        ) == "Solved_To_Acceptable_Level"
        # A restoration failure can hand back a garbage iterate, so finiteness
        # is checked even when the caller asked to accept whatever came out.
        accepted = finite and (converged or self._on_failure == "accept")

        if accepted:
            self._Z = Z
            self._lam_x = np.asarray(sol["lam_x"], dtype=np.float64)
            self._last_cost = float(sol["f"])
            self._z_filtered = Z[:, -1:].copy()
        else:
            self._n_solver_failures += 1
            if self._on_failure == "raise":
                self._predicted = False
                raise RuntimeError(
                    "the moving-horizon NLP failed: the solver returned "
                    f"{stats.get('return_status', 'an unknown status')!r} after "
                    f"{stats.get('iter_count', '?')} iterations at window fill "
                    f"{n_fill}/{self._N + 1}. Loosen the tolerances, raise "
                    "max_iter, rescale the states, or pass "
                    "on_solver_failure='fallback' to keep running on the "
                    "companion filter's estimate."
                )
            # The companion filter has already consumed this measurement, so it
            # is a current estimate rather than a stale one.
            self._z_filtered = self._aekf.z_est
            self._Z = self._shift_forward(self._z_filtered)
            self._lam_x = None
            self._last_cost = None

        self._z_est = self._z_filtered.copy()
        self._predicted = False

    def reset(
        self,
        x0: Optional[npt.ArrayLike] = None,
        du_bias0: Optional[npt.ArrayLike] = None,
        dy_bias0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Reset the estimate and discard the window.

        Parameters
        ----------
        x0 : array_like, optional
            New plant state estimate. If None, resets to zeros.
        du_bias0 : array_like, shape (nu,) or (nu, 1), optional
            New input bias at full input width. If None, resets to zeros.
        dy_bias0 : array_like, shape (ny,) or (ny, 1), optional
            New output bias at full output width. If None, resets to zeros.
        P0 : array_like, shape (n_aug, n_aug), optional
            New augmented covariance. If None, resets to identity.

        Notes
        -----
        The buffered window is **not** carried across: a reset says the past is
        no longer to be trusted, and the next ``horizon`` cycles re-fill it. Use
        :meth:`retune` instead to change the tuning while keeping the window.
        """
        self._aekf.reset(x0=x0, du_bias0=du_bias0, dy_bias0=dy_bias0, P0=P0)
        self._P0 = self._aekf.P
        self._W_P0 = _inv_chol(self._P0, "P0")

        self._u_buf.clear()
        self._y_buf.clear()
        self._snap_buf.clear()

        self._z0 = self._aekf.z_est
        self._z_filtered = self._z0.copy()
        self._z_est = self._z0.copy()
        self._Z = None
        self._lam_x = None
        self._predicted = False
        self._last_stats = {}
        self._last_cost = None
        self._last_solve_time_s = None

    def retune(
        self,
        Q_x: Optional[npt.ArrayLike] = None,
        Q_du: Optional[npt.ArrayLike] = None,
        Q_dy: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Change the tuning in place, keeping the estimate and the window.

        The weights live in the NLP's parameter vector rather than in its
        expression graph, so this only rewrites numbers -- the problem is not
        rebuilt and the buffered window survives. That matters online: rebuilding
        would stall the control cycle on a fresh compile *and* throw away
        ``horizon`` cycles of data.

        Parameters
        ----------
        Q_x, Q_du, Q_dy, R, P0 : array_like, optional
            New covariances; omitted ones keep their current value. Shapes are
            as in the constructor.

        Notes
        -----
        The companion filter has no setters, so it is rebuilt and its estimate
        and covariance restored -- the same manoeuvre a hot retune performs on
        the augmented filter. Its detectability check is not re-run, because the
        channel selection cannot change here and it already passed.

        The arrival snapshots buffered from earlier cycles were computed under
        the *old* tuning; they age out over ``horizon`` cycles.
        """
        cur_Q_x, cur_Q_du, cur_Q_dy = (b for b, _ in self._q_blocks(self._Q_aug))

        new_Q_x = cur_Q_x if Q_x is None else _ensure_array(Q_x, "Q_x")
        new_Q_du = cur_Q_du if Q_du is None else _ensure_array(Q_du, "Q_du")
        new_Q_dy = cur_Q_dy if Q_dy is None else _ensure_array(Q_dy, "Q_dy")
        new_R = self._R if R is None else _ensure_array(R, "R")
        new_P0 = self._P0 if P0 is None else _ensure_array(P0, "P0")

        old = self._aekf
        rebuilt = AugmentedExtendedKalmanFilter(
            f=self._f_orig,
            h=self._h_orig,
            du_index=list(old.du_index),
            dy_index=list(old.dy_index),
            Q_x=new_Q_x,
            Q_du=new_Q_du,
            Q_dy=new_Q_dy,
            R=new_R,
            u_lin=self._u_lin_arg,
            check_detectability=False,
        )
        rebuilt.reset(
            x0=old.x_est,
            du_bias0=old.du_bias_est,
            dy_bias0=old.dy_bias_est,
            P0=old.P,
        )

        # Factor before committing, so a rejected covariance leaves the
        # estimator exactly as it was.
        W_Q = _inv_chol_blocks(self._q_blocks(rebuilt._Q_aug))
        W_R = _inv_chol(rebuilt._ekf._R, "R")
        W_P0 = _inv_chol(new_P0, "P0")

        self._aekf = rebuilt
        self._f_aug, self._h_aug = rebuilt._f_aug, rebuilt._h_aug
        self._Q_aug = rebuilt._Q_aug.copy()
        self._R = rebuilt._ekf._R.copy()
        self._P0 = np.asarray(new_P0, dtype=np.float64)
        self._W_Q, self._W_R, self._W_P0 = W_Q, W_R, W_P0

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def nx(self) -> int:
        """Plant state dimension."""
        return self._nx

    @property
    def nu(self) -> int:
        """Input dimension."""
        return self._nu

    @property
    def ny(self) -> int:
        """Output dimension."""
        return self._ny

    @property
    def n_du(self) -> int:
        """Number of estimated input-bias states."""
        return self._n_du

    @property
    def n_dy(self) -> int:
        """Number of estimated output-bias states."""
        return self._n_dy

    @property
    def n_aug(self) -> int:
        """Total augmented state dimension (nx + n_du + n_dy)."""
        return self._n_aug

    @property
    def du_index(self) -> tuple[int, ...]:
        """Input channels carrying a bias."""
        return self._aekf.du_index

    @property
    def dy_index(self) -> tuple[int, ...]:
        """Output channels carrying a bias."""
        return self._aekf.dy_index

    @property
    def horizon(self) -> int:
        """Number of past intervals in the window."""
        return self._N

    @property
    def arrival_cost(self) -> str:
        """How the window's left edge is anchored (``'ekf'`` or ``'constant'``)."""
        return self._arrival_cost

    @property
    def x_est(self) -> npt.NDArray[np.floating]:
        """Plant state estimate, shape (nx, 1)."""
        return self._z_est[: self._nx].copy()

    @property
    def du_bias_est(self) -> npt.NDArray[np.floating]:
        """Input bias estimate at full input width, shape (nu, 1).

        Channels outside :attr:`du_index` are zero.
        """
        return self._S_du @ self._z_est[self._nx : self._nx + self._n_du]

    @property
    def dy_bias_est(self) -> npt.NDArray[np.floating]:
        """Output bias estimate at full output width, shape (ny, 1).

        Channels outside :attr:`dy_index` are zero.
        """
        return self._S_dy @ self._z_est[self._nx + self._n_du :]

    @property
    def z_est(self) -> npt.NDArray[np.floating]:
        """Full augmented estimate, shape (n_aug, 1), with compact biases."""
        return self._z_est.copy()

    @property
    def P(self) -> npt.NDArray[np.floating]:
        """Augmented error covariance, shape (n_aug, n_aug).

        This is the **companion filter's** covariance, not the estimator's. A
        moving-horizon estimator produces no covariance of its own: where a
        bound is active the true uncertainty is smaller than this, and under
        model mismatch it can be larger. Treat it as the tuning-consistent
        second moment it is, which is what makes it the right thing to feed a
        Kalman-gain diagnostic, and not as a confidence interval on
        :attr:`x_est`.
        """
        return self._aekf.P

    @property
    def window_fill(self) -> int:
        """Number of filled slots, from 0 up to ``horizon + 1``."""
        return len(self._y_buf)

    @property
    def is_window_full(self) -> bool:
        """Whether the window has seen ``horizon + 1`` measurements."""
        return len(self._y_buf) == self._N + 1

    @property
    def z_traj(self) -> npt.NDArray[np.floating]:
        """Smoothed augmented window, shape (n_aug, window_fill), newest last.

        Only the filled slots are returned, so this is empty before the first
        solve and grows to ``horizon + 1`` columns. The last column is
        :attr:`z_est` (after a successful :meth:`update`).
        """
        if self._Z is None:
            return np.zeros((self._n_aug, 0), dtype=np.float64)
        return self._Z[:, self._N - self.window_fill + 1 :].copy()

    @property
    def x_traj(self) -> npt.NDArray[np.floating]:
        """Smoothed plant-state window, shape (nx, window_fill), newest last."""
        return self.z_traj[: self._nx]

    @property
    def last_success(self) -> Optional[bool]:
        """Whether the last solve converged; None before the first solve."""
        if not self._last_stats:
            return None
        return bool(self._last_stats.get("success", False))

    @property
    def last_status(self) -> Optional[str]:
        """Solver return status of the last solve."""
        status = self._last_stats.get("return_status")
        return None if status is None else str(status)

    @property
    def last_cost(self) -> Optional[float]:
        """Objective of the last accepted solve; None if it failed.

        A better divergence alarm than the innovation: model mismatch is spread
        across the window rather than concentrated in one residual, and active
        bounds clip the residual, so this drifts up while the innovation still
        looks healthy.
        """
        return self._last_cost

    @property
    def last_solve_time_s(self) -> Optional[float]:
        """Wall-clock time of the last solve, in seconds."""
        return self._last_solve_time_s

    @property
    def n_solver_failures(self) -> int:
        """Cumulative count of solves that were not accepted."""
        return self._n_solver_failures

    @property
    def n_arrival_repairs(self) -> int:
        """Cumulative count of arrival covariances that needed repair.

        Persistent growth means the companion filter's covariance is losing
        positive definiteness, i.e. it is diverging.
        """
        return self._n_arrival_repairs

    @property
    def solver_stats(self) -> dict:
        """Copy of the solver's statistics for the last solve."""
        return dict(self._last_stats)

    def get_mpc_biases(self) -> dict[str, npt.NDArray[np.floating]]:
        """Get estimated biases in MPC-compatible format.

        Returns a dictionary ready to be passed to ``Mpc.solve_mpc()`` via the
        ``dynamic_pars`` argument, with the same keys and full-width shapes as
        :meth:`AugmentedExtendedKalmanFilter.get_mpc_biases`.

        Returns
        -------
        dict
            ``"du_bias"`` of shape (nu, 1) and ``"dy_bias"`` of shape (ny, 1).
        """
        return {
            "du_bias": self.du_bias_est,
            "dy_bias": self.dy_bias_est,
        }

    def detectability_report(self) -> dict[str, Union[bool, int, tuple[int, ...]]]:
        """Check whether the bias augmentation is detectable at the current estimate.

        Delegates to the companion
        :meth:`AugmentedExtendedKalmanFilter.detectability_report`; the budget
        is a property of the augmented model, and a longer window does not buy
        observability.

        Returns
        -------
        dict
            See :meth:`AugmentedExtendedKalmanFilter.detectability_report`.
        """
        return self._aekf.detectability_report()
