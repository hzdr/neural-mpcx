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

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Literal, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from numpy.polynomial import polynomial as P
from scipy.interpolate import pade as scipy_pade
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_are
from scipy.signal import cont2discrete, tf2ss

T = TypeVar("T")


# =============================================================================
# Data Classes for Transfer Function Representation
# =============================================================================


@dataclass
class TransferFunctionTerm:
    """A single term in a SISO transfer function G_ij(s).

    Represents a term of the form::

        K * exp(-delay * s) / [denominator factors]

    where the denominator is the product of:
    - First-order factors: (tau*s + 1) for each tau in time_constants
    - Second-order factors: (s^2 + 2*zeta*wn*s + wn^2) for each (zeta, wn)
    - Integrator: s (if has_integrator is True)

    Parameters
    ----------
    gain : float
        Static gain K of the term.
    delay : float, optional
        Time delay (dead time) in seconds. Default is 0.0.
    time_constants : list of float, optional
        List of time constants [tau1, tau2, ...] for first-order factors.
        Each tau gives a factor (tau*s + 1) in the denominator.
    second_order_factors : list of tuple[float, float], optional
        List of (zeta, wn) pairs for second-order factors.
        Each pair gives (s^2 + 2*zeta*wn*s + wn^2) in the denominator.
    has_integrator : bool, optional
        If True, adds an integrator (1/s) to the denominator. Default False.

    Examples
    --------
    >>> # G(s) = -0.58 * exp(-41s) / (83s + 1)
    >>> TransferFunctionTerm(gain=-0.58, delay=41.0, time_constants=[83.0])

    >>> # G(s) = 0.001 * exp(-30s) / [s * (150s + 1)]  (with integrator)
    >>> TransferFunctionTerm(gain=0.001, delay=30.0, time_constants=[150.0],
    ...                      has_integrator=True)

    >>> # G(s) = 5.0 / (s^2 + 0.4s + 4)  (underdamped second-order, zeta=0.1, wn=2)
    >>> TransferFunctionTerm(gain=5.0, second_order_factors=[(0.1, 2.0)])
    """

    gain: float
    delay: float = 0.0
    time_constants: List[float] = field(default_factory=list)
    second_order_factors: List[Tuple[float, float]] = field(default_factory=list)
    has_integrator: bool = False


@dataclass(frozen=True)
class DiscreteStateSpace:
    """Immutable container for discrete-time state-space matrices.

    Represents the system::

        x[k+1] = Ad @ x[k] + Bd @ u[k]
        y[k]   = Cd @ x[k] + Dd @ u[k]

    Attributes
    ----------
    Ad : np.ndarray
        Discrete state transition matrix (nx, nx).
    Bd : np.ndarray
        Discrete input matrix (nx, nu).
    Cd : np.ndarray
        Output matrix (ny, nx).
    Dd : np.ndarray
        Feedthrough matrix (ny, nu).
    Ts : float
        Sample time in seconds.
    nx : int
        Number of states.
    nu : int
        Number of inputs.
    ny : int
        Number of outputs.
    A : np.ndarray, optional
        Continuous state matrix (for reference).
    B : np.ndarray, optional
        Continuous input matrix (for reference).
    C : np.ndarray, optional
        Continuous output matrix (for reference).
    D : np.ndarray, optional
        Continuous feedthrough matrix (for reference).
    pade_order : int
        Padé order used for delay approximation.
    """

    Ad: npt.NDArray[np.floating]
    Bd: npt.NDArray[np.floating]
    Cd: npt.NDArray[np.floating]
    Dd: npt.NDArray[np.floating]
    Ts: float
    nx: int
    nu: int
    ny: int
    A: Optional[npt.NDArray[np.floating]] = None
    B: Optional[npt.NDArray[np.floating]] = None
    C: Optional[npt.NDArray[np.floating]] = None
    D: Optional[npt.NDArray[np.floating]] = None
    pade_order: int = 2


# =============================================================================
# Polynomial Utility Functions (Internal)
# =============================================================================


def _poly_mul_desc(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Multiply two polynomials with coefficients in descending order.

    Parameters
    ----------
    p : array
        First polynomial coefficients (descending order).
    q : array
        Second polynomial coefficients (descending order).

    Returns
    -------
    array
        Product polynomial coefficients (descending order).
    """
    # numpy.polynomial uses ascending order, so reverse input/output
    pr = p[::-1]
    qr = q[::-1]
    rr = P.polymul(pr, qr)
    result = rr[::-1]
    # Trim leading zeros
    return np.trim_zeros(result, "f") if np.any(result) else np.array([0.0])


def _poly_scale_var_desc(coeffs: npt.NDArray, a: float) -> npt.NDArray:
    """Scale polynomial variable: p(s) -> p(a*s).

    Given p(s) = sum_{i=0}^{n} c_i * s^{n-i}, returns coefficients of p(a*s).

    Parameters
    ----------
    coeffs : array
        Polynomial coefficients in descending order.
    a : float
        Scale factor for the variable.

    Returns
    -------
    array
        Scaled polynomial coefficients (descending order).
    """
    n = len(coeffs) - 1
    scaled = [coeffs[i] * (a ** (n - i)) for i in range(len(coeffs))]
    return np.array(scaled, dtype=float)


def _pade_delay_poly(
    delay: float, order: int
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute Padé approximation polynomials for exp(-delay*s).

    Uses scipy.interpolate.pade to approximate e^x and substitutes x = -delay*s.

    Parameters
    ----------
    delay : float
        Time delay in seconds.
    order : int
        Order of Padé approximation (same order for num and den).

    Returns
    -------
    tuple of (numerator, denominator)
        Polynomial coefficients in descending order of s.
    """
    if delay <= 0 or order <= 0:
        return np.array([1.0]), np.array([1.0])

    # Series coefficients of e^x: a_k = 1/k!
    an = [1.0 / math.factorial(k) for k in range(2 * order + 1)]
    p_poly, q_poly = scipy_pade(an, order, order)

    # e^{-delay*s} ≈ p(-delay*s) / q(-delay*s)
    num = _poly_scale_var_desc(p_poly.coeffs, -delay)
    den = _poly_scale_var_desc(q_poly.coeffs, -delay)
    return num, den


def _first_order_poly(tau: float) -> npt.NDArray[np.floating]:
    """Return polynomial (tau*s + 1) in descending order."""
    return np.array([tau, 1.0], dtype=float)


def _second_order_poly(zeta: float, wn: float) -> npt.NDArray[np.floating]:
    """Return polynomial (s^2 + 2*zeta*wn*s + wn^2) in descending order."""
    return np.array([1.0, 2.0 * zeta * wn, wn * wn], dtype=float)


def _integrator_poly() -> npt.NDArray[np.floating]:
    """Return polynomial s in descending order."""
    return np.array([1.0, 0.0], dtype=float)


def _const_poly(k: float) -> npt.NDArray[np.floating]:
    """Return constant polynomial k in descending order."""
    return np.array([float(k)], dtype=float)


# =============================================================================
# Transfer Function to State-Space Conversion
# =============================================================================


def _term_to_tf(
    term: TransferFunctionTerm, pade_order: int
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Convert a TransferFunctionTerm to (numerator, denominator) polynomials.

    Parameters
    ----------
    term : TransferFunctionTerm
        The transfer function term to convert.
    pade_order : int
        Order of Padé approximation for delays.

    Returns
    -------
    tuple of (num, den)
        Numerator and denominator polynomial coefficients (descending order).
    """
    # Build denominator from factors
    den = np.array([1.0])

    # First-order factors: (tau*s + 1)
    for tau in term.time_constants:
        den = _poly_mul_desc(den, _first_order_poly(tau))

    # Second-order factors: (s^2 + 2*zeta*wn*s + wn^2)
    for zeta, wn in term.second_order_factors:
        den = _poly_mul_desc(den, _second_order_poly(zeta, wn))

    # Integrator: s
    if term.has_integrator:
        den = _poly_mul_desc(den, _integrator_poly())

    # Delay approximation via Padé
    if term.delay > 0.0 and pade_order > 0:
        num_delay, den_delay = _pade_delay_poly(term.delay, pade_order)
    else:
        num_delay, den_delay = np.array([1.0]), np.array([1.0])

    # Total numerator: K * num_delay
    num = _poly_mul_desc(_const_poly(term.gain), num_delay)

    # Total denominator: den * den_delay
    den = _poly_mul_desc(den, den_delay)

    return num, den


def _assemble_mimo_ss(
    G: Dict[Tuple[int, int], List[TransferFunctionTerm]],
    ny: int,
    nu: int,
    pade_order: int,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    int,
]:
    """Assemble continuous MIMO state-space from transfer function terms.

    Terms with ``has_integrator=True`` share a single integrator state per
    output row: the stable part of each such term feeds an auxiliary signal
    w_i, and one appended state integrates it (x_int_i' = w_i, y_i += x_int_i).
    Realizing each channel with its own integrator would create multiple
    eigenvalues at the origin observable only through their sum — structurally
    undetectable difference modes that break any Riccati-based design (LQR
    terminal costs, steady-state Kalman filters).

    Parameters
    ----------
    G : dict
        Transfer function matrix as {(i, j): [terms...]}.
    ny : int
        Number of outputs.
    nu : int
        Number of inputs.
    pade_order : int
        Padé order for delay approximation.

    Returns
    -------
    tuple of (A, B, C, D, n_int)
        Continuous-time state-space matrices; the last ``n_int`` states are
        the shared integrator states appended after the stable blocks.
    """
    # Output rows containing integrator terms each get one shared integrator
    int_rows = sorted(
        {i for (i, _), terms in G.items() for t in terms if t.has_integrator}
    )
    aux_of_row = {i: k for k, i in enumerate(int_rows)}
    n_int = len(int_rows)

    A_blocks: List[npt.NDArray] = []
    B_blocks: List[npt.NDArray] = []
    C_blocks: List[npt.NDArray] = []  # direct contributions to the outputs
    W_blocks: List[npt.NDArray] = []  # contributions to the shared integrators
    D_accum = np.zeros((ny, nu), dtype=float)
    D_w = np.zeros((n_int, nu), dtype=float)

    # Process each (i, j) entry
    for (i, j), terms in G.items():
        for term in terms:
            # Convert the stable part to transfer function polynomials;
            # the integrator (if any) is factored out and shared per row
            num, den = _term_to_tf(replace(term, has_integrator=False), pade_order)

            if len(den) == 1:
                # Static gain: scipy.signal.tf2ss would pad this to a spurious
                # 1-state realization with a pole at the origin
                k_gain = float(num[0]) / float(den[0])
                if term.has_integrator:
                    D_w[aux_of_row[i], j] += k_gain
                else:
                    D_accum[i, j] += k_gain
                continue

            # Convert SISO TF to state-space
            A_ij, B_ij, C_ij, D_ij = tf2ss(num, den)
            n_states = A_ij.shape[0]

            A_blocks.append(A_ij)

            # B_global: input j receives B_ij, others zero
            B_global_block = np.zeros((n_states, nu), dtype=float)
            B_global_block[:, j : j + 1] = B_ij
            B_blocks.append(B_global_block)

            # Route the block output either directly to output i or into the
            # row's shared integrator
            C_global_block = np.zeros((ny, n_states), dtype=float)
            W_global_block = np.zeros((n_int, n_states), dtype=float)
            if term.has_integrator:
                W_global_block[aux_of_row[i] : aux_of_row[i] + 1, :] = C_ij
                D_w[aux_of_row[i], j] += float(np.squeeze(D_ij))
            else:
                C_global_block[i : i + 1, :] = C_ij
                D_accum[i, j] += float(np.squeeze(D_ij))
            C_blocks.append(C_global_block)
            W_blocks.append(W_global_block)

    n_stable = sum(A_k.shape[0] for A_k in A_blocks)
    n_total = n_stable + n_int

    if n_total == 0:
        # No dynamics, only feedthrough
        return (
            np.zeros((0, 0), dtype=float),
            np.zeros((0, nu), dtype=float),
            np.zeros((ny, 0), dtype=float),
            D_accum,
            0,
        )

    # Assemble block-diagonal A for the stable blocks
    A_cont = np.zeros((n_total, n_total), dtype=float)
    offset = 0
    for A_k in A_blocks:
        n = A_k.shape[0]
        A_cont[offset : offset + n, offset : offset + n] = A_k
        offset += n

    B_cont = np.zeros((n_total, nu), dtype=float)
    C_cont = np.zeros((ny, n_total), dtype=float)
    if n_stable > 0:
        B_cont[:n_stable, :] = np.vstack(B_blocks)
        C_cont[:, :n_stable] = np.hstack(C_blocks)

    # Append the shared integrator states:
    # x_int_i' = w_i = W x_stable + D_w u,  y_i += x_int_i
    if n_int > 0:
        if n_stable > 0:
            A_cont[n_stable:, :n_stable] = np.hstack(W_blocks)
        B_cont[n_stable:, :] = D_w
        for i, k in aux_of_row.items():
            C_cont[i, n_stable + k] = 1.0

    return A_cont, B_cont, C_cont, D_accum, n_int


def _balanced_realization(
    A: npt.NDArray[np.floating],
    B: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Apply balanced realization to a continuous-time state-space (A, B, C).

    Computes a similarity transformation T such that in the new coordinates
    the controllability and observability Gramians are equal and diagonal,
    with the Hankel singular values on the diagonal.  This gives the
    best-conditioned numerically equivalent state-space representation.

    Algorithm (Gramian method):

    1. Solve controllability Gramian:  A Wc + Wc A^T + B B^T = 0
    2. Solve observability Gramian:   A^T Wo + Wo A + C^T C = 0
    3. Cholesky:  Wc = Lc Lc^T,  Wo = Lo Lo^T
    4. SVD:  Lo^T Lc = U S V^T
    5. T = Lc V S^{-1/2},  T_inv = S^{-1/2} U^T Lo^T
    6. A_bal = T_inv A T,  B_bal = T_inv B,  C_bal = C T

    Parameters
    ----------
    A : array (n, n)
        Continuous-time state matrix.  Must be Hurwitz (all eigenvalues
        strictly in the open left-half plane).
    B : array (n, nu)
        Input matrix.
    C : array (ny, n)
        Output matrix.

    Returns
    -------
    tuple of (A_bal, B_bal, C_bal)
        Balanced state-space matrices.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the Gramians are not positive definite (system not
        controllable/observable, or A not Hurwitz).
    """
    Wc = solve_continuous_lyapunov(A, -(B @ B.T))
    Wo = solve_continuous_lyapunov(A.T, -(C.T @ C))

    from scipy.linalg import cholesky

    Lc = cholesky(Wc, lower=True)
    Lo = cholesky(Wo, lower=True)

    U, S, Vt = np.linalg.svd(Lo.T @ Lc)
    s_sqrt_inv = np.diag(S**-0.5)

    T = Lc @ Vt.T @ s_sqrt_inv
    T_inv = s_sqrt_inv @ U.T @ Lo.T

    return T_inv @ A @ T, T_inv @ B, C @ T


def mimo_tf2ss(
    G: Dict[Tuple[int, int], List[TransferFunctionTerm]],
    ny: int,
    nu: int,
    Ts: float,
    pade_order: int = 2,
    method: Literal["zoh", "foh", "impulse", "tustin", "bilinear"] = "zoh",
    store_continuous: bool = True,
    balanced: bool = True,
) -> DiscreteStateSpace:
    """Convert a MIMO transfer function matrix to discrete state-space form.

    Parameters
    ----------
    G : dict of {(i, j): list of TransferFunctionTerm}
        Transfer function matrix specification. Keys are (row, column) indices
        (0-based), values are lists of TransferFunctionTerm objects that sum
        to form G_ij(s). Missing entries are assumed to be zero.
    ny : int
        Number of outputs (rows in G).
    nu : int
        Number of inputs (columns in G).
    Ts : float
        Sample time for discretization in seconds.
    pade_order : int, optional
        Order of Padé approximation for time delays. Default is 2.
    method : str, optional
        Discretization method passed to scipy.signal.cont2discrete.
        Options: 'zoh' (default), 'foh', 'impulse', 'tustin', 'bilinear'.
    store_continuous : bool, optional
        If True, also stores continuous-time matrices in result. Default True.
    balanced : bool, optional
        If True (default), applies a balanced realization to the stable part
        of the assembled continuous-time state-space before discretization.
        The similarity transformation equalises the controllability and
        observability Gramians, giving the best-conditioned numerically
        equivalent representation.  Shared integrator states (see Notes) are
        left untouched.  A ``UserWarning`` is emitted and the unbalanced
        (controllable canonical) form is kept when the stable part contains
        other marginally stable or unstable modes.

    Returns
    -------
    DiscreteStateSpace
        Dataclass containing Ad, Bd, Cd, Dd matrices and metadata.

    Notes
    -----
    Terms with ``has_integrator=True`` share a single integrator state per
    output row (the stable parts are summed and integrated once).  This is
    input-output equivalent to per-channel integrators but avoids multiple
    origin poles that are observable only through their sum — such jointly
    unobservable modes make Riccati-based designs (LQR terminal costs,
    steady-state Kalman filters) infeasible.

    Examples
    --------
    >>> from neuralmpcx.util.control import mimo_tf2ss, TransferFunctionTerm as TF
    >>>
    >>> # Define a 2x2 MIMO system
    >>> G = {
    ...     (0, 0): [TF(gain=-0.58, delay=41.0, time_constants=[83.0])],
    ...     (0, 1): [TF(gain=0.97, delay=40.0, time_constants=[125.0, 195.0]),
    ...              TF(gain=-0.97*1.08, delay=272.0, time_constants=[125.0, 195.0])],
    ...     (1, 0): [TF(gain=0.62, time_constants=[123.0])],
    ...     (1, 1): [TF(gain=-1.75, time_constants=[118.0])],
    ... }
    >>> ss = mimo_tf2ss(G, ny=2, nu=2, Ts=30.0, pade_order=2)
    >>> print(f"State dimension: {ss.nx}")
    """
    import warnings

    # Assemble continuous MIMO state-space; the last n_int states are the
    # shared integrators appended after the stable blocks
    A_cont, B_cont, C_cont, D_cont, n_int = _assemble_mimo_ss(G, ny, nu, pade_order)

    # Apply balanced realization to the stable subsystem (the integrator
    # states are excluded: Gramians only exist for Hurwitz dynamics, and the
    # integrator states are already perfectly conditioned)
    ns = A_cont.shape[0] - n_int
    if balanced and ns > 0:
        eigs = np.linalg.eigvals(A_cont[:ns, :ns])
        if np.all(eigs.real < 0):
            # The integrator couplings read the stable states too, so they
            # must enter the observability Gramian alongside the true outputs
            C_stack = np.vstack([C_cont[:, :ns], A_cont[ns:, :ns]])
            try:
                A_s, B_s, C_stack = _balanced_realization(
                    A_cont[:ns, :ns], B_cont[:ns, :], C_stack
                )
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Balanced realization failed (Gramians not positive definite). "
                    "Using controllable canonical form.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                A_cont[:ns, :ns] = A_s
                A_cont[ns:, :ns] = C_stack[ny:, :]
                B_cont[:ns, :] = B_s
                C_cont[:, :ns] = C_stack[:ny, :]
        else:
            warnings.warn(
                "System has marginally stable or unstable modes besides the "
                "shared integrators. Balanced realization requires a Hurwitz "
                "A matrix; using controllable canonical form.",
                UserWarning,
                stacklevel=2,
            )

    # Discretize
    if A_cont.size > 0:
        Ad, Bd, Cd, Dd, _ = cont2discrete(
            (A_cont, B_cont, C_cont, D_cont), Ts, method=method
        )
    else:
        # No dynamics case
        Ad = np.zeros((0, 0), dtype=float)
        Bd = np.zeros((0, nu), dtype=float)
        Cd = np.zeros((ny, 0), dtype=float)
        Dd = D_cont

    nx = Ad.shape[0]

    return DiscreteStateSpace(
        Ad=Ad,
        Bd=Bd,
        Cd=Cd,
        Dd=Dd,
        Ts=Ts,
        nx=nx,
        nu=nu,
        ny=ny,
        A=A_cont if store_continuous else None,
        B=B_cont if store_continuous else None,
        C=C_cont if store_continuous else None,
        D=D_cont if store_continuous else None,
        pade_order=pade_order,
    )


def dlqr(
    A: npt.NDArray[np.floating],
    B: npt.NDArray[np.floating],
    Q: npt.NDArray[np.floating],
    R: npt.NDArray[np.floating],
    M: Optional[npt.NDArray[np.floating]] = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Get the discrete-time LQR for the given system. Stage costs are
    ```
        x'Qx + 2*x'Mu + u'Ru
    ```
    with `M = 0`, if not provided.

    Parameters
    ----------
    A : array
        State matrix.
    B : array
        Control input matrix.
    Q : array
        State weighting matrix.
    R : array
        Control input weighting matrix.
    M : array, optional
        Mixed state-input weighting matrix, by default None.

    Returns
    -------
    tuple of two arrays
        Returns the optimal state feedback matrix `K` and the quadratic terminal
        cost-to-go matrix `P`.

    Note
    ----
    Inspired by
    https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py.
    """
    if M is not None:
        RinvMT = np.linalg.solve(R, M.T)
        Atilde = A - B.dot(RinvMT)
        Qtilde = Q - M.dot(RinvMT)
    else:
        Atilde = A
        Qtilde = Q
        M = np.zeros(B.shape)
    try:
        P = solve_discrete_are(Atilde, B, Qtilde, R)
    except (np.linalg.LinAlgError, ValueError) as err:
        raise np.linalg.LinAlgError(
            f"Discrete-time algebraic Riccati equation failed: {err} "
            "A stabilizing solution exists only if every unstable or "
            "unit-circle mode of A is stabilizable through B and detectable "
            "through Q. A common pitfall is several integrator channels "
            "measured only through their sum: the difference modes are "
            "jointly unobservable."
        ) from err
    K = np.linalg.solve(B.T.dot(P).dot(B) + R, B.T.dot(P).dot(A) + M.T)
    return K, P


def rk4(f: Callable[[T], T], x0: T, dt: float = 1, M: int = 1) -> T:
    """Computes the Runge-Kutta 4 integration of the given function `f` with initial
    state x0.

    Parameters
    ----------
    f : Callable[[casadi or array], casadi or array]
        A function that takes a state as input and returns the derivative of the state,
        i.e., continuous-time dynamics.
    x0 : casadi or array
        The initial state. Must be compatible as argument to `f`.
    dt : float, optional
        The discretization timestep, by default `1`.
    M : int, optional
        How many RK4 steps to take in one `dt` interval, by default `1`.

    Returns
    -------
    new state : casadi or array
        The new state after `dt` time, according to the discretization.

    Note
    ----
    Inspired by
    https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py.
    """
    dt /= M
    x = x0
    for _ in range(M):
        k1 = f(x)
        k2 = f(x + k1 * dt / 2)  # type: ignore
        k3 = f(x + k2 * dt / 2)  # type: ignore
        k4 = f(x + k3 * dt)  # type: ignore
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6  # type: ignore
    return x
