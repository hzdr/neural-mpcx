# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Helmholtz-Zentrum Dresden-Rossendorf e.V. (HZDR)
# Authors: 
# - Ênio Lopes Júnior
# - Sebastian Felix Reinecke
#
# Contains modifications of code from casadi-nlp
# (https://github.com/FilippoAiraldi/casadi-nlp),
# Copyright (c) 2024 Filippo Airaldi, licensed under the MIT License.

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from numpy.polynomial import polynomial as P
from scipy.interpolate import pade as scipy_pade
from scipy.linalg import solve_discrete_are
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
]:
    """Assemble continuous MIMO state-space from transfer function terms.

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
    tuple of (A, B, C, D)
        Continuous-time state-space matrices.
    """
    A_blocks: List[npt.NDArray] = []
    B_blocks: List[npt.NDArray] = []
    C_blocks: List[npt.NDArray] = []
    D_accum = np.zeros((ny, nu), dtype=float)

    # Process each (i, j) entry
    for (i, j), terms in G.items():
        for term in terms:
            # Convert term to transfer function polynomials
            num, den = _term_to_tf(term, pade_order)

            # Convert SISO TF to state-space
            A_ij, B_ij, C_ij, D_ij = tf2ss(num, den)

            n_states = A_ij.shape[0]
            if n_states == 0:
                # Direct feedthrough only (no dynamics)
                D_accum[i, j] += float(np.squeeze(D_ij))
                continue

            A_blocks.append(A_ij)

            # B_global: input j receives B_ij, others zero
            B_global_block = np.zeros((n_states, nu), dtype=float)
            B_global_block[:, j : j + 1] = B_ij
            B_blocks.append(B_global_block)

            # C_global: output i receives C_ij, others zero
            C_global_block = np.zeros((ny, n_states), dtype=float)
            C_global_block[i : i + 1, :] = C_ij
            C_blocks.append(C_global_block)

            # Accumulate D
            D_accum[i, j] += float(np.squeeze(D_ij))

    if not A_blocks:
        # No dynamics, only feedthrough
        return (
            np.zeros((0, 0), dtype=float),
            np.zeros((0, nu), dtype=float),
            np.zeros((ny, 0), dtype=float),
            D_accum,
        )

    # Assemble block-diagonal A
    sizes = [A.shape[0] for A in A_blocks]
    total_states = sum(sizes)

    A_cont = np.zeros((total_states, total_states), dtype=float)
    offset = 0
    for k, A_k in enumerate(A_blocks):
        n = sizes[k]
        A_cont[offset : offset + n, offset : offset + n] = A_k
        offset += n

    # Stack B and C blocks
    B_cont = np.vstack(B_blocks)
    C_cont = np.hstack(C_blocks)

    return A_cont, B_cont, C_cont, D_accum


def mimo_tf2ss(
    G: Dict[Tuple[int, int], List[TransferFunctionTerm]],
    ny: int,
    nu: int,
    Ts: float,
    pade_order: int = 2,
    method: Literal["zoh", "foh", "impulse", "tustin", "bilinear"] = "zoh",
    store_continuous: bool = True,
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

    Returns
    -------
    DiscreteStateSpace
        Dataclass containing Ad, Bd, Cd, Dd matrices and metadata.

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
    # Assemble continuous MIMO state-space
    A_cont, B_cont, C_cont, D_cont = _assemble_mimo_ss(G, ny, nu, pade_order)

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
    P = solve_discrete_are(Atilde, B, Qtilde, R)
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
