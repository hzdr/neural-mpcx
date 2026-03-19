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
AugmentedKalmanFilter
    Extended Kalman filter for joint state and bias estimation.

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

from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = ["KalmanFilter", "AugmentedKalmanFilter"]


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


class AugmentedKalmanFilter:
    """Augmented Kalman filter for joint state and bias estimation.

    Extends the standard Kalman filter to simultaneously estimate:

    - Plant state x (nx dimensions)
    - Input bias du_bias (nu dimensions)
    - Output bias dy_bias (ny dimensions)

    Biases are modeled as random walks (integrated white noise).

    Augmented state vector::

        z = [x; du_bias; dy_bias]

    Augmented system::

        z[k+1] = A_aug @ z[k] + B_aug @ u[k]
        y[k]   = C_aug @ z[k] + D_aug @ u[k]

    where::

        A_aug = [Ad   0    0  ]      B_aug = [Bd]
                [0    I_nu 0  ]              [0 ]
                [0    0    I_ny]             [0 ]

        C_aug = [Cd  Dd  I_ny]       D_aug = Dd

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
    Q_x : array_like, shape (nx, nx), optional
        Process noise covariance for plant states. Default is 0.1 * I.
    Q_du : array_like, shape (nu, nu), optional
        Process noise covariance for input bias (random walk intensity).
        Default is 0.01 * I.
    Q_dy : array_like, shape (ny, ny), optional
        Process noise covariance for output bias (random walk intensity).
        Default is 0.01 * I.
    R : array_like, shape (ny, ny), optional
        Measurement noise covariance. Default is 1.0 * I.
    x0 : array_like, shape (nx,) or (nx, 1), optional
        Initial plant state estimate. Default is zeros.
    du_bias0 : array_like, shape (nu,) or (nu, 1), optional
        Initial input bias estimate. Default is zeros.
    dy_bias0 : array_like, shape (ny,) or (ny, 1), optional
        Initial output bias estimate. Default is zeros.
    P0 : array_like, shape (n_aug, n_aug), optional
        Initial augmented error covariance. Default is identity.

    Attributes
    ----------
    nx : int
        Plant state dimension.
    nu : int
        Input dimension.
    ny : int
        Output dimension.
    n_aug : int
        Total augmented state dimension (nx + nu + ny).

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
    ...     Q_x=np.eye(2) * 0.1,
    ...     Q_du=np.eye(1) * 0.01,
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
    the MPC solver via the ``dynamic_pars`` argument.

    References
    ----------
    .. [1] Maciejowski, J. M. (2002). "Predictive Control with Constraints",
           Chapter 4. Prentice Hall.
    """

    def __init__(
        self,
        Ad: npt.ArrayLike,
        Bd: npt.ArrayLike,
        Cd: npt.ArrayLike,
        Dd: Optional[npt.ArrayLike] = None,
        Q_x: Optional[npt.ArrayLike] = None,
        Q_du: Optional[npt.ArrayLike] = None,
        Q_dy: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        x0: Optional[npt.ArrayLike] = None,
        du_bias0: Optional[npt.ArrayLike] = None,
        dy_bias0: Optional[npt.ArrayLike] = None,
        P0: Optional[npt.ArrayLike] = None,
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
        self._n_aug = self._nx + self._nu + self._ny

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

        # Construct augmented dynamics matrix A_aug
        self._A_aug = np.block(
            [
                [
                    Ad_arr,
                    np.zeros((self._nx, self._nu)),
                    np.zeros((self._nx, self._ny)),
                ],
                [
                    np.zeros((self._nu, self._nx)),
                    np.eye(self._nu),
                    np.zeros((self._nu, self._ny)),
                ],
                [
                    np.zeros((self._ny, self._nx)),
                    np.zeros((self._ny, self._nu)),
                    np.eye(self._ny),
                ],
            ]
        )

        # Construct augmented input matrix B_aug
        self._B_aug = np.vstack(
            [
                Bd_arr,
                np.zeros((self._nu, self._nu)),
                np.zeros((self._ny, self._nu)),
            ]
        )

        # Construct augmented output matrix C_aug
        self._C_aug = np.hstack([Cd_arr, Dd_arr, np.eye(self._ny)])
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

        if Q_du is None:
            Q_du_arr = np.eye(self._nu, dtype=np.float64) * 0.01
        else:
            Q_du_arr = _ensure_array(Q_du, "Q_du")
            if Q_du_arr.shape != (self._nu, self._nu):
                raise ValueError(
                    f"Q_du must have shape ({self._nu}, {self._nu}), "
                    f"got {Q_du_arr.shape}"
                )

        if Q_dy is None:
            Q_dy_arr = np.eye(self._ny, dtype=np.float64) * 0.01
        else:
            Q_dy_arr = _ensure_array(Q_dy, "Q_dy")
            if Q_dy_arr.shape != (self._ny, self._ny):
                raise ValueError(
                    f"Q_dy must have shape ({self._ny}, {self._ny}), "
                    f"got {Q_dy_arr.shape}"
                )

        # Construct augmented process noise covariance Q_aug
        self._Q_aug = np.block(
            [
                [
                    Q_x_arr,
                    np.zeros((self._nx, self._nu)),
                    np.zeros((self._nx, self._ny)),
                ],
                [
                    np.zeros((self._nu, self._nx)),
                    Q_du_arr,
                    np.zeros((self._nu, self._ny)),
                ],
                [
                    np.zeros((self._ny, self._nx)),
                    np.zeros((self._ny, self._nu)),
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
        if x0 is None:
            x0_arr = np.zeros((self._nx, 1), dtype=np.float64)
        else:
            x0_arr = _ensure_column_vector(x0, self._nx, "x0")

        if du_bias0 is None:
            du_bias0_arr = np.zeros((self._nu, 1), dtype=np.float64)
        else:
            du_bias0_arr = _ensure_column_vector(du_bias0, self._nu, "du_bias0")

        if dy_bias0 is None:
            dy_bias0_arr = np.zeros((self._ny, 1), dtype=np.float64)
        else:
            dy_bias0_arr = _ensure_column_vector(dy_bias0, self._ny, "dy_bias0")

        self._z_est = np.vstack([x0_arr, du_bias0_arr, dy_bias0_arr])

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
    def n_aug(self) -> int:
        """Total augmented state dimension (nx + nu + ny)."""
        return self._n_aug

    @property
    def x_est(self) -> npt.NDArray[np.floating]:
        """Plant state estimate, shape (nx, 1)."""
        return self._z_est[: self._nx].copy()

    @property
    def du_bias_est(self) -> npt.NDArray[np.floating]:
        """Input bias estimate, shape (nu, 1)."""
        return self._z_est[self._nx : self._nx + self._nu].copy()

    @property
    def dy_bias_est(self) -> npt.NDArray[np.floating]:
        """Output bias estimate, shape (ny, 1)."""
        return self._z_est[self._nx + self._nu :].copy()

    @property
    def z_est(self) -> npt.NDArray[np.floating]:
        """Full augmented state estimate, shape (n_aug, 1)."""
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
        remain constant during prediction with noise added through Q_aug.
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
        du_bias0 : array_like, optional
            New input bias estimate. If None, resets to zeros.
        dy_bias0 : array_like, optional
            New output bias estimate. If None, resets to zeros.
        P0 : array_like, optional
            New error covariance. If None, resets to identity.
        """
        if x0 is None:
            x0_arr = np.zeros((self._nx, 1), dtype=np.float64)
        else:
            x0_arr = _ensure_column_vector(x0, self._nx, "x0")

        if du_bias0 is None:
            du_bias0_arr = np.zeros((self._nu, 1), dtype=np.float64)
        else:
            du_bias0_arr = _ensure_column_vector(du_bias0, self._nu, "du_bias0")

        if dy_bias0 is None:
            dy_bias0_arr = np.zeros((self._ny, 1), dtype=np.float64)
        else:
            dy_bias0_arr = _ensure_column_vector(dy_bias0, self._ny, "dy_bias0")

        self._z_est = np.vstack([x0_arr, du_bias0_arr, dy_bias0_arr])

        if P0 is None:
            self._P = np.eye(self._n_aug, dtype=np.float64)
        else:
            self._P = _ensure_array(P0, "P0")
            if self._P.shape != (self._n_aug, self._n_aug):
                raise ValueError(
                    f"P0 must have shape ({self._n_aug}, {self._n_aug}), "
                    f"got {self._P.shape}"
                )
