import numpy as np
from functools import partial
from typing import Dict, Tuple, Optional


class LinearKalmanFilter:
    """
    Simple and autonomous linear Kalman Filter.
    """
    
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 initial_state: np.ndarray,
                 initial_P: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 A: Optional[np.ndarray] = None,
                 H: Optional[np.ndarray] = None,
                 bias_list: Optional[list] = None):
        """
        Parameters
        ----------
        state_dim : int
            State dimension
        obs_dim : int
            Observation/measurement dimension
        initial_state : np.ndarray
            Initial state (mean)
        initial_P : np.ndarray
            Initial state covariance
        Q : np.ndarray or list
            Process noise covariance
        R : np.ndarray or list
            Measurement noise covariance
        A : np.ndarray, optional
            State transition matrix (default: identity)
        H : np.ndarray, optional
            Observation/measurement matrix (default: [1, 0, 0, ...])
        bias_list : list, optional
            List of biases to add to state
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # State and covariance
        self.x = initial_state.copy().astype(np.float64)
        self.P = initial_P.copy().astype(np.float64)
        
        # Transition and observation matrices
        self.A = A if A is not None else np.eye(state_dim, dtype=np.float64)
        self.H = H if H is not None else self._default_H()
        
        # Noise covariances
        self.Q = self._ensure_covariance(Q, state_dim)
        self.R = self._ensure_covariance(R, obs_dim)
        
        # Optional biases
        self.bias_list = bias_list or [0.0] * state_dim
        
        # History for logging
        self.innovation_history = []
        self.y_pred_history = []
    
    def _default_H(self) -> np.ndarray:
        """
        Default observation matrix: measures first state only
        """
        H = np.zeros((self.obs_dim, self.state_dim), dtype=np.float64)
        for i in range(min(self.obs_dim, self.state_dim)):
            H[i, i] = 1.0
        return H
    
    def _ensure_covariance(self, cov, dim: int) -> np.ndarray:
        """
        Convert list or scalar to diagonal covariance matrix
        """
        if isinstance(cov, list):
            return np.diag(cov).astype(np.float64)
        elif isinstance(cov, (int, float)):
            return cov * np.eye(dim, dtype=np.float64)
        else:
            return np.array(cov, dtype=np.float64)
    
    def predict(self, A: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of the Kalman Filter.
        
        Parameters
        ----------
        A : np.ndarray, optional
            State transition matrix (if None, uses self.A)
        
        Returns
        -------
        x_pred : np.ndarray
            Predicted state
        P_pred : np.ndarray
            Predicted covariance
        """
        if A is not None:
            self.A = A.copy().astype(np.float64)
        
        # Apply biases
        x_biased = self.x + np.array(self.bias_list, dtype=np.float64)
        
        # Prediction
        x_pred = self.A @ x_biased
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step of the Kalman Filter.
        
        Handles singular matrices robustly using pseudo-inverse.
        
        Parameters
        ----------
        z : np.ndarray
            Observation/measurement
        
        Returns
        -------
        x_updated : np.ndarray
            Updated state estimate
        P_updated : np.ndarray
            Updated covariance
        """
        z = np.array(z, dtype=np.float64).flatten()
        
        # Innovation (residual)
        y_pred = self.H @ self.x
        innovation = z - y_pred
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain with robust matrix inversion
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if matrix is singular
            S_inv = np.linalg.pinv(S)
        
        K = self.P @ self.H.T @ S_inv
        
        # Update
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        
        # Logging
        self.innovation_history.append(innovation)
        self.y_pred_history.append(y_pred)
        
        return self.x.copy(), self.P.copy()
    
    def step(self,
             z: np.ndarray,
             A: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete step: prediction + update.
        
        Parameters
        ----------
        z : np.ndarray
            Observation/measurement
        A : np.ndarray, optional
            State transition matrix
        
        Returns
        -------
        x : np.ndarray
            Estimated state
        P : np.ndarray
            State covariance
        innovation : np.ndarray
            Innovation/residual
        """
        # Prediction
        x_pred, P_pred = self.predict(A)
        self.x = x_pred
        self.P = P_pred
        
        # Update
        z = np.array(z, dtype=np.float64).flatten()
        y_pred = self.H @ self.x
        innovation = z - y_pred
        
        S = self.H @ self.P @ self.H.T + self.R
        
        # Use robust matrix inversion
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if matrix is singular
            S_inv = np.linalg.pinv(S)
        
        K = self.P @ self.H.T @ S_inv
        
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        
        self.innovation_history.append(innovation)
        self.y_pred_history.append(y_pred)
        
        return self.x.copy(), self.P.copy(), innovation
    
    def set_A(self, A: np.ndarray):
        """Set the state transition matrix"""
        self.A = np.array(A, dtype=np.float64)
    
    def set_H(self, H: np.ndarray):
        """Set the observation/measurement matrix"""
        self.H = np.array(H, dtype=np.float64)
    
    def set_state(self, x: np.ndarray, P: Optional[np.ndarray] = None):
        """Set state and optionally covariance"""
        self.x = np.array(x, dtype=np.float64).flatten()
        if P is not None:
            self.P = np.array(P, dtype=np.float64)
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        return self.x.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance"""
        return self.P.copy()
    
    def get_measurement_prediction(self) -> np.ndarray:
        """Predict next measurement"""
        return self.H @ self.x
    
    def reset_to_initial(self, initial_state: np.ndarray, initial_P: np.ndarray):
        """Reset filter to initial state and covariance"""
        self.x = initial_state.copy().astype(np.float64)
        self.P = initial_P.copy().astype(np.float64)
        self.innovation_history.clear()
        self.y_pred_history.clear()


def kf_step_simple(kf: LinearKalmanFilter,
                   measurements: np.ndarray,
                   time: int,
                   A_matrix: Optional[np.ndarray] = None,
                   bias_list: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function for a single Kalman Filter step.
    Replaces 'test_KF_step_new'.
    
    Parameters
    ----------
    kf : LinearKalmanFilter
        Kalman Filter instance
    measurements : np.ndarray
        All measurements (full trajectory)
    time : int
        Current timestep
    A_matrix : np.ndarray, optional
        State transition matrix (overrides self.A)
    bias_list : list, optional
        Biases to apply
    
    Returns
    -------
    x_est : np.ndarray
        Estimated state
    P_est : np.ndarray
        State covariance
    """
    if bias_list is not None:
        kf.bias_list = bias_list
    
    z_t = measurements[time]
    x_est, P_est, _ = kf.step(z_t, A=A_matrix)
    
    return x_est, P_est

def create_dataset_observability_controlled( 
    n_steps=300,
    n_dim=3,
    x0=None,
    Q_coeff=1e-4,
    R_coeff=1e-4,
    seed=None,
    H=None,   # matrice de mesure (m x n_dim) ou vecteur (n_dim,)
    A=None    # matrice dynamique fournie (ne pas recalculer)
):
    rng = np.random.default_rng(seed)

    # === Matrice de mesure H ===
    if H is None:
        H = np.zeros((1, n_dim))
        H[0, 0] = 1.0
    else:
        H = np.array(H)
        # si utilisateur passe un vecteur (n_dim,), convertir en (1, n_dim)
        if H.ndim == 1:
            if H.size == n_dim:
                H = H.reshape(1, n_dim)
            else:
                raise ValueError("Si H est 1D, sa taille doit être n_dim")

    m = H.shape[0]   # nombre de mesures par pas de temps

    # === Matrice dynamique A ===
    if A is None:
        raise ValueError("A doit être fourni pour éviter le recalcul à chaque appel")
    A = np.array(A)
    A_mean = A.copy()
    A_std = 0.00002
    A_sampled = rng.normal(loc=A_mean, scale=A_std * np.abs(A_mean))
    A = A_sampled

    # === Initialisation état ===
    if x0 is None:
        x0 = rng.uniform(0.5, 1.5, size=n_dim)

    true_state_hist = np.zeros((n_steps + 1, n_dim))
    true_state_hist[0] = x0

    # Simulation dynamique
    for k in range(n_steps):
        x_next = A @ true_state_hist[k]
        w = rng.normal(0, np.sqrt(Q_coeff), size=n_dim)
        true_state_hist[k+1] = x_next + w

    # --- Mesures bruitées (vectorisé) ---
    # true_state_hist: (T, n_dim), H: (m, n_dim) -> true_state_hist @ H.T -> (T, m)
    signal = true_state_hist @ H.T

    # bruit : gérer R_coeff scalaire ou matrice
    if np.isscalar(R_coeff):
        noise = rng.normal(0, np.sqrt(R_coeff), size=(n_steps+1, m))
        R_inv = (1.0 / R_coeff) * np.eye(m)
    else:
        R = np.array(R_coeff)
        if R.shape != (m, m):
            raise ValueError("Si R_coeff n'est pas scalaire, il doit avoir la forme (m, m)")
        # pour bruit multivarié, on échantillonne un bruit gaussien ayant covariance R
        # on diagonalise ou utilise Cholesky
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            # si pas définie positive (ou condition), fallback sur sqrt des diag
            L = np.linalg.cholesky(R + 1e-12 * np.eye(m))
        z = rng.normal(size=(n_steps+1, m))
        noise = z @ L.T
        R_inv = np.linalg.inv(R)

    observed_measure = signal + noise   # shape (n_steps+1, m)

    # --- Gramienne d’observabilité discrète ---
    Gram = np.zeros((n_dim, n_dim))
    for k in range(n_steps+1):
        A_k = np.linalg.matrix_power(A, k)   # A^k
        # terme = (A^k).T @ H.T @ R_inv @ H @ (A^k)
        Gram += A_k.T @ H.T @ R_inv @ H @ A_k

    # conditionnement (peut être inf si mal conditionnée)
    cond_gram = np.linalg.cond(Gram)

    return {
        "true_state_hist": true_state_hist,         # (n_steps+1, n_dim)
        "observed_measure": observed_measure,       # (n_steps+1, m)
        "x0": x0,
        "cond_gram": cond_gram,
        "H": H,
        "A": A
    }

