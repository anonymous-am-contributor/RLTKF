"""
Test Functions Module
Contains all baseline methods, test functions, and plotting utilities.
Includes correct Holt's exponential smoothing formulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from kf_utils import*


# ============================================================
# 1) GRAMIAN & OBSERVABILITY METRIC
# ============================================================

def compute_observability_gramian(A, H, horizon=10, R=None):
    """
    Compute the observability Gramian over a given horizon.
    
    Parameters
    ----------
    A : np.ndarray
        State transition matrix (n x n)
    H : np.ndarray
        Observation matrix (m x n)
    horizon : int
        Time horizon for Gramian computation
    R : float or np.ndarray, optional
        Measurement noise covariance (default: identity)
    
    Returns
    -------
    W : np.ndarray
        Observability Gramian matrix (n x n)
    """
    A = np.array(A, dtype=float)
    H = np.array(H, dtype=float)
    n = A.shape[0]
    m = H.shape[0]
    
    if R is None:
        R_inv = np.eye(m)
    else:
        R = np.array(R, dtype=float)
        if R.ndim == 0:
            if R <= 0:
                raise ValueError("R must be positive.")
            R_inv = np.eye(m) / float(R)
        elif R.shape == (m, m):
            R_inv = np.linalg.inv(R)
        else:
            raise ValueError("R must be scalar or (m,m) matrix.")
    
    W = np.zeros((n, n), dtype=float)
    Ak = np.eye(n)
    
    for k in range(horizon):
        term = Ak.T @ (H.T @ (R_inv @ (H @ Ak)))
        W += term
        Ak = Ak @ A
    
    return W


def normalize_gramian(W, method="lmax", eps=1e-30):
    """
    Normalize the Gramian matrix W.
    
    Parameters
    ----------
    W : np.ndarray
        Gramian matrix
    method : str
        Normalization method ("lmax" uses largest eigenvalue)
    eps : float
        Small value to avoid numerical issues
    
    Returns
    -------
    W_norm : np.ndarray
        Normalized Gramian
    """
    W = 0.5 * (W + W.T)
    
    if method == "lmax":
        try:
            eig = np.linalg.eigvalsh(W)
            lmax = np.max(eig)
        except np.linalg.LinAlgError:
            s = np.linalg.svd(W, compute_uv=False)
            lmax = np.max(s)
        
        s = lmax if lmax > eps else 1.0
        return W / s
    else:
        return W


def obs_metric_from_gramian(A, H, horizon=10, R=1e-3, metric="lambda_min_norm", eps=1e-52):
    """
    Compute observability metric from the Gramian.
    
    Parameters
    ----------
    A : np.ndarray
        State transition matrix
    H : np.ndarray
        Observation matrix
    horizon : int
        Time horizon for Gramian
    R : float or np.ndarray
        Measurement noise covariance
    metric : str
        Metric type ("lambda_min_norm" or "lambda_min")
    eps : float
        Small value to avoid division by zero
    
    Returns
    -------
    obs_value : float
        Observability metric value
    """
    W = compute_observability_gramian(A, H, horizon=horizon, R=R)
    Wn = normalize_gramian(W, method="lmax", eps=1e-30)
    
    try:
        eigvals = np.linalg.eigvalsh(Wn)
    except np.linalg.LinAlgError:
        s = np.linalg.svd(Wn, compute_uv=False)
        eigvals = np.maximum(s**2, 0.0)
    
    lam_min = np.min(eigvals)
    tr = np.trace(Wn)
    
    if metric == "lambda_min_norm":
        return lam_min / (tr + eps)
    else:
        return lam_min


# ============================================================
# 2) DATA GENERATION
# ============================================================

def generate_data(A, H, x0, n_steps, Q_gen, R_gen, rng):
    """
    Generate measurements and true states according to the model.
    
    x_{k+1} = A*x_k + w_k
    y_k = H*x_k + v_k
    
    Parameters
    ----------
    A : np.ndarray
        State transition matrix
    H : np.ndarray
        Observation matrix
    x0 : np.ndarray
        Initial state
    n_steps : int
        Number of timesteps
    Q_gen : float
        Process noise covariance (scalar, expanded to diagonal)
    R_gen : float
        Measurement noise covariance (scalar, expanded to diagonal)
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    states : np.ndarray
        True state trajectory (n_steps x n)
    measurements : np.ndarray
        Measurement trajectory (n_steps x m)
    """
    n = A.shape[0]
    m = H.shape[0]
    states = np.zeros((n_steps, n))
    measurements = np.zeros((n_steps, m))
    
    x = x0.copy()
    
    for t in range(n_steps):
        states[t] = x
        w = rng.multivariate_normal(np.zeros(n), Q_gen * np.eye(n))
        x = A @ x + w
        v = rng.multivariate_normal(np.zeros(m), R_gen * np.eye(m))
        measurements[t] = H @ states[t] + v
    
    return states, measurements


# ============================================================
# 3) KALMAN FILTER
# ============================================================

def kalman_filter(A, H, measurements, x0_estimate, Q, R, n_steps):
    """
    Standard Linear Kalman Filter.
    
    Parameters
    ----------
    A : np.ndarray
        State transition matrix
    H : np.ndarray
        Observation matrix
    measurements : np.ndarray
        Measurement sequence (n_steps x m)
    x0_estimate : np.ndarray
        Initial state estimate
    Q : float
        Process noise covariance
    R : float
        Measurement noise covariance
    n_steps : int
        Number of timesteps
    
    Returns
    -------
    estimates : np.ndarray
        State estimates (n_steps x n)
    """
    n = A.shape[0]
    m = H.shape[0]
    estimates = np.zeros((n_steps, n))
    
    P = np.eye(n) * 1.0
    x = x0_estimate.copy()
    
    for t in range(n_steps):
        # Prediction
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q * np.eye(n)
        
        # Update
        y = measurements[t] - H @ x_pred
        S = H @ P_pred @ H.T + R * np.eye(m)
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(n) - K @ H) @ P_pred
        
        estimates[t] = x
    
    return estimates


def kalman_filter_adaptive(A_func, H, measurements, x0_estimate, Q, R, n_steps, n_dim):
    """
    Adaptive Kalman Filter with time-varying state transition matrix.
    A_t is determined by function A_func at each timestep.
    
    Parameters
    ----------
    A_func : callable
        Function(t, measurement, x_pred) -> A_t (n x n matrix)
    H : np.ndarray
        Observation matrix
    measurements : np.ndarray
        Measurement sequence
    x0_estimate : np.ndarray
        Initial state estimate
    Q : float
        Process noise covariance
    R : float
        Measurement noise covariance
    n_steps : int
        Number of timesteps
    n_dim : int
        State dimension
    
    Returns
    -------
    estimates : np.ndarray
        State estimates
    """
    m = H.shape[0]
    estimates = np.zeros((n_steps, n_dim))
    P = np.eye(n_dim) * 1.0
    x = x0_estimate.copy()
    
    for t in range(n_steps):
        # Get A_t from function (can depend on t, measurements[t], x, etc.)
        A_t = A_func(t, measurements[t], x)
        
        # Prediction
        x_pred = A_t @ x
        P_pred = A_t @ P @ A_t.T + Q * np.eye(n_dim)
        
        # Update
        y = measurements[t] - H @ x_pred
        S = H @ P_pred @ H.T + R * np.eye(m)
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(n_dim) - K @ H) @ P_pred
        
        estimates[t] = x
    
    return estimates


# ============================================================
# 4) HOLT'S EXPONENTIAL SMOOTHING + KALMAN FILTER (CORRECT FORMULATION)
# ============================================================

def holts_exponential_smoothing_kf(H, measurements, x0_estimate, Q, R, n_steps, n_dim, alpha=0.1, beta=0.05):
    """
    Hybrid filter combining Holt's exponential smoothing with Kalman filter.
    
    Uses Holt's technique to dynamically estimate state transition matrix F_k
    and trend vector g_k at each timestep.
    
    Correct formulation from power systems literature:
    F_k = alpha*(1 + beta)*I
    a_k = alpha*x_k + (1 - alpha)*x_k^pred
    g_k = (1 + beta_k)*(1 - alpha_k)*x_k - beta_k*a_{k-1} + (1 - beta_k)*b_{k-1}
    b_k = beta_k*(a_k - a_{k-1}) + (1 - beta_k)*b_{k-1}
    
    Parameters
    ----------
    H : np.ndarray
        Observation matrix (m x n)
    measurements : np.ndarray
        Measurement sequence (n_steps x m)
    x0_estimate : np.ndarray
        Initial state estimate (n,)
    Q : float
        Process noise covariance
    R : float
        Measurement noise covariance
    n_steps : int
        Number of timesteps
    n_dim : int
        State dimension
    alpha : float
        Smoothing parameter for level [0, 1]
    beta : float
        Smoothing parameter for trend [0, 1]
    
    Returns
    -------
    estimates : np.ndarray
        State estimates (n_steps x n_dim)
    """
    H = np.array(H, dtype=float)
    n = n_dim
    m = H.shape[0]
    
    estimates = np.zeros((n_steps, n))
    P = np.eye(n) * 1.0
    x = x0_estimate.copy()
    
    # Initialize trend vectors
    a_prev = x.copy()
    b_prev = np.zeros(n)
    
    for t in range(n_steps):
        # Prediction with F_k and g_k dynamically estimated
        # F_k = alpha*(1 + beta)*I
        F_k = alpha * (1 + beta) * np.eye(n)
        
        # Predict state before correction
        x_pred_base = F_k @ x
        
        # g_k = (1 + beta)*(1 - alpha)*x_k - beta*a_{k-1} + (1 - beta)*b_{k-1}
        g_k = (1 + beta) * (1 - alpha) * x - beta * a_prev + (1 - beta) * b_prev
        
        # Full prediction
        x_pred = x_pred_base + g_k
        P_pred = F_k @ P @ F_k.T + Q * np.eye(n)
        
        # Kalman update step
        y = measurements[t] - H @ x_pred
        S = H @ P_pred @ H.T + R * np.eye(m)
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K = P_pred @ H.T @ S_inv
        x = x_pred + K @ y
        P = (np.eye(n) - K @ H) @ P_pred
        
        # Update trend parameters (Holt's smoothing)
        # a_k = alpha*x_k + (1 - alpha)*x_pred
        a_k = alpha * x + (1 - alpha) * x_pred
        
        # b_k = beta*(a_k - a_{k-1}) + (1 - beta)*b_{k-1}
        b_k = beta * (a_k - a_prev) + (1 - beta) * b_prev
        
        estimates[t] = x
        a_prev = a_k.copy()
        b_prev = b_k.copy()
    
    return estimates


# ============================================================
# 5) SYSTEM IDENTIFICATION METHODS
# ============================================================

def moesp_identify(measurements, n_dim, block_rows=None, verbose=False):
    """
    MOESP (Multivariable Output-Only Sub-space Identification) - output-only identification.
    Estimates A matrix from measurements only.
    
    Parameters
    ----------
    measurements : np.ndarray
        Measurement sequence (n_steps x m)
    n_dim : int
        State dimension to identify
    block_rows : int, optional
        Number of block rows for Hankel matrix
    verbose : bool
        Print debug information
    
    Returns
    -------
    A_est : np.ndarray
        Estimated state transition matrix (n_dim x n_dim)
    """
    Y = np.array(measurements, dtype=float)
    n_steps, m = Y.shape
    n = int(n_dim)
    
    if block_rows is None:
        s = max(2, min(20, n_steps // 4))
    else:
        s = int(block_rows)
    
    nc = max(1, n_steps - 2 * s + 1)
    
    # Build Hankel matrices
    Yp = np.zeros((m * s, nc))
    Yf = np.zeros((m * s, nc))
    
    for i in range(nc):
        Yp[:, i] = Y[i:i+s, :].T.flatten()
        Yf[:, i] = Y[i+s:i+2*s, :].T.flatten()
    
    Yp_mean = np.mean(Yp, axis=1, keepdims=True)
    Yp_c = Yp - Yp_mean
    
    # SVD on past Hankel matrix
    try:
        U, S, Vt = np.linalg.svd(Yp_c, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            U, S, Vt = np.linalg.svd(Yp_c + 1e-12 * np.random.randn(*Yp_c.shape), full_matrices=False)
        except Exception:
            if verbose:
                print("[moesp_identify] SVD failed, fallback to I")
            return np.eye(n)
    
    # Select principal components
    r = min(n, U.shape[1])
    U_r = U[:, :r]
    S_r = S[:r]
    
    # Build observability matrix
    Ob = U_r @ np.diag(np.sqrt(np.maximum(S_r, 1e-15)))
    
    # Estimate A: relation between future and past Hankel matrices
    rows_expected = m * (s - 1)
    if Ob.shape[0] < m + rows_expected:
        if verbose:
            print("[moesp_identify] Ob too small, fallback to I")
        return np.eye(n)
    
    Ts1 = Ob[:rows_expected, :]
    Ts2 = Ob[m: m + rows_expected, :]
    
    try:
        pinv_Ts1 = np.linalg.pinv(Ts1)
        A_est = pinv_Ts1 @ Ts2
    except Exception:
        try:
            U_t, S_t, Vt_t = np.linalg.svd(Ts1, full_matrices=False)
            S_inv = np.diag(1.0 / np.maximum(S_t, 1e-15))
            pinv_Ts1 = Vt_t.T @ S_inv @ U_t.T
            A_est = pinv_Ts1 @ Ts2
        except Exception:
            if verbose:
                print("[moesp_identify] pinv/SVD fallback failed, return I")
            return np.eye(n)
    
    if A_est.shape != (n, n):
        try:
            X, *_ = np.linalg.lstsq(Ts1.T, Ts2.T, rcond=None)
            A_est = X.T
        except Exception:
            if verbose:
                print("[moesp_identify] lstsq fallback failed, return I")
            return np.eye(n)
    
    # Stabilize A if necessary
    try:
        eigs = np.linalg.eigvals(A_est)
        rho = np.max(np.abs(eigs))
        if rho > 1.0:
            A_est = A_est / rho * 0.99
    except Exception:
        pass
    
    return A_est


def moesp_identify_A_with_true_H(measurements, true_H, n_dim, block_rows=None, verbose=False):
    """
    MOESP: Estimate A using known (true) H matrix.
    
    Parameters
    ----------
    measurements : np.ndarray
        Measurement sequence
    true_H : np.ndarray
        Known observation matrix
    n_dim : int
        State dimension
    block_rows : int, optional
        Number of block rows for Hankel matrix
    verbose : bool
        Print debug information
    
    Returns
    -------
    A_est : np.ndarray
        Estimated state transition matrix
    """
    Y = np.array(measurements, dtype=float)
    n_steps, m = Y.shape
    n = int(n_dim)
    true_H = np.array(true_H, dtype=float)

    if block_rows is None:
        s = max(2, min(20, n_steps // 4))
    else:
        s = int(block_rows)
    
    nc = max(1, n_steps - 2 * s + 1)

    # Build Hankel matrices
    Yp = np.zeros((m * s, nc))
    Yf = np.zeros((m * s, nc))
    
    for i in range(nc):
        Yp[:, i] = Y[i:i+s, :].T.flatten()
        Yf[:, i] = Y[i+s:i+2*s, :].T.flatten()

    Yp_mean = np.mean(Yp, axis=1, keepdims=True)
    Yp_c = Yp - Yp_mean

    try:
        U, S, Vt = np.linalg.svd(Yp_c, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            U, S, Vt = np.linalg.svd(Yp_c + 1e-12 * np.random.randn(*Yp_c.shape), full_matrices=False)
        except Exception:
            if verbose:
                print("[moesp_trueH] SVD failed, fallback to I")
            return np.eye(n)

    r = min(n, U.shape[1])
    U_r = U[:, :r]
    S_r = S[:r]
    Ob = U_r @ np.diag(np.sqrt(np.maximum(S_r, 1e-15)))

    rows_expected = m * (s - 1)
    if Ob.shape[0] < m + rows_expected:
        if verbose:
            print("[moesp_trueH] Ob too small, fallback to I")
        return np.eye(n)

    Gamma_full = np.vstack([true_H, Ob[m:m + rows_expected, :]])
    Gamma_past = Gamma_full[:-m, :]
    Gamma_fut  = Gamma_full[m:, :]

    try:
        A_est = np.linalg.lstsq(Gamma_past, Gamma_fut, rcond=None)[0].T
    except Exception:
        if verbose:
            print("[moesp_trueH] lstsq failed, fallback to I")
        return np.eye(n)

    # Stabilize
    try:
        eigs = np.linalg.eigvals(A_est)
        rho = np.max(np.abs(eigs))
        if rho > 1.0:
            A_est = A_est / rho * 0.99
    except Exception:
        pass

    return A_est

# ============================================================
# 6) ERROR METRICS
# ============================================================

def compute_rmse(estimates, true_states):
    """
    Compute Root Mean Square Error between estimates and true states.
    
    Parameters
    ----------
    estimates : np.ndarray
        State estimates (n_steps x n)
    true_states : np.ndarray
        True states (n_steps x n)
    
    Returns
    -------
    rmse : float
        RMSE value
    """
    errors = estimates - true_states
    rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    return rmse

# ============================================================
# 7) AGENT CONTROLLER
# ============================================================

class AgentController:
    """
    Encapsulates the prediction logic of an RL agent.
    """
    
    def __init__(self, agent_path, agent_type, n_dim):
        """
        Parameters
        ----------
        agent_path : str
            Path to the PPO model
        agent_type : str
            "measurement_only", "measurement_obs", "innovation_only", "innovation_obs"
        n_dim : int
            State dimension
        """
        self.agent = PPO.load(agent_path)
        self.agent_type = agent_type
        self.n_dim = n_dim
        self.a_min, self.a_max = 1e-18, 1.0
    
    def _build_observation(self, measurement, obs_metric=None, innovation=None):
        """Build observation according to agent type."""
        if self.agent_type == "measurement_only":
            return measurement
        elif self.agent_type == "measurement_obs":
            return np.concatenate([measurement, [obs_metric]])
        elif self.agent_type == "innovation_only":
            return innovation
        elif self.agent_type == "innovation_obs":
            return np.concatenate([innovation, [obs_metric]])
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
    
    def predict_A(self, measurement, obs_metric=None, innovation=None):
        """
        Predict state transition matrix A from observation.
        
        Parameters
        ----------
        measurement : np.ndarray
            Current measurement
        obs_metric : float, optional
            Observability metric value
        innovation : np.ndarray, optional
            Innovation (residual)
        
        Returns
        -------
        A_matrix : np.ndarray
            Predicted A matrix (n_dim x n_dim)
        """
        obs = self._build_observation(measurement, obs_metric, innovation)
        action, _ = self.agent.predict(obs, deterministic=True)
        action = np.clip(action, -1, 1)
        A_values = self.a_min + (action[:self.n_dim**2] + 1) / 2 * (self.a_max - self.a_min)
        return A_values.reshape(self.n_dim, self.n_dim)


# ============================================================
# 8) BASELINE CONTROLLER
# ============================================================

class BaselineController:
    """Encapsulates the logic of a baseline method."""
    
    def __init__(self, baseline_type, n_dim, H=None):
        """
        Parameters
        ----------
        baseline_type : str
            "correct_A", "identity_A", "moesp", "moesp_trueH", "holts_kf"
        n_dim : int
            State dimension
        H : np.ndarray, optional
            Observation matrix (required for some baselines)
        """
        self.baseline_type = baseline_type
        self.n_dim = n_dim
        self.H = H
        self.A_static = None  # For static baselines
    
    def set_A_static(self, A):
        """Set static A matrix (for correct_A and identity_A)."""
        self.A_static = A
    
    def get_A(self, t, measurement, measurements_full, x_pred):
        """Get A matrix at timestep t."""
        if self.baseline_type == "correct_A":
            return self.A_static
        elif self.baseline_type == "identity_A":
            return self.A_static  # Identity in this case
        elif self.baseline_type == "moesp":
            # Estimate A once from measurements
            if not hasattr(self, '_A_cached'):
                self._A_cached = moesp_identify(measurements_full, self.n_dim)
            return self._A_cached
        elif self.baseline_type == "moesp_trueH":
            if not hasattr(self, '_A_cached'):
                self._A_cached = moesp_identify_A_with_true_H(measurements_full, self.H, self.n_dim)
            return self._A_cached
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")


# ============================================================
# 9) MAIN TEST FUNCTION
# ============================================================

def test_agents(
    obs_level_indices=[1, 2, 3],
    n_trajectories=10,
    state_dim=3,
    agents_config=None,
    baselines_config=None,
    include_holts_kf=False,
    holts_alpha=0.1,
    holts_beta=0.05,
):
    """
    Evaluate specified agents and baselines.
    
    Parameters
    ----------
    obs_level_indices : list
        Indices of observability levels to test
    n_trajectories : int
        Number of trajectories per (A,H) couple
    state_dim : int
        State dimension
    agents_config : dict, optional
        Mapping {"agent_name": (agent_path, agent_type)}
        agent_type in ["measurement_only", "measurement_obs", "innovation_only", "innovation_obs"]
    baselines_config : dict, optional
        Mapping {"baseline_name": baseline_type}
        baseline_type in ["correct_A", "identity_A", "moesp", "moesp_trueH"]
    include_holts_kf : bool
        If True, automatically add "Holts_KF" to baselines
    holts_alpha : float
        Alpha parameter for Holt's exponential smoothing
    holts_beta : float
        Beta parameter for Holt's exponential smoothing
    
    Returns
    -------
    results : dict
        Dictionary of results keyed by observability level index
    """
    
    # Default configuration if not provided
    if agents_config is None:
        agents_config = {
            "RL_measurement": ("./path/to/agent_measurement.zip", "measurement_only"),
            "RL_measurement_obs": ("./path/to/agent_measurement_obs.zip", "measurement_obs"),
        }
    
    if baselines_config is None:
        baselines_config = {
            "rwKF": "identity_A",
            "true-A-KF": "correct_A",
            "MOESP": "moesp",
            "H-MOESP": "moesp_trueH",
        }
    
    # Add Holt's KF if requested
    if include_holts_kf:
        baselines_config["Holt-KF"] = "holts_kf"
    
    n_dim = state_dim
    n_steps = 100
    Q_gen = 1e-4
    R_gen = 1e-4
    Q_kf = Q_gen
    R_kf = R_gen
    init_min, init_max = 0.5, 1.5

    bank_dir = "./AH_bank_generated_h_fixed_v2_new_test_pour_deploiement_article/"
    rng_main = np.random.default_rng(42)
    results = {}

    print("="*70)
    print(f"EVALUATION RLKF")
    print(f"Agents: {list(agents_config.keys())}")
    print(f"Baselines: {list(baselines_config.keys())}")
    print("="*70)

    # Load agents
    agent_controllers = {}
    for agent_name, (agent_path, agent_type) in agents_config.items():
        try:
            agent_controllers[agent_name] = AgentController(agent_path, agent_type, n_dim)
            print(f"✓ Agent '{agent_name}' loaded ({agent_type})")
        except Exception as e:
            print(f"✗ Error loading agent '{agent_name}': {e}")

    for obs_level_idx in obs_level_indices:
        npz_file = os.path.join(bank_dir, f"AH_level_{obs_level_idx:02d}.npz")
        if not os.path.exists(npz_file):
            print(f"⊘ File not found: {npz_file}")
            continue

        npz = np.load(npz_file)
        A_array = npz["A_array"]
        H_array = npz["H_array"]
        obs_achieved = npz["obs_achieved"]
        obs_mean = float(np.mean(obs_achieved))

        print(f"\n--- Observability level {obs_level_idx} (obs_mean={obs_mean:.6f}) ---")

        # Initialize result accumulators
        rmse_results = {name: [] for name in agent_controllers}
        rmse_results.update({name: [] for name in baselines_config})
        
        rel_results = {name: [] for name in agent_controllers}
        rel_results.update({name: [] for name in baselines_config})
        
        rmse_std = {name: [] for name in agent_controllers}
        rmse_std.update({name: [] for name in baselines_config})
        
        rel_std = {name: [] for name in agent_controllers}
        rel_std.update({name: [] for name in baselines_config})

        for k in range(len(A_array)):
            A_true = A_array[k]
            H = H_array[k]
            obs_metric = obs_achieved[k]

            rmse_traj = {name: [] for name in agent_controllers}
            rmse_traj.update({name: [] for name in baselines_config})
            
            rel_traj = {name: [] for name in agent_controllers}
            rel_traj.update({name: [] for name in baselines_config})

            for traj_idx in range(n_trajectories):
                rng = np.random.default_rng(rng_main.integers(int(1e9)))
                x0_true = rng.uniform(init_min, init_max, size=n_dim)
                x0_init = rng.uniform(init_min, init_max, size=n_dim)

                true_x, measurements = generate_data(A_true, H, x0_true, n_steps, Q_gen, R_gen, rng)

                # ===== BASELINES =====
                for baseline_name, baseline_type in baselines_config.items():
                    baseline = BaselineController(baseline_type, n_dim, H)
                    
                    if baseline_type == "correct_A":
                        baseline.set_A_static(A_true)
                        est = kalman_filter(A_true, H, measurements, x0_init, Q_kf, R_kf, n_steps)
                    elif baseline_type == "identity_A":
                        baseline.set_A_static(np.eye(n_dim))
                        est = kalman_filter(np.eye(n_dim), H, measurements, x0_init, Q_kf, R_kf, n_steps)
                    elif baseline_type == "moesp":
                        A_est = moesp_identify(measurements, n_dim=n_dim)
                        est = kalman_filter(A_est, H, measurements, x0_init, Q_kf, R_kf, n_steps)
                    elif baseline_type == "moesp_trueH":
                        A_est = moesp_identify_A_with_true_H(measurements, H, n_dim=n_dim)
                        est = kalman_filter(A_est, H, measurements, x0_init, Q_kf, R_kf, n_steps
                    elif baseline_type == "holts_kf":
                        est = holts_exponential_smoothing_kf(
                            H, measurements, x0_init, Q_kf, R_kf, n_steps, n_dim,
                            alpha=holts_alpha, beta=holts_beta
                        )
                    
                    rmse_traj[baseline_name].append(compute_rmse(est, true_x))

                # ===== RL AGENTS =====
                for agent_name, agent_ctrl in agent_controllers.items():
                    # Adaptive Kalman: A decided at each timestep
                    def A_func(t, measurement, x_pred):
                        innovation = measurement - H @ x_pred
                        return agent_ctrl.predict_A(
                            measurement,
                            obs_metric=obs_metric,
                            innovation=innovation
                        )
                    
                    est = kalman_filter_adaptive(A_func, H, measurements, x0_init, Q_kf, R_kf, n_steps, n_dim)
                    rmse_traj[agent_name].append(compute_rmse(est, true_x))

            # Compute means and standard deviations
            for name in rmse_results.keys():
                rmse_results[name].append(np.mean(rmse_traj[name]))
                rmse_std[name].append(np.std(rmse_traj[name]))
                rel_results[name].append(np.mean(rel_traj[name]))
                rel_std[name].append(np.std(rel_traj[name]))

        # Flatten results: average over all (A, H) pairs
        result_entry = {"obs_mean": obs_mean}
        for name in rmse_results.keys():
            result_entry[f"rmse_{name}"] = np.mean(rmse_results[name])
            result_entry[f"rmse_{name}_std"] = np.mean(rmse_std[name])
            result_entry[f"rel_{name}"] = np.mean(rel_results[name])
            result_entry[f"rel_{name}_std"] = np.mean(rel_std[name])
        
        results[obs_level_idx] = result_entry

    return results


# ============================================================
# 10) PLOTTING FUNCTIONS
# ============================================================

def plot_rmse_results(results):
    """
    Display RMSE results with line styles grouped by method family.
    
    Parameters
    ----------
    results : dict
        Results dictionary from test_agents()
    """
    if not results:
        print("No results to display.")
        return

    # ---------- Display name mapping ----------
    display_names = {
        "RL_measurement_obs": "OM-E-RLTKF(ours)",
        "RL_measurement_obs_A": "OM-A-RLTKF(ours)",
        "rwKF": "rwKF",
        "true-A-KF": "true-A-KF",
        "MOESP": "MOESP",
        "H-MOESP": "H-MOESP",
        "Holt-KF": "Holt-KF",
    }

    # ---------- Style mapping by family ----------
    style_map = {
        # RL agents
        "OM-E-RLTKF": dict(linestyle="-", linewidth=2.5),
        "OM-A-RLTKF": dict(linestyle="-", linewidth=2.5),

        # KF baselines
        "rwKF": dict(linestyle="--", linewidth=2.0),
        "true-A-KF": dict(linestyle="--", linewidth=2.0),

        # Subspace identification
        "MOESP": dict(linestyle=":", linewidth=2.5),
        "H-MOESP": dict(linestyle=":", linewidth=2.5),

        # Holt
        "Holt-KF": dict(linestyle="-.", linewidth=2.0),
    }

    markers = ["o", "s", "^", "D", "x", "*", "v", "p", "h", "+"]

    # ---------- Extract method names ----------
    first_result = results[next(iter(results))]
    method_keys = sorted([
        k.replace("rmse_", "")
        for k in first_result.keys()
        if k.startswith("rmse_") and not k.endswith("_std")
    ])

    obs_means = np.array([results[k]["obs_mean"] for k in sorted(results.keys())])

    def plot_with_std(ax, x, y, ystd, label, marker, style):
        """Plot line with error band."""
        ax.plot(x, y, marker=marker, label=label, **style)
        ax.fill_between(x, y - ystd, y + ystd, alpha=0.15)

    # ================= FIGURE 1 : LOG–LOG =================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for idx, method in enumerate(method_keys):
        label = display_names.get(method, method)
        style = style_map.get(label, dict(linestyle="-", linewidth=2.0))
        marker = markers[idx % len(markers)]

        rmse = np.array([results[k][f"rmse_{method}"] for k in sorted(results.keys())])
        rmse_std = np.array([results[k][f"rmse_{method}_std"] for k in sorted(results.keys())])

        plot_with_std(ax1, obs_means, rmse, rmse_std, label, marker, style)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"observability metric $\kappa$", fontsize=20)
    ax1.set_ylabel("Average root mean square error", fontsize=20)
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend()
    plt.tight_layout()

    # ================= FIGURE 2 : LIN–LOG =================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for idx, method in enumerate(method_keys):
        label = display_names.get(method, method)
        style = style_map.get(label, dict(linestyle="-", linewidth=2.0))
        marker = markers[idx % len(markers)]

        rmse = np.array([results[k][f"rmse_{method}"] for k in sorted(results.keys())])
        rmse_std = np.array([results[k][f"rmse_{method}_std"] for k in sorted(results.keys())])

        plot_with_std(ax2, obs_means, rmse, rmse_std, label, marker, style)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"observability metric $\kappa$", fontsize=20)
    ax2.set_ylabel("Average root mean square error", fontsize=20)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=13, loc="upper right", framealpha=0.92)
    plt.tight_layout()

    plt.show()


# ===================== EXECUTION =====================
if __name__ == "__main__":
    # Agent configuration (adapt with your actual paths)
    agents_cfg = {
        "RL_measurement_obs": ("./your_agent/best_model.zip", "measurement_obs"),
        "RL_measurement_obs_A": ("./yout_agent/best_model.zip", "measurement_obs")
    }

    # Baseline configuration
    baselines_cfg = {
        "rwKF": "identity_A",
        "true-A-KF": "correct_A",
        "MOESP": "moesp",
        "H-MOESP": "moesp_trueH",
    }

    # Run tests
    results = test_agents(
        obs_level_indices=list(range(9)),
        n_trajectories=10,
        state_dim=3,
        agents_config=agents_cfg,
        baselines_config=baselines_cfg,
        include_holts_kf=True,
        holts_alpha=0.4,
        holts_beta=0.55,
    )
    
    # Plot results
    plot_rmse_results(results)