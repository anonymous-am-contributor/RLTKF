import os
import numpy as np

# ============================================================
# 1) GRAMIAN & OBSERVABILITY METRIC
# ============================================================

def compute_observability_gramian(A, H, horizon=30, R=1e-4):
    """
    Compute the observability Gramian matrix.
    
    Parameters
    ----------
    A : np.ndarray
        State transition matrix (n x n)
    H : np.ndarray
        Observation matrix (m x n)
    horizon : int
        Time horizon for Gramian computation
    R : float or np.ndarray
        Measurement noise covariance
    
    Returns
    -------
    W : np.ndarray
        Observability Gramian matrix (n x n)
    """
    A = np.asarray(A, float)
    H = np.asarray(H, float)
    n = A.shape[0]
    m = H.shape[0]

    if np.isscalar(R):
        R_inv = np.eye(m) / R
    else:
        R_inv = np.linalg.inv(R)

    W = np.zeros((n, n))
    Ak = np.eye(n)

    for _ in range(horizon):
        W += Ak.T @ H.T @ R_inv @ H @ Ak
        Ak = Ak @ A

    return 0.5 * (W + W.T)


def observability_metric(A, H, horizon=30, R=1e-4, eps=1e-12):
    """
    Compute observability metric: min_eigenvalue / trace.
    Higher value = more observable system.
    
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
    eps : float
        Small value to avoid division by zero
    
    Returns
    -------
    metric : float
        Observability metric value
    """
    W = compute_observability_gramian(A, H, horizon, R)
    eigvals = np.linalg.eigvalsh(W)
    lam_min = np.min(eigvals)
    tr = np.trace(W)
    return lam_min / (tr + eps)


# ============================================================
# 2) GENERATE A MATRIX: STABLE, STRUCTURED, LOG-COUPLED
# ============================================================

def sample_strictly_stable_A(
    n,
    rng,
    diag_min=0.98,
    diag_max=1.0,
    log_min=-11,
    log_max=0,
    eps_stability=1e-3
):
    """
    Generate a strictly stable matrix A with structured coupling.
    
    Properties:
    - Diagonal elements A_ii in [diag_min, diag_max]
    - Off-diagonal elements A_ij > 0 and log-uniformly distributed
    - Spectral radius guaranteed < 1 via infinity norm
    
    Parameters
    ----------
    n : int
        Dimension of matrix
    rng : np.random.Generator
        Random number generator
    diag_min : float
        Minimum diagonal value
    diag_max : float
        Maximum diagonal value
    log_min : float
        Minimum log10 value for off-diagonal elements
    log_max : float
        Maximum log10 value for off-diagonal elements
    eps_stability : float
        Stability margin epsilon
    
    Returns
    -------
    A : np.ndarray
        Stable matrix (n x n)
    """
    A = np.zeros((n, n))

    # Off-diagonal elements: positive, log-uniformly distributed
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = 10 ** rng.uniform(log_min, log_max)

    # Diagonal elements: uniform in specified range
    for i in range(n):
        A[i, i] = rng.uniform(diag_min, diag_max)

    # --- STABILITY GUARANTEE ---
    # Scale to ensure infinity norm < 1
    row_sums = np.sum(A, axis=1)
    max_row_sum = np.max(row_sums)

    if max_row_sum >= 1.0:
        alpha = (1.0 - eps_stability) / max_row_sum
        A *= alpha

    return A

# ============================================================
# 3) GENERATE H MATRIX (CUSTOM STRUCTURE)
# ============================================================

def generate_H_custom(n, n_ones=3, small_val=1e-8):
    """
    Generate observation matrix H with custom diagonal structure.
    
    First n_ones diagonal elements are set to 1.0,
    remaining diagonal elements are set to small_val.
    
    Parameters
    ----------
    n : int
        Dimension of matrix (e.g., 5)
    n_ones : int
        Number of diagonal elements to set to 1.0 (e.g., 3)
    small_val : float
        Value for other diagonal elements (e.g., 1e-8)
    
    Returns
    -------
    H : np.ndarray
        Observation matrix (n x n)
    """
    H = np.zeros((n, n))
    
    # Set first n_ones diagonal elements to 1.0
    for i in range(min(n_ones, n)):
        H[i, i] = 1.0
    
    # Fill remaining diagonal elements with small_val
    for i in range(n_ones, n):
        H[i, i] = small_val
    
    return H


# ============================================================
# 4) SEARCH FOR A MATRIX ACHIEVING TARGET OBSERVABILITY
# ============================================================

def generate_A_for_target_obs(
    H,
    target_obs,
    rng,
    n_iter=1000,
    horizon=30,
    R=1e-4
):
    """
    Generate a matrix A that achieves a target observability level.
    Uses random search to minimize distance from target.
    
    Parameters
    ----------
    H : np.ndarray
        Fixed observation matrix
    target_obs : float
        Target observability metric value
    rng : np.random.Generator
        Random number generator
    n_iter : int
        Number of iterations for random search
    horizon : int
        Horizon for Gramian computation
    R : float or np.ndarray
        Measurement noise covariance
    
    Returns
    -------
    best_A : np.ndarray
        Matrix A achieving closest to target observability
    best_val : float
        Achieved observability metric value
    """
    best_A = None
    best_val = None

    for _ in range(n_iter):
        A = sample_strictly_stable_A(
            n=H.shape[1],
            rng=rng,
            diag_min=0.98,
            diag_max=0.99,
            log_min=-10,
            log_max=0
        )

        val = observability_metric(A, H, horizon, R)

        if best_A is None or abs(val - target_obs) < abs(best_val - target_obs):
            best_A = A.copy()
            best_val = val

    return best_A, best_val


# ============================================================
# 5) GENERATE (A,H) COUPLES FOR OBSERVABILITY LEVEL
# ============================================================

def generate_couples_for_obs_level(
    H,
    target_obs,
    n_couples=5,
    n_iter_A=1000,
    horizon=30,
    R=1e-4,
    rng=None
):
    """
    Generate multiple (A, H) couples for a given observability level.
    H is fixed while A matrices are searched for each couple.
    
    Parameters
    ----------
    H : np.ndarray
        Fixed observation matrix
    target_obs : float
        Target observability level
    n_couples : int
        Number of couples to generate
    n_iter_A : int
        Iterations for finding each A matrix
    horizon : int
        Horizon for Gramian computation
    R : float or np.ndarray
        Measurement noise covariance
    rng : np.random.Generator, optional
        Random number generator (creates new if None)
    
    Returns
    -------
    couples : list
        List of (A, H) tuples
    obs_vals : np.ndarray
        Array of achieved observability values
    """
    if rng is None:
        rng = np.random.default_rng()
    
    couples = []
    obs_vals = []

    for _ in range(n_couples):
        A, obs = generate_A_for_target_obs(
            H, target_obs, rng, n_iter_A, horizon, R
        )
        couples.append((A, H))
        obs_vals.append(obs)

    return couples, np.array(obs_vals)


# ============================================================
# 5b) GENERATE (A,H) COUPLES FOR HIGH OBSERVABILITY
# ============================================================

def generate_couples_for_high_obs(
    n_dim,
    target_obs,
    n_couples=5,
    n_iter_A=1000,
    horizon=30,
    R=1e-4,
    n_ones_min=2,
    n_ones_max=None,
    small_val=1e-8,
    rng=None
):
    """
    Generate (A,H) couples for HIGH observability levels.
    Varies the number of 1.0 elements in H diagonal between n_ones_min and n_ones_max.
    
    Parameters
    ----------
    n_dim : int
        Dimension of matrices
    target_obs : float
        Target observability level
    n_couples : int
        Number of couples to generate
    n_iter_A : int
        Iterations for finding each A matrix
    horizon : int
        Horizon for Gramian computation
    R : float or np.ndarray
        Measurement noise covariance
    n_ones_min : int
        Minimum number of 1.0 diagonal elements (e.g., 2)
    n_ones_max : int, optional
        Maximum number of 1.0 diagonal elements (defaults to n_dim)
    small_val : float
        Value for non-1.0 diagonal elements
    rng : np.random.Generator, optional
        Random number generator (creates new if None)
    
    Returns
    -------
    couples : list
        List of (A, H) tuples
    obs_vals : np.ndarray
        Array of achieved observability values
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if n_ones_max is None:
        n_ones_max = n_dim
    
    couples = []
    obs_vals = []

    for _ in range(n_couples):
        # Randomly alternate between n_ones_min and n_ones_max
        n_ones = rng.integers(n_ones_min, n_ones_max + 1)
        
        H = generate_H_custom(n_dim, n_ones, small_val)
        
        A, obs = generate_A_for_target_obs(
            H, target_obs, rng, n_iter_A, horizon, R
        )
        couples.append((A, H))
        obs_vals.append(obs)

    return couples, np.array(obs_vals)


# ============================================================
# 6) SAVE BANK
# ============================================================

def generate_and_save_AH_bank(
    obs_levels,
    output_dir,
    n_couples_per_level=50,
    n_dim=5,
    n_ones_H=3,
    small_val_H=1e-8,
    n_iter_A=1000,
    seed=42
):
    """
    Generate and save (A, H) bank across multiple observability levels.
    
    Parameters
    ----------
    obs_levels : np.ndarray
        Array of target observability levels
    output_dir : str
        Output directory for .npz files
    n_couples_per_level : int
        Number of couples per observability level
    n_dim : int
        Dimension of matrices (e.g., 5)
    n_ones_H : int
        Number of 1.0 elements in H diagonal
    small_val_H : float
        Value for non-1.0 H diagonal elements
    n_iter_A : int
        Iterations for finding A matrices
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    results : dict
        Dictionary with statistics for each level
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    # Generate H matrix once (it is fixed across all couples)
    H = generate_H_custom(n_dim, n_ones_H, small_val_H)
    print(f"\nGenerated H matrix (dim {n_dim}, {n_ones_H} ones):")
    print(H)

    results = {}

    for idx, target_obs in enumerate(obs_levels):
        print(f"\nProcessing level {idx}: target_obs = {target_obs:.2e}")
        
        couples, obs_vals = generate_couples_for_obs_level(
            H=H,
            target_obs=target_obs,
            n_couples=n_couples_per_level,
            n_iter_A=n_iter_A,
            rng=rng
        )

        A_array = np.array([c[0] for c in couples])
        H_array = np.array([c[1] for c in couples])

        filename = os.path.join(output_dir, f"AH_level_{idx:02d}.npz")
        np.savez(
            filename,
            A_array=A_array,
            H_array=H_array,
            obs_achieved=obs_vals,
            target_obs=target_obs
        )

        results[idx] = {
            "target_obs": target_obs,
            "mean_obs": obs_vals.mean(),
            "min_obs": obs_vals.min(),
            "max_obs": obs_vals.max(),
        }
        
        print(f"  Saved: {filename}")
        print(f"    Achieved obs: {obs_vals.mean():.2e} (min: {obs_vals.min():.2e}, max: {obs_vals.max():.2e})")
    
    return results


# ============================================================
# 6b) SAVE BANK FOR HIGH OBSERVABILITY
# ============================================================

def generate_and_save_AH_bank_high_obs(
    obs_levels,
    output_dir,
    n_couples_per_level=50,
    n_dim=5,
    n_ones_min=2,
    n_ones_max=None,
    small_val_H=1e-8,
    n_iter_A=1000,
    seed=42,
    start_idx=0
):
    """
    Generate and save (A, H) bank for HIGH observability levels (1e-2 to 1/n).
    Varies the number of 1.0 elements in H at each couple.
    
    Parameters
    ----------
    obs_levels : np.ndarray
        Array of high observability target levels
    output_dir : str
        Output directory for .npz files
    n_couples_per_level : int
        Number of couples per observability level
    n_dim : int
        Dimension (e.g., 5)
    n_ones_min : int
        Minimum number of 1.0 elements (e.g., 2)
    n_ones_max : int, optional
        Maximum number of 1.0 elements (defaults to n_dim)
    small_val_H : float
        Value for non-1.0 diagonal elements
    n_iter_A : int
        Iterations for finding A matrices
    seed : int
        Random seed for reproducibility
    start_idx : int
        Starting index for filenames (e.g., 12 if 12 low-obs levels already exist)
    
    Returns
    -------
    results : dict
        Dictionary with statistics for each level
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    if n_ones_max is None:
        n_ones_max = n_dim
    
    results = {}

    for idx, target_obs in enumerate(obs_levels):
        print(f"\nProcessing HIGH observability level {start_idx + idx}: target_obs = {target_obs:.2e}")
        
        couples, obs_vals = generate_couples_for_high_obs(
            n_dim=n_dim,
            target_obs=target_obs,
            n_couples=n_couples_per_level,
            n_iter_A=n_iter_A,
            n_ones_min=n_ones_min,
            n_ones_max=n_ones_max,
            small_val=small_val_H,
            rng=rng
        )

        A_array = np.array([c[0] for c in couples])
        H_array = np.array([c[1] for c in couples])

        filename = os.path.join(output_dir, f"AH_level_{start_idx + idx:02d}.npz")
        np.savez(
            filename,
            A_array=A_array,
            H_array=H_array,
            obs_achieved=obs_vals,
            target_obs=target_obs
        )

        results[idx] = {
            "target_obs": target_obs,
            "mean_obs": obs_vals.mean(),
            "min_obs": obs_vals.min(),
            "max_obs": obs_vals.max(),
        }
        
        print(f"  ✓ Saved: {filename}")
        print(f"    Achieved obs: {obs_vals.mean():.2e} (min: {obs_vals.min():.2e}, max: {obs_vals.max():.2e})")
    
    return results


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    # Example
    n_dim = 3
    n_ones_H = 1
    small_val_H = 1e-8
    
    bank_dir = "./test_bank/"
    
    # ========== LOW OBSERVABILITY LEVELS ==========
    obs_levels_low = np.logspace(-9, -2, 8)
    print(f"Low observability levels: {obs_levels_low}")
    print(f"\n[!] Generating LOW observability for dim={n_dim}, {n_ones_H} ones")
    
    results_low = generate_and_save_AH_bank(
        obs_levels=obs_levels_low,
        output_dir=bank_dir,
        n_couples_per_level=10,
        n_dim=n_dim,
        n_ones_H=n_ones_H,
        small_val_H=small_val_H,
        n_iter_A=4000,
        seed=42
    )
    
    print("\n=== SUMMARY: LOW OBSERVABILITY LEVELS ===")
    for idx, stats in results_low.items():
        print(f"Level {idx}: target={stats['target_obs']:.2e}, mean_obs={stats['mean_obs']:.2e}")
    
    # ========== HIGH OBSERVABILITY LEVELS ==========
    # Between 1e-2 and 1/n_dim (1/5 = 0.2)
    obs_levels_high = np.logspace(np.log10(1/n_dim), np.log10(1/n_dim), 1)
    print(f"\n\nHigh observability levels (1e-2 to 1/{n_dim}): {obs_levels_high}")
    print(f"\n[!] Generating HIGH observability with varied H")
    
    results_high = generate_and_save_AH_bank_high_obs(
        obs_levels=obs_levels_high,
        output_dir=bank_dir,
        n_couples_per_level=10,
        n_dim=n_dim,
        n_ones_min=1,              # Minimum: 1 one
        n_ones_max=n_dim,          # Maximum: n_dim ones
        small_val_H=small_val_H,
        n_iter_A=4000,
        seed=43,
        start_idx=8          # Starts after 8 low observability levels
    )
    
    print("\n=== SUMMARY: HIGH OBSERVABILITY LEVELS ===")
    for idx, stats in results_high.items():
        print(f"Level {8 + idx}: target={stats['target_obs']:.2e}, mean_obs={stats['mean_obs']:.2e}")