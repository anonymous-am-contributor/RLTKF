import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.metrics import mean_squared_error
import random
from kf_utils import *
import warnings
warnings.filterwarnings("ignore")


class KalmanEnv_new(gym.Env):
    def __init__(self,
                 kf_config,  # Au lieu de twin_reference
                 dataset_fn,
                 dataset_params=None,
                 dataset_bank=None,
                 action_type="noise",
                 observation_type="innovation",  # "innovation" or "measurements"
                 mode="train",
                 bias_list=None,
                 h_fn=None,
                 log_dir=None,
                 true_states=None,
                 measurements=None,
                 sensors=None,
                 include_obs_in_observation=True,
                 obs_metric_type="none"):
        super(KalmanEnv_new, self).__init__()

        # === Store configuration parameters
        self.kf_config = kf_config  # Config for KF creation
        self.dataset_fn = dataset_fn
        self.dataset_params = dataset_params or {}
        self.action_type = action_type
        self.observation_type = observation_type
        self.mode = mode
        self.bias_list = bias_list
        self.h_fn = h_fn
        self.log_dir = log_dir
        self.dataset_bank = dataset_bank
        self.include_obs_in_observation = include_obs_in_observation
        self.obs_metric_type = obs_metric_type

        # === Initial dataset generation (before first reset)
        # Store A_sel and H_sel for KF configuration
        self.A_sel = None
        self.H_sel = None
        
        if self.mode == "train":
            if self.dataset_bank is not None and len(self.dataset_bank) > 0:
                selection = random.choice(self.dataset_bank)
                if len(selection) == 3:
                    A_sel, H_sel, obs_sel = selection
                else:
                    A_sel, H_sel = selection
                    obs_sel = None
                
                # Store for use in KF
                self.A_sel = A_sel.copy()
                self.H_sel = H_sel.copy()
                
                dp = dict(self.dataset_params)
                dp["A"] = A_sel.copy()
                dp["H"] = H_sel.copy()
                dp["seed"] = np.random.randint(0, 2**31 - 1)
                data = self.dataset_fn(**dp)
            else:
                # Use default H from dataset_params
                self.H_sel = self.dataset_params.get("H", np.eye(3)[:1])
                data = self.dataset_fn(**self.dataset_params)
            
            if isinstance(data, dict):
                self.true_states = data["true_state_hist"]
                self.measurements = data["observed_measure"]
                self.x0 = data.get("x0", np.zeros(3))
                self.obs_metric_true = data.get("cond_gram", None)
            else:
                self.true_states, self.measurements, self.sensors, self.obs_metric_true = data
                
        elif self.mode == "test":
            self.true_states = true_states
            self.measurements = measurements
            self.sensors = sensors

        # === Default bias (if not provided)
        self.bias_list = bias_list or [0] * self.true_states.shape[1]
        self.non_obs_errors = []

        self.T = len(self.true_states)
        self.t = 0

        self.state_dim = self.true_states.shape[1]
        self.obs_dim = self.measurements.shape[1]
        
        self.current_obs_value = None
        self.window_size = 30

        # === Define action space depending on action type
        if self.action_type == "transition-matrix":
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.state_dim**2,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown action_type {self.action_type}")

        # === Observation space size (possibly augmented with observability metric)
        obs_space_size = self.obs_dim + (1 if self.include_obs_in_observation else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32)

        # === Logging containers
        self.estimated_states = []
        self.true_states_log = []
        self.innovations_log = []
        self.covariances_log = []
        self.actions_log = []
        self.A_history = []
        self.A_true_history = []
        self.obs_history = []

        # === Initialize Kalman Filter (replaces Twin)
        self.kf = self._create_kalman_filter()
        self.estimated_states.append(self.kf.get_state())

        self.rlkf_rmse = 0
        self.ukf_rmse = 0
        self.x0 = np.array([0, 0, 0])

        self._dataset_params_for_reset = None

        self.reset()

    # ==========================================================
    # === Create Kalman Filter Instance
    # ==========================================================
    def _create_kalman_filter(self) -> LinearKalmanFilter:
        """
        Create an instance of the linear Kalman Filter.
        Uses H_sel if available from dataset selection, otherwise uses kf_config.
        """
        state_dim = self.state_dim
        obs_dim = self.obs_dim
        
        # State and initial covariance
        initial_state = self.kf_config.get("initial_state", np.zeros(state_dim))
        initial_P = self.kf_config.get("initial_P", 1e-5 * np.eye(state_dim))
        
        # Noise covariances
        Q = self.kf_config.get("Q", [1e-4] * state_dim)
        R = self.kf_config.get("R", [1e-3] * obs_dim)
        
        # Matrices: use H_sel if available, otherwise use kf_config
        A = self.A_sel if self.A_sel is not None else self.kf_config.get("A", np.eye(state_dim))
        H = self.H_sel if self.H_sel is not None else self.kf_config.get("H", self._default_H())
        
        # Biases
        bias_list = self.kf_config.get("bias_list", [0] * state_dim)
        
        kf = LinearKalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            initial_state=initial_state,
            initial_P=initial_P,
            Q=Q,
            R=R,
            A=A,
            H=H,
            bias_list=bias_list
        )
        
        return kf
    
    def _default_H(self) -> np.ndarray:
        """Default observation matrix"""
        H = np.zeros((self.obs_dim, self.state_dim))
        for i in range(min(self.obs_dim, self.state_dim)):
            H[i, i] = 1.0
        return H

    # ==========================================================
    # === Observability metric (Hankel condition number)
    # ==========================================================
    def _compute_hankel_observability(self, measurements_window):
        """
        Compute the condition number of a Hankel matrix
        built from a sliding window of measurements.
        """

        if measurements_window.shape[0] < 2:
            return np.inf
        
        n_samples = measurements_window.shape[0]
        obs_dim = measurements_window.shape[1]
        hankel_size = max(2, n_samples // 2)
        
        hankel_matrices = []

        # Build Hankel matrix for each output dimension
        for d in range(obs_dim):
            y = measurements_window[:, d]
            hankel = np.zeros((hankel_size, hankel_size))

            for i in range(hankel_size):
                for j in range(hankel_size):
                    if i + j < len(y):
                        hankel[i, j] = y[i + j]

            hankel_matrices.append(hankel)
        
        hankel_full = np.vstack(hankel_matrices)
        
        try:
            cond_number = np.linalg.cond(hankel_full)
            if np.isinf(cond_number) or np.isnan(cond_number):
                return 1e10
            return float(cond_number)
        except:
            return 1e10

    # ==========================================================
    # === Update observability metric depending on mode
    # ==========================================================
    def _update_observation_with_hankel_metric(self):
        if self.obs_metric_type == "online_hankel":
            start_idx = max(0, self.t - self.window_size + 1)
            measurements_window = self.measurements[start_idx:self.t + 1]
            self.current_obs_value = self._compute_hankel_observability(measurements_window)

        elif self.obs_metric_type == "pre_computed":
            pass

        else:
            self.current_obs_value = None

    # ==========================================================
    # === Augment observation vector with observability metric
    # ==========================================================
    def _augment_observation(self, base_obs):
        if self.include_obs_in_observation and self.current_obs_value is not None:

            if base_obs.ndim == 1:
                base_obs_arr = base_obs
            else:
                base_obs_arr = np.array(base_obs, dtype=np.float32).flatten()
            
            if np.isinf(self.current_obs_value) or np.isnan(self.current_obs_value):
                obs_log = 10.0
            else:
                obs_log = np.log10(self.current_obs_value + 1e-20)
            
            augmented = np.concatenate([base_obs_arr, [obs_log]])
            return augmented.astype(np.float32)
        
        return np.asarray(base_obs, dtype=np.float32)

    # ==========================================================
    # === STEP FUNCTION
    # ==========================================================
    def step(self, action):

        true_state = self.true_states[self.t]
        measure_t = self.measurements[self.t]

        A_true = None
        if (hasattr(self, "_dataset_params_for_reset") and 
            self._dataset_params_for_reset is not None and 
            "A" in self._dataset_params_for_reset):
            A_true = self._dataset_params_for_reset["A"].copy()

        # === Transition matrix controlled by RL agent
        if self.action_type == "transition-matrix":

            action = np.clip(np.array(action), -1, 1)

            a_min, a_max = 1e-18, 1.0
            A_values = a_min + (action[:self.state_dim**2] + 1) / 2 * (a_max - a_min)
            A_matrix = A_values.reshape(self.state_dim, self.state_dim)

            self.A_history.append(A_matrix.copy())

            q_diag = [1e-4] * self.state_dim

            # === Kalman Filter step using our object
            est_state, est_cov = kf_step_simple(
                self.kf,
                measurements=self.measurements,
                time=self.t,
                A_matrix=A_matrix,
                bias_list=self.bias_list
            )
            print(est_state)

            # Uncomment this section to use distance to A true as reward
            # if A_true is not None:
            #     distance_A = np.linalg.norm(A_matrix - A_true, ord=2)
            #     norm_A_true = np.linalg.norm(A_true, ord=2)
            #     reward = -distance_A / (norm_A_true + 1e-10)
            # else:
            #     reward = 0

        # === Innovation
        y_pred = self.kf.get_measurement_prediction()
        innovation = (measure_t - y_pred)**2
        self.innovations_log.append(innovation)

        # === Reward: negative RMSE
        reward = -np.sqrt(mean_squared_error(est_state, true_state))

        est_last = est_state[-1]
        true_last = true_state[-1]
        err_non_obs = (est_last - true_last)**2
        self.non_obs_errors.append(err_non_obs)

        # === Logging
        self.estimated_states.append(est_state)
        self.true_states_log.append(true_state)
        self.covariances_log.append(est_cov)
        self.actions_log.append(action)

        done = self.t >= self.T - 1
        self.t += 1

        self._update_observation_with_hankel_metric()

        if self.observation_type == "innovation":
            base_obs = self._get_observation(innovation)
        else:
            if not done:
                base_obs = self._get_observation(self.measurements[self.t])
            else:
                base_obs = self._get_observation(self.measurements[self.t-1])

        obs = self._augment_observation(base_obs)

        return obs, reward, done, None, {}

    def _get_observation(self, innovation):
        return np.array(innovation)

    def render(self, mode="human"):
        pass

    def get_logs(self):
        return {
            'estimates': self.estimated_states,
            'true_states': self.true_states_log,
            'innovations': self.innovations_log,
            'covariances': self.covariances_log,
            'actions': self.actions_log
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment and sample new (A,H) couple from dataset_bank.
        
        Generates a new trajectory using dataset_fn with the sampled matrices.
        Initializes Kalman Filter with random initial state in [0.5, 1.5].
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional options (not used)
        
        Returns
        -------
        obs : np.ndarray
            Initial observation
        info : dict
            Additional information
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        self.t = 0
        self.estimated_states = []
        self.true_states_log = []
        self.innovations_log = []
        self.covariances_log = []
        self.actions_log = []
        self.A_history = []
        self.A_true_history = []
        self.obs_history = []
        self.non_obs_errors = []
        
        # === Sample new (A, H) couple from dataset_bank during training
        if self.mode == "train":
            if self.dataset_bank is not None and len(self.dataset_bank) > 0:
                # Randomly select a couple (A, H) from the bank
                selection = random.choice(self.dataset_bank)
                selection_len = len(selection)
                
                if selection_len == 3:
                    A_sel, H_sel, obs_sel = selection
                else:
                    A_sel, H_sel = selection
                    obs_sel = None
                
                # Store for use in KF creation
                self.A_sel = A_sel.copy()
                self.H_sel = H_sel.copy()
                
                # Set pre-computed observability metric if available
                if self.obs_metric_type == "pre_computed" and obs_sel is not None:
                    self.current_obs_value = obs_sel
                
                # Prepare dataset parameters with selected A and H
                dp = dict(self.dataset_params)
                dp["A"] = A_sel.copy()
                dp["H"] = H_sel.copy()
                if seed is not None:
                    dp["seed"] = seed
                else:
                    dp["seed"] = np.random.randint(0, 2**31 - 1)
                
                self._dataset_params_for_reset = dp
            else:
                # No bank, use default parameters
                dp = dict(self.dataset_params)
                if seed is not None:
                    dp["seed"] = seed
                else:
                    dp["seed"] = np.random.randint(0, 2**31 - 1)
                self._dataset_params_for_reset = dp
            
            # Generate new trajectory with the selected (A, H)
            data = self.dataset_fn(**self._dataset_params_for_reset)
            
            if isinstance(data, dict):
                self.true_states = data["true_state_hist"]
                self.measurements = data["observed_measure"]
                self.x0 = data.get("x0", np.zeros(self.state_dim if hasattr(self, "state_dim") else 3))
                self.obs_metric_true = data.get("cond_gram", None)
            else:
                self.true_states, self.measurements, self.sensors, self.obs_metric_true = data
            
            # Update state and observation dimensions
            self.state_dim = self.true_states.shape[1]
            self.obs_dim = self.measurements.shape[1]
            
            # Store true A matrix if available
            if (hasattr(self, "_dataset_params_for_reset") and 
                self._dataset_params_for_reset is not None and
                "A" in self._dataset_params_for_reset):
                self.A_true_history.append(self._dataset_params_for_reset["A"].copy())
            
            self.T = len(self.true_states)
        
        # === Generate random initial state in [0.5, 1.5]
        rng = np.random.default_rng(seed)
        random_initial_state = rng.uniform(0.5, 1.5, size=self.state_dim)
        
        # === Reset the Kalman Filter with:
        # - Random initial state (new each reset)
        # - H_sel from dataset_bank
        # - Same Q and R from kf_config
        initial_P = self.kf_config.get("initial_P", 1e-5 * np.eye(self.state_dim))
        
        # Recreate KF with potentially new H matrix (from H_sel)
        self.kf = self._create_kalman_filter()
        self.kf.reset_to_initial(random_initial_state, initial_P)
        
        self.estimated_states.append(self.kf.get_state())
        
        # Update observability metric
        self._update_observation_with_hankel_metric()
        
        # If current_obs_value is still None, set a default value
        if self.current_obs_value is None:
            self.current_obs_value = 1.0  # Default value
        
        # Build initial observation
        if self.observation_type == "innovation":
            # Initial innovation is zero (no previous measurement)
            base_obs = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            # Observation is current measurement
            base_obs = self._get_observation(self.measurements[0])
        
        obs = self._augment_observation(base_obs)
        
        return obs, {}