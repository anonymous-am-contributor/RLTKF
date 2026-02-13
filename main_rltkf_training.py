import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Imports from your code
from kalman_env import KalmanEnv_new
from kf_utils import (
    create_dataset_observability_controlled,
)

# ===================== Checkpoints callback =====================
class PeriodicCheckpointCallback(BaseCallback):
    """
    Periodically saves the PPO model during training.
    """
    def __init__(self, save_freq, save_path, name_prefix="ppo_checkpoint", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            save_file = os.path.join(
                self.save_path,
                f"{self.name_prefix}_step_{self.num_timesteps}"
            )
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"[Checkpoint] Model saved at {self.num_timesteps} steps")

        return True

# ===================== LOAD UNIFIED AH BANK =====================

def load_unified_AH_bank(bank_dir="./your_bank/"):
    """
    Loads ALL .npz files from bank_dir and merges them
    into a single bank containing (A, H, obs) tuples from all levels.
    """
    if not os.path.exists(bank_dir):
        raise FileNotFoundError(f"Bank directory not found: {bank_dir}")
    
    print("="*70)
    print("LOADING UNIFIED BANK")
    print("="*70)
    print(f"Directory: {bank_dir}\n")
    
    unified_bank = []
    obs_values_all = []
    level_info = []
    
    # List all .npz files
    npz_files = sorted([f for f in os.listdir(bank_dir) if f.endswith('.npz')])
    
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {bank_dir}")
    
    # Load each level and add to unified bank
    for file_idx, npz_filename in enumerate(npz_files):
        npz_path = os.path.join(bank_dir, npz_filename)
        npz_data = np.load(npz_path)
        
        A_array = npz_data["A_array"]       # shape (n_couples, n_dim, n_dim)
        H_array = npz_data["H_array"]       # shape (n_couples, n_dim, n_dim)
        obs_achieved = npz_data["obs_achieved"]  # shape (n_couples,)
        target_obs = float(npz_data["target_obs"])
        
        # Add all couples from this level to the bank
        for couple_idx in range(len(A_array)):
            A = A_array[couple_idx]
            H = H_array[couple_idx]
            obs = float(obs_achieved[couple_idx])
            unified_bank.append((A, H, obs))
            obs_values_all.append(obs)
        
        obs_mean = np.mean(obs_achieved)
        obs_min = np.min(obs_achieved)
        obs_max = np.max(obs_achieved)
        
        level_info.append({
            "level_idx": file_idx,
            "filename": npz_filename,
            "target_obs": target_obs,
            "obs_mean": obs_mean,
            "obs_min": obs_min,
            "obs_max": obs_max,
            "n_couples": len(A_array)
        })
        
        print(f"[Level {file_idx:02d}] {npz_filename}")
        print(f"  target_obs: {target_obs:.2e}")
        print(f"  obs_achieved: mean={obs_mean:.2e}, min={obs_min:.2e}, max={obs_max:.2e}")
        print(f"  n_couples: {len(A_array)}\n")
    
    # Global statistics
    obs_values_all = np.array(obs_values_all)
    print("="*70)
    print("UNIFIED BANK GLOBAL STATISTICS")
    print("="*70)
    print(f"Number of levels: {len(npz_files)}")
    print(f"Total (A,H) couples: {len(unified_bank)}")
    print(f"Observability:")
    print(f"  min: {np.min(obs_values_all):.2e}")
    print(f"  mean: {np.mean(obs_values_all):.2e}")
    print(f"  max: {np.max(obs_values_all):.2e}")
    print("="*70 + "\n")
    
    return unified_bank, obs_values_all, level_info

# ===================== CUSTOM CALLBACK =====================

class RewardLogger(BaseCallback):
    """Logs rewards at each episode"""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_reward = 0.0
        self.episode_count = 0

    def _on_step(self):
        self.episode_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.episode_reward)
            self.episode_count += 1
            self.episode_reward = 0.0
        return True

# ===================== PARAMETERS (An example) =====================

n_dim = 3
n_steps = 300
action_type = "transition-matrix"
bias_list = [0.0] * n_dim #If you want to have a transition model : x_k+1 = Ax_k + bias

# === DATASET PARAMETERS ===
dataset_params = {"n_steps": n_steps, "Q_coeff": 1e-4, "R_coeff": 1e-4,"seed": 42}

# === KALMAN FILTER CONFIGURATION (An example) ===
kf_config = {
    "initial_state": np.array([1.0, 0.9, 0.8]),
    "initial_P": 1e-5 * np.eye(n_dim),
    "Q": [1e-4] * n_dim,
    "R": [1e-4],
    "A": np.eye(n_dim),
    "H": np.array([[1.0, 1e-8, 1e-8]]),
    "bias_list": bias_list,
}

# ===================== 1) LOAD UNIFIED BANK =====================

print("\n")
bank_dir = "./test_bank/"
unified_bank, obs_values_all, level_info = load_unified_AH_bank(bank_dir=bank_dir)

# ===================== 2) CREATE ENVIRONMENTS =====================

log_dir = "./name_to_choose/"
os.makedirs(log_dir, exist_ok=True)

print("CREATING TRAINING ENVIRONMENT")
print("="*70)

env = KalmanEnv_new(
    kf_config=kf_config,
    dataset_fn=create_dataset_observability_controlled,
    dataset_params=dataset_params,
    dataset_bank=unified_bank,  # Unified bank with ALL couples and levels
    action_type=action_type,
    bias_list=bias_list,
    log_dir=log_dir,
    mode="train",
    include_obs_in_observation=False,
    observation_type="measurements",
    # obs_metric_type="pre_computed", #pre_computed or hankel_online if you want to give an online estimate of kappa during training
)
env = Monitor(env, log_dir)

eval_env = KalmanEnv_new(
    kf_config=kf_config,
    dataset_fn=create_dataset_observability_controlled,
    dataset_params=dataset_params,
    dataset_bank=unified_bank,  # Same unified bank for eval
    action_type=action_type,
    bias_list=bias_list,
    log_dir=log_dir,
    mode="train",
    include_obs_in_observation=False,
    observation_type="measurements",
    # obs_metric_type="pre_computed",
)
eval_env = Monitor(eval_env, log_dir)

print(f"Environment created with {len(unified_bank)} (A,H) couples")
print(f"Log directory: {log_dir}\n")

# ===================== 3) CALLBACKS =====================

reward_logger = RewardLogger()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=2000,
    deterministic=True,
    render=False
)

checkpoint_dir = os.path.join(log_dir, "checkpoints")
checkpoint_callback = PeriodicCheckpointCallback(
    save_freq=1_000_000,
    save_path=checkpoint_dir,
    name_prefix="ppo_rlkf",
    verbose=1
)

# ===================== 4) TRAINING =====================

print("="*70)
print("LAUNCHING UNIFIED TRAINING")
print("="*70)
print(f"Mixed couples from all observability levels")
print(f"Observability: [{np.min(obs_values_all):.2e}, {np.max(obs_values_all):.2e}]")
print("="*70 + "\n")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    device='cpu'
)

total_timesteps = 10_000_000
model.learn(
    total_timesteps=total_timesteps,
    callback=[reward_logger, eval_callback, checkpoint_callback]
)

# ===================== 5) SAVE MODEL =====================

model.save(os.path.join(log_dir, "ppo_rlkf_unified"))
print(f"\n Model saved: {os.path.join(log_dir, 'ppo_rlkf_unified.zip')}")

# ===================== 6) PLOTS =====================

rewards = reward_logger.episode_rewards

# Plot 1: Rewards over episodes
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(rewards, linewidth=1, alpha=0.7, label='Episode Reward')

# Add moving average
if len(rewards) > 100:
    moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    ax.plot(range(100-1, len(rewards)), moving_avg, linewidth=2, color='red', label='Moving Avg (100 episodes)')

ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
ax.set_title('Training Rewards - Unified Training (Mixed Observability Levels)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "rewards_unified.png"), dpi=150)
plt.close()

# Plot 2: Distribution of observabilities in bank
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(obs_values_all, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
ax.set_xlabel('Observability Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Observability Levels in Unified Bank', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "obs_distribution.png"), dpi=150)
plt.close()

# Plot 3: Final statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Final reward
final_reward = rewards[-1]
avg_last_100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
avg_last_1000 = np.mean(rewards[-1000:]) if len(rewards) >= 1000 else np.mean(rewards)

axes[0, 0].bar(['Final', 'Last 100', 'Last 1000'], 
               [final_reward, avg_last_100, avg_last_1000],
               color=['steelblue', 'coral', 'green'])
axes[0, 0].set_ylabel('Reward', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Training Metrics', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Episodes count
axes[0, 1].text(0.5, 0.5, f"{len(rewards)} episodes", 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=axes[0, 1].transAxes)
axes[0, 1].set_title('Total Episodes', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Bank composition
level_names = [info['filename'].replace('.npz', '') for info in level_info]
level_counts = [info['n_couples'] for info in level_info]
axes[1, 0].barh(range(len(level_names)), level_counts, color='coral')
axes[1, 0].set_yticks(range(len(level_names)))
axes[1, 0].set_yticklabels(level_names, fontsize=9)
axes[1, 0].set_xlabel('Number of Couples', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Bank Composition by Level', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Observability stats
axes[1, 1].bar(['Min', 'Mean', 'Max'], 
               [np.min(obs_values_all), np.mean(obs_values_all), np.max(obs_values_all)],
               color=['lightcoral', 'steelblue', 'lightgreen'])
axes[1, 1].set_ylabel('Observability', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Observability Statistics', fontsize=12, fontweight='bold')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(log_dir, "training_summary.png"), dpi=150, bbox_inches='tight')
plt.close()

# ===================== 7) FINAL SUMMARY =====================

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Episodes trained: {len(rewards)}")
print(f"Total timesteps: {total_timesteps:,}")
print(f"\nRewards:")
print(f"  Final: {final_reward:.4f}")
print(f"  Avg last 100: {avg_last_100:.4f}")
print(f"  Avg last 1000: {avg_last_1000:.4f}")
print(f"\nBank:")
print(f"  Total couples: {len(unified_bank)}")
print(f"  Levels: {len(level_info)}")
print(f"  Observability: [{np.min(obs_values_all):.2e}, {np.max(obs_values_all):.2e}]")
print(f"\nFiles saved in: {log_dir}")
print(f"  - ppo_rlkf_unified.zip (model)")
print(f"  - rewards_unified.png")
print(f"  - obs_distribution.png")
print(f"  - training_summary.png")
print("="*70)

env.close()
eval_env.close()

print("\n Training complete!")