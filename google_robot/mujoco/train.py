import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from google_robot_env import GoogleRobotPickPlaceEnv

# --- CONFIGURATION ---
SCENE_PATH = "/Users/dhruvpatel29/mujoco/google_robot/google_robot/scene.xml"
# Use 8 or 16 depending on your Mac (M1/M2/M3 Max can handle 16 easily)
NUM_ENVS = 30           
TOTAL_TIMESTEPS = 2_000_000  # Increased to 2M for better results
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def make_env(rank, seed=0):
    def _init():
        # It's important to pass the path to each instance
        env = GoogleRobotPickPlaceEnv(SCENE_PATH)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # 1. Start Parallel Environments
    # This uses all your CPU cores to run the MuJoCo physics
    print(f"Launching {NUM_ENVS} parallel robots on {DEVICE}...")
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    
    # 2. Add Normalization (REQUIRED for Robotics)
    # This keeps the math within a range the GPU can handle efficiently
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. PPO Optimized for Apple Silicon + Parallel Physics
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device=DEVICE,
        n_steps=2048,             # Total samples per iteration = 2048 * 30 = 61,440
        batch_size=1024,          # Large batch size makes the GPU (MPS) much faster
        n_epochs=10,              # Number of times to reuse the data per update
        learning_rate=3e-4,
        gamma=0.99,               # Standard discount factor
        gae_lambda=0.95,
        ent_coef=0.005,           # Reduced slightly to focus on learning known good paths
        policy_kwargs=dict(net_arch=[256, 256, 256]) # Deeper network for 9-DOF
    )

    # 4. Training Loop
    print(f"Training started. Target: {TOTAL_TIMESTEPS} steps.")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current progress...")

    # 5. Save Model and Normalization Stats
    model.save("google_robot_ppo_final")
    env.save("vec_normalize_final.pkl")
    
    print("------------------------------------------")
    print("DONE! Model and Normalizers saved.")