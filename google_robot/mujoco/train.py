import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from google_robot_env import GoogleRobotPickPlaceEnv

SCENE_PATH = "/Users/dhruvpatel29/mujoco/google_robot/google_robot/scene.xml"

# --- CONFIGURATION ---
NUM_ENVS = 48        
N_STEPS = 1024
# One iteration is exactly 61,440 steps (30 * 2048)
ITERATION_SIZE = NUM_ENVS * N_STEPS 
TOTAL_TIMESTEPS = 2_000_000  
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def make_env(rank, seed=0):
    def _init():
        env = GoogleRobotPickPlaceEnv(SCENE_PATH)
        return env
    set_random_seed(seed)
    return _init

class RenderCallback(BaseCallback):
    def __init__(self, eval_env: GoogleRobotPickPlaceEnv, render_freq_iterations=1):
        super().__init__()
        self.eval_env = eval_env
        # render_freq_iterations=1 means show it EVERY time the table prints
        self.render_freq_steps = render_freq_iterations * ITERATION_SIZE

    def _on_step(self) -> bool:
        # Trigger ONLY at the end of an iteration
        if self.num_timesteps % self.render_freq_steps == 0:
            print(f"\n--- [ITERATION FINISHED] Visualizing policy at step {self.num_timesteps} ---")
            
            # 1. Reset env and get initial obs
            obs, _ = self.eval_env.reset()
            
            # 2. IMPORTANT: Manually normalize the observation
            # The model was trained on normalized data, so we must scale the eval obs
            if isinstance(self.training_env, VecNormalize):
                self.eval_env.obs_rms = self.training_env.obs_rms

            for _ in range(400): # Run for roughly one full episode
                # Normalize the observation before passing to model
                if isinstance(self.training_env, VecNormalize):
                    norm_obs = self.training_env.normalize_obs(obs)
                else:
                    norm_obs = obs

                action, _ = self.model.predict(norm_obs, deterministic=True)
                
                # Step the environment
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                # Show the 3D window
                self.eval_env.render()
                
                if done or truncated: 
                    break
                    
        return True

if __name__ == "__main__":
    # 1. Start Training Workers
    print(f"Launching {NUM_ENVS} parallel robots...")
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Start Visual Environment
    eval_env = GoogleRobotPickPlaceEnv(SCENE_PATH)
    eval_env.render_mode = "human" 

    # 3. Setup Callback (1 = every iteration)
    render_callback = RenderCallback(eval_env, render_freq_iterations=1)

    # 4. Initialize PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device=DEVICE,
        n_steps=N_STEPS,
        batch_size=1024,
        policy_kwargs=dict(net_arch=[256, 256, 256])
    )

    # 5. Learn
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,callback=render_callback)
    except KeyboardInterrupt:
        print("Stopped by user.")

    model.save("google_robot_ppo_final")
    env.save("vec_normalize_final.pkl")