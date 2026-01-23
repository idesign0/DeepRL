import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from google_robot_env import GoogleRobotPickPlaceEnv

# --- CONFIGURATION ---
SCENE_PATH = "/Users/dhruvpatel29/mujoco/google_robot/google_robot/scene.xml"
MODEL_PATH = "google_robot_ppo_final.zip"
STATS_PATH = "vec_normalize_final.pkl"

def main():
    # 1. Create the environment
    # We don't use a lambda here so we can keep a reference to the base env
    base_env = GoogleRobotPickPlaceEnv(SCENE_PATH)
    env = DummyVecEnv([lambda: base_env])

    # 2. Load Normalizer
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False
    env.norm_reward = False

    # 3. Load Model (Force CPU for stability during viewing)
    model = PPO.load(MODEL_PATH, env=env, device="cpu")

    print("🚀 Model loaded! Attempting to open MuJoCo viewer...")

    obs = env.reset()
    
    # --- MUJOCO VIEWER INITIALIZATION ---
    import mujoco.viewer
    # This opens the interactive window using the underlying MuJoCo model/data
    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        while viewer.is_running():
            # Get action from AI
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, rewards, dones, infos = env.step(action)
            
            # Update the visual window
            viewer.sync()
            
            # Slow down to real-time (approx 60fps)
            time.sleep(0.01)

            if dones[0]:
                print("Resetting Episode...")
                obs = env.reset()

if __name__ == "__main__":
    main()