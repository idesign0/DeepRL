import mujoco
import mujoco.viewer
import time

scene_path = "/Users/dhruvpatel29/mujoco/google_robot/google_robot/scene.xml"

try:
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    print("Scene loaded successfully.")
    
    # Let's see what the robot 'sees'
    print("Available Sites (for RL targets):", [model.site(i).name for i in range(model.nsite)])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Physics step
            mujoco.mj_step(model, data)
            
            # Use sync() instead of render() for passive viewer
            viewer.sync()

            # Real-time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
except Exception as e:
    print(f"Error: {e}")