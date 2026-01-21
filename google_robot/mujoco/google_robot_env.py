from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

MAX_EPISODE_STEPS = (1/2)*750  # e.g., 0.04s timestep * 750 ≈ 30s in sim

class GoogleRobotPickPlaceEnv(gym.Env):
    def __init__(self, model_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.step_counter = 0  # track steps in episode

        # Actions: 9 (Torso, Shoulder, Bicep, Elbow, Forearm, Wrist, Gripper, 2 Fingers)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        
        # Updated Observation Dimension:
        # qpos (varies) + qvel (varies) + gripper(3) + cube(3) + target(3) + 2 relative vectors(6)
        obs_dim = self.model.nq + self.model.nv + 3 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        # 1. Spatial positions
        # Using .copy() ensures we don't accidentally modify MuJoCo internal data
        gripper_pos = self.data.site_xpos[self.model.site("gripper").id].copy()
        cube_pos = self.data.site_xpos[self.model.site("cube_site").id].copy()
        target_pos = self.data.site_xpos[self.model.site("target_site").id].copy()
        
        # 2. Relative vectors (crucial for RL spatial awareness)
        dist_to_cube = cube_pos - gripper_pos
        dist_to_target = target_pos - cube_pos

        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flatten(),
            gripper_pos,
            cube_pos,
            target_pos,
            dist_to_cube,
            dist_to_target
        ]).astype(np.float32)

    def step(self, action):
        self.step_counter += 1

        # Physics step (unchanged)
        ctrl_range = self.model.actuator_ctrlrange
        scaled_action = ctrl_range[:, 0] + (action + 1.0) * 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self.data.ctrl[:] = scaled_action
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # 1. Extract state
        gripper_pos = self.data.site_xpos[self.model.site("gripper").id].copy()
        cube_pos = self.data.site_xpos[self.model.site("cube_site").id].copy()
        gripper_mat = self.data.site_xmat[self.model.site("gripper").id].reshape(3, 3)
        cube_mat = self.data.xmat[self.model.body("cube").id].reshape(3, 3)

        # 2. Distance and Orientation Errors (Decoupled)
        dist = np.linalg.norm(cube_pos - gripper_pos)
        
        # Gripper symmetry: A parallel gripper is the same if rotated 180 deg (Pi)
        mirror_yz = np.diag([1, 1, 1])
        target_mat = cube_mat @ mirror_yz
        
        r_err = R.from_matrix(target_mat.T @ gripper_mat)
        angle_err = np.linalg.norm(r_err.as_rotvec()) 

        # 3. Reward Components
        # Reach: Smooth exponential reward (0 to 1)
        reward_reach = np.exp(-5.0 * dist) 
        
        # Alignment
        reward_align = 0.0
        reward_contact = 0.0
        if dist < 1.0:
            reward_align = 0.5 * (np.exp(-angle_err))
            # Reward the robot for closing the gripper (action[6]) when close to the cube
            gripper_action = action[6]
            if gripper_action > 0:  # If the robot is attempting to close
                reward_contact = 0.2 * gripper_action

        # Grasping & Lifting
        reward_pick = 0.0
        has_contact = self._finger_cube_contact()
        cube_height = cube_pos[2]
        
        if has_contact:
            reward_pick += 2.0  # Bonus for touching
            if cube_height > 0.04:  # Bonus for lifting
                reward_pick += 10.0 + (50.0 * (cube_height - 0.02))

        # Velocity Penalty (to encourage smooth actions)
        velocity_penalty = -0.01 * np.linalg.norm(self.data.qvel)

        # Total Reward
        reward = reward_reach + reward_align + reward_contact + reward_pick + velocity_penalty

        # 4. Termination logic
        # Don't terminate just for being low; only if the robot falls over or fails critically
        terminated = False 
        # Optional: Terminate if cube falls off table
        if cube_pos[2] < -0.05 or np.linalg.norm(cube_pos[:2]) > 1.0:
            terminated = True
            reward -= 10.0

        truncated = self.step_counter >= MAX_EPISODE_STEPS

        return obs, reward, terminated, truncated, {}



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        mujoco.mj_resetData(self.model, self.data)

        # start index of the cube joint's qpos
        cube_qpos_adr = int(self.model.joint("cube_joint").qposadr)  # safer: ensure int

        # spawn coordinates (x, y, z)
        far_x = 0.4 + np.random.uniform(0, 0.2)   # X: 0.4 → 0.6
        random_y = -0.5 + np.random.uniform(0, 0.4)  # Y: -0.2 → 0.2
        z = 0.02

        # For a free joint we must set 7 values: tx, ty, tz, qw, qx, qy, qz
        # Identity rotation quaternion = (1, 0, 0, 0)
        self.data.qpos[cube_qpos_adr : cube_qpos_adr + 7] = np.array([far_x, random_y, z, 1.0, 0.0, 0.0, 0.0])

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _finger_cube_contact(self):
        """
        Returns True if either finger tip touches the cube.
        Works with the current XML (unnamed geoms).
        """
        # --- Step 1: Identify geom indices ---
        # Cube geom: the only geom in body 'cube'
        cube_body_id = self.model.body("cube").id
        cube_geom_idx = None
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == cube_body_id:
                cube_geom_idx = g
                break
        if cube_geom_idx is None:
            raise RuntimeError("Cannot find cube geom index")

        # Left finger tip geom: last geom in 'link_finger_tip_left' body
        left_tip_body_id = self.model.body("link_finger_tip_left").id
        left_finger_geom_idx = None
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == left_tip_body_id:
                left_finger_geom_idx = g
                break
        if left_finger_geom_idx is None:
            raise RuntimeError("Cannot find left finger tip geom index")

        # Right finger tip geom: last geom in 'link_finger_tip_right' body
        right_tip_body_id = self.model.body("link_finger_tip_right").id
        right_finger_geom_idx = None
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == right_tip_body_id:
                right_finger_geom_idx = g
                break
        if right_finger_geom_idx is None:
            raise RuntimeError("Cannot find right finger tip geom index")

        finger_geom_indices = {left_finger_geom_idx, right_finger_geom_idx}

        # --- Step 2: Check contacts ---
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if cube_geom_idx in (c.geom1, c.geom2) and \
            (c.geom1 in finger_geom_indices or c.geom2 in finger_geom_indices):
                return True

        return False