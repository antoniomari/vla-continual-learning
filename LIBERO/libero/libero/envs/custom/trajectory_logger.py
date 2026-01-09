import json
from pathlib import Path

import imageio
import numpy as np


class TrajectoryLoggerWrapper:
    """Wrapper that logs trajectories for any LIBERO environment"""

    def __init__(self, env, log_dir="./trajectory_logs", env_id=0):
        self.env = env
        self.log_dir = Path(log_dir) / f"env_{env_id}"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.current_traj_id = 0
        self.current_traj_data = []
        self.step_count = 0

    def __getattr__(self, name):
        """Delegate attribute access to wrapped env"""
        return getattr(self.env, name)

    def __del__(self):
        """Save trajectory when object is destroyed"""
        if self.current_traj_data:
            self._save_trajectory()

    def reset(self, **kwargs):
        # Save previous trajectory if it exists
        if self.current_traj_data:
            self._save_trajectory()

        # Start fresh
        self.current_traj_data = []
        self.step_count = 0

        obs = self.env.reset(**kwargs)

        # Log initial state
        state_info = self._get_state_info()
        self.current_traj_data.append(
            {
                "step": self.step_count,
                "action": None,
                "state": state_info,
                "obs": self._extract_obs_data(obs),
            }
        )

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        state_info = self._get_state_info()
        self.current_traj_data.append(
            {
                "step": self.step_count,
                "action": action.tolist() if isinstance(action, np.ndarray) else action,
                "state": state_info,
                "obs": self._extract_obs_data(obs),
                "reward": float(reward),
                "done": bool(done),
            }
        )

        return obs, reward, done, info

    def _get_state_info(self):
        sim = self.env.sim

        state_info = {
            "time": float(sim.data.time),
            "qpos": sim.data.qpos.copy().tolist(),
            "qvel": sim.data.qvel.copy().tolist(),
        }

        # EEF state
        try:
            eef_site_id = sim.model.site_name2id("grip_site")
            state_info["eef_pos"] = sim.data.site_xpos[eef_site_id].copy().tolist()
        except:
            pass

        # All object states
        objects = {}
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if name and not name.startswith("robot"):
                objects[name] = {
                    "pos": sim.data.body_xpos[i].copy().tolist(),
                    "quat": sim.data.body_xquat[i].copy().tolist(),
                }
        state_info["objects"] = objects

        return state_info

    def _extract_obs_data(self, obs):
        """Extract image data from observations"""
        obs_data = {}

        if "agentview_image" in obs:
            obs_data["agentview_image"] = obs["agentview_image"]

        if "robot0_eye_in_hand_image" in obs:
            obs_data["wrist_image"] = obs["robot0_eye_in_hand_image"]

        return obs_data

    def _save_trajectory(self):
        if not self.current_traj_data:
            return

        traj_dir = self.log_dir / f"traj_{self.current_traj_id}"
        traj_dir.mkdir(exist_ok=True, parents=True)

        # Save all images
        for step_data in self.current_traj_data:
            step = step_data["step"]
            obs_data = step_data.pop("obs")

            if "agentview_image" in obs_data:
                imageio.imwrite(
                    traj_dir / f"agentview_{step:04d}.png",
                    obs_data["agentview_image"],
                )

            if "wrist_image" in obs_data:
                imageio.imwrite(
                    traj_dir / f"wrist_{step:04d}.png",
                    obs_data["wrist_image"],
                )

        with open(traj_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "trajectory_id": self.current_traj_id,
                    "num_steps": len(self.current_traj_data),
                    "steps": self.current_traj_data,
                },
                f,
                indent=2,
            )

        self.current_traj_id += 1
