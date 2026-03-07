"""
Shared utilities for LIBERO spatial preview scripts.
"""

import numpy as np


def _hide_robosuite_visualizations(env):
    """
    Turn off robosuite's native visual aids (green line, red dot at gripper).
    Uses env.reset()'s visualize() when possible; falls back to direct site manipulation.
    """
    inner = env.env
    try:
        vis_settings = {vis: False for vis in inner._visualizations}
        inner.visualize(vis_settings=vis_settings)
    except (KeyError, TypeError, AttributeError):
        # Fallback: directly hide gripper sites on each robot (same as Manipulator._visualize_grippers)
        sim = inner.sim
        for robot in inner.robots:
            if hasattr(robot, "gripper") and robot.gripper is not None:
                robot.gripper.set_sites_visibility(sim=sim, visible=False)


def _hide_gripper_sites_directly(env):
    """
    Directly hide gripper visualization sites (green line, red dot).
    Uses robosuite's set_sites_visibility when possible, then brute-force hides
    any remaining sites matching gripper patterns (handles naming/prefix mismatches).
    """
    inner = env.env
    sim = inner.sim
    model = sim.model

    # Method 1: Use robosuite's set_sites_visibility
    for robot in inner.robots:
        if hasattr(robot, "gripper") and robot.gripper is not None:
            try:
                robot.gripper.set_sites_visibility(sim=sim, visible=False)
            except Exception:
                pass

    # Method 2: Brute-force - hide any site whose name suggests gripper visualization
    # grip_site = red dot, grip_site_cylinder/grip_cylinder = green line, ee/ee_x/ee_y/ee_z = axis markers
    GRIPPER_SITE_PATTERNS = (
        "grip_site",
        "grip_cylinder",
        "grip_site_cylinder",
        "ee_x",
        "ee_y",
        "ee_z",
    )
    for i in range(model.nsite):
        try:
            name = model.id2name(i, "site")
        except (AttributeError, TypeError):
            continue
        if name is None:
            continue
        name_lower = name.lower()
        if any(p in name_lower for p in GRIPPER_SITE_PATTERNS):
            model.site_size[i] = 0.0
            # Negative alpha = hidden (robosuite uses this convention)
            model.site_rgba[i, 3] = -1.0


def settle_physics(env, num_steps: int = 100):
    """
    Let the physics simulation run for a few seconds so objects settle (e.g. fall to the table).
    Runs env.step with zero action for num_steps. At 20Hz control freq, 100 steps ≈ 5 seconds.
    """
    zero_action = np.zeros(7, dtype=np.float64)  # xyz, rot_axangle, gripper
    for _ in range(num_steps):
        env.step(zero_action)


def hide_gripper_orientation_site(env):
    """
    Hide the green virtual line and red dot emanating from the gripper.
    Directly calls set_sites_visibility on each robot's gripper (bypasses visualize API
    which can fail or be skipped when using set_init_state instead of reset).
    """
    _hide_gripper_sites_directly(env)
