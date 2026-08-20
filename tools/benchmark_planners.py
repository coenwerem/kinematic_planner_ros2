#!/usr/bin/env python3
"""Benchmark RRT* against Informed RRT* on the bundled 3R demo scene.

Runs both planners over multiple seeded trials on the same start/goal/
obstacle scene tools/render_demo.py uses (the dense 3R scene), recording
best-path-cost-so-far at every iteration via the on_iteration hook, then
plots the mean convergence curve (cost vs. iteration) for each algorithm.

Requires the workspace built and sourced (colcon build, source
install/setup.bash).

Usage:
    python3 tools/benchmark_planners.py
    python3 tools/benchmark_planners.py --trials 20 --max-iter 800
"""
import argparse
import os
import subprocess
import sys

for _name in [n for n in sys.modules if n == "mpl_toolkits" or n.startswith("mpl_toolkits.")]:
    del sys.modules[_name]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

from kinematic_planner.collision.robot_collision_model import build_link_collision_shapes
from kinematic_planner.planning.informed_rrt_star import InformedRRTStar
from kinematic_planner.planning.rrt_star import RRTStar
from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.scripts.obstacle_publisher import default_obstacle_scene
from kinematic_planner.scripts.planner_node import _build_rtb_model, build_collision_fn
from kinematic_planner_interfaces.msg import SceneObstacles
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory

START = [0.0, 0.0, 0.0]
GOAL = [-1.5, 0.5, -0.9]
PLATFORM_HEIGHT = 0.755
IS_DENSE = True
MIN_OBS_DIST = 0.1
EXPAND_DIST = 0.3
PATH_RESOLUTION = 0.1
CONNECT_CIRCLE_DIST = 20
GOAL_SAMPLE_RATE = 0.3
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")


def load_urdf_string() -> str:
    urdf_xacro = os.path.join(
        get_package_share_directory("robot_3r_description"), "urdf", "robot_3r.urdf.xacro"
    )
    return subprocess.check_output(["xacro", urdf_xacro], text=True)


def obstacle_scene() -> SceneObstacles:
    positions, sizes = default_obstacle_scene(is_dense=IS_DENSE, platform_height=PLATFORM_HEIGHT)
    n = len(positions) // 3
    scene = SceneObstacles()
    boxes, poses, ids = [], [], []
    for i in range(n):
        pos = positions[3 * i:3 * i + 3]
        size = sizes[3 * i:3 * i + 3]
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos
        pose.pose.orientation.w = 1.0
        boxes.append(box)
        poses.append(pose)
        ids.append(i)
    scene.scene_obstacles = boxes
    scene.obstacle_poses = poses
    scene.obstacle_ids = ids
    return scene


def build_collision_fn_for_3r():
    urdf_str = load_urdf_string()
    robot_config = RobotConfig.from_urdf(urdf_str, base_link_name="base_link")
    link_shapes = build_link_collision_shapes(ET.fromstring(urdf_str))

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_str)
        rtb_path = f.name
    rtb_model = _build_rtb_model(rtb_path)
    os.remove(rtb_path)

    return robot_config, build_collision_fn(
        robot_config=robot_config, link_shapes=link_shapes, obstacle_geom=obstacle_scene(),
        rtb_model=rtb_model, collision_checker="proximity", min_obs_dist=MIN_OBS_DIST,
        check_collision=True,
    )


def run_trial(planner_cls, seed, max_iter, joint_limits, collision_fn):
    rng = np.random.default_rng(seed)
    kwargs = dict(
        start=START, goal=GOAL, joint_limits=joint_limits,
        expand_dist=EXPAND_DIST, path_resolution=PATH_RESOLUTION, max_iter=max_iter,
        connect_circle_dist=CONNECT_CIRCLE_DIST, collision_fn=collision_fn,
        search_until_max_iter=True, rng=rng,
    )
    if planner_cls is RRTStar:
        kwargs["goal_sample_rate"] = GOAL_SAMPLE_RATE
    planner = planner_cls(**kwargs)

    costs = np.full(max_iter, np.inf)

    def on_iteration(i, best_cost):
        costs[i] = best_cost

    planner.plan(on_iteration=on_iteration)
    return costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    robot_config, collision_fn = build_collision_fn_for_3r()

    results = {}
    for name, planner_cls in [("RRT*", RRTStar), ("Informed RRT*", InformedRRTStar)]:
        runs = []
        for trial in range(args.trials):
            costs = run_trial(planner_cls, seed=trial, max_iter=args.max_iter,
                               joint_limits=robot_config.joint_limits, collision_fn=collision_fn)
            runs.append(costs)
            found_at = np.argmax(np.isfinite(costs)) if np.any(np.isfinite(costs)) else -1
            print(f"{name} trial {trial}: first solution at iter "
                  f"{found_at if found_at >= 0 else 'never'}, final cost "
                  f"{costs[-1]:.3f}" if np.isfinite(costs[-1]) else f"{name} trial {trial}: no solution")
        runs = np.array(runs)
        results[name] = runs
        solved = np.isfinite(runs[:, -1]).sum()
        print(f"{name}: {solved}/{args.trials} trials solved by iteration {args.max_iter}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"RRT*": "#2f6fb0", "Informed RRT*": "#d9722c"}
    for name, runs in results.items():
        mean_cost = np.nanmean(np.where(np.isfinite(runs), runs, np.nan), axis=0)
        iters = np.arange(runs.shape[1])
        ax.plot(iters, mean_cost, label=name, color=colors[name], linewidth=2)
        for row in runs:
            ax.plot(iters, row, color=colors[name], alpha=0.08, linewidth=1)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best path cost so far (rad)")
    ax.set_title(f"RRT* vs Informed RRT* convergence ({args.trials} trials, 3R dense scene)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(args.out_dir, "benchmark_convergence.png")
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
