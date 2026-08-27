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

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.font_manager as _fm
import matplotlib.gridspec as gridspec
from pathlib import Path as _Path

for _f in ("/usr/share/fonts/truetype/cmu/cmunsx.ttf",
           "/usr/share/fonts/truetype/cmu/cmunss.ttf"):
    if _Path(_f).exists():
        _fm.fontManager.addfont(_f)

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["CMU Sans Serif", "DejaVu Sans", "Arial"],
    "font.size":          13,
    "axes.labelsize":     18,
    "axes.titlesize":     20,
    "axes.titleweight":   "bold",
    "axes.labelweight":   "normal",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.linewidth":     1,
    "legend.fontsize":    12,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "figure.dpi":         600,
    "grid.alpha":         0.12,
    "grid.linewidth":     0.3,
    "text.usetex":        False,
    "axes.unicode_minus": False,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
})
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

    fig = plt.figure(figsize=(9.0, 5.2))
    gs = gridspec.GridSpec(1, 1, figure=fig,
                           left=0.12, right=0.985, bottom=0.14, top=0.92)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    colors = {"RRT*": "#bf360c", "Informed RRT*": "#0d47a1"}
    for name, runs in results.items():
        finite = np.where(np.isfinite(runs), runs, np.nan)
        iters = np.arange(runs.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(finite, axis=0)
            lo = np.nanpercentile(finite, 25, axis=0)
            hi = np.nanpercentile(finite, 75, axis=0)
        solved = int(np.isfinite(runs[:, -1]).sum())
        ax.fill_between(iters, lo, hi, color=colors[name], alpha=0.16, linewidth=0)
        ax.plot(iters, median, color=colors[name], linewidth=2.2,
                label=f"{name}  ({solved}/{args.trials} solved)")

    ax.set_xlabel("iteration", fontsize=18)
    ax.set_ylabel("best path cost so far (rad)", fontsize=18)
    ax.set_title("Convergence on the 3R dense-obstacle scene", fontsize=20, fontweight="bold")
    ax.tick_params(labelsize=14)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    png_path = os.path.join(args.out_dir, "benchmark_convergence.png")
    pdf_path = os.path.join(args.out_dir, "benchmark_convergence.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
