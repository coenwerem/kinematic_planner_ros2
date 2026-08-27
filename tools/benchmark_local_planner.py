#!/usr/bin/env python3
"""Run the kinematic_planner RRT* on the query used by the MoveIt comparison.

`benchmark_moveit_planners.py` measures MoveIt's pipelines through move_group.
This script solves the same start, goal, and obstacle set with the planner and
collision stack in `src/kinematic_planner`, so the two results can be read side
by side. It reuses the recorded query rather than restating it, and it merges
its own measurements into the same JSON file.

The two stacks build collision geometry independently, so the comparison is
between implementations and not a controlled ablation. The straight-line
diagnostic reported below is the cross-check: both stacks are expected to label
the same fraction of direct-interpolation waypoints as in collision.

Usage:
    python3 tools/benchmark_moveit_planners.py --out results/moveit_comparison.json
    python3 tools/benchmark_local_planner.py results/moveit_comparison.json
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive

from kinematic_planner.collision.robot_collision_model import build_link_collision_shapes
from kinematic_planner.planning.interpolate import interpolate_waypoints
from kinematic_planner.planning.rrt_star import RRTStar
from kinematic_planner.planning.tree import TreeNode
from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.scripts.planner_node import _build_rtb_model, build_collision_fn
from kinematic_planner_interfaces.msg import SceneObstacles

MIN_OBS_DIST = 0.02


def _node(q):
    node = TreeNode(np.asarray(q))
    node.path_q = [node.q]
    return node


def build_stack(obstacles, min_obs_dist=MIN_OBS_DIST):
    """Return (robot_config, collision_fn) for the recorded obstacle set.

    min_obs_dist is the clearance the predicate demands. The planner runs with a
    margin; the cross-check against move_group runs at zero so both stacks are
    asked the same question.
    """
    urdf_path = os.path.join(
        get_package_share_directory("xarm7_description"), "urdf", "xarm7.urdf"
    )
    urdf = open(urdf_path).read()
    robot_config = RobotConfig.from_urdf(urdf)
    link_shapes = build_link_collision_shapes(ET.fromstring(urdf))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as fh:
        fh.write(urdf)
        rtb_path = fh.name
    rtb_model = _build_rtb_model(rtb_path)
    os.remove(rtb_path)

    scene = SceneObstacles()
    boxes, poses, ids = [], [], []
    for i, entry in enumerate(obstacles):
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(v) for v in entry["size"]]
        pose = PoseStamped()
        pose.pose.position.x = float(entry["pos"][0])
        pose.pose.position.y = float(entry["pos"][1])
        pose.pose.position.z = float(entry["pos"][2])
        pose.pose.orientation.w = 1.0
        boxes.append(box)
        poses.append(pose)
        ids.append(i)
    scene.scene_obstacles = boxes
    scene.obstacle_poses = poses
    scene.obstacle_ids = ids

    collision_fn = build_collision_fn(
        robot_config=robot_config,
        link_shapes=link_shapes,
        obstacle_geom=scene,
        rtb_model=rtb_model,
        collision_checker="proximity",
        min_obs_dist=min_obs_dist,
        check_collision=True,
    )
    return robot_config, collision_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="JSON written by benchmark_moveit_planners.py")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--expand-dist", type=float, default=0.3)
    args = ap.parse_args()

    with open(args.results) as fh:
        data = json.load(fh)
    start, goal = data["start"], data["goal"]
    robot_config, collision_fn = build_stack(data["obstacles"])

    # The cross-check runs at zero clearance because move_group's diagnostic does.
    # A margin here would make the two stacks disagree by construction.
    _, bare_collision_fn = build_stack(data["obstacles"], min_obs_dist=0.0)
    samples = data["straight_line_samples"]
    line = interpolate_waypoints([np.array(start), np.array(goal)],
                                 steps_per_segment=samples - 1)
    blocked = sum(1 for q in line if not bare_collision_fn(_node(q)))
    agree = "agree" if blocked == data["straight_line_in_collision"] else "DIFFER"
    print(f"straight-line cross-check at zero clearance: {blocked}/{len(line)} in collision, "
          f"move_group {data['straight_line_in_collision']}/{samples} -> {agree}")
    if not collision_fn(_node(np.array(start))) or not collision_fn(_node(np.array(goal))):
        print("error: an endpoint is in collision under this collision stack", file=sys.stderr)
        return 1

    solved, times, costs, counts = 0, [], [], []
    kept = None
    for trial in range(args.trials):
        planner = RRTStar(
            start=start, goal=goal, joint_limits=robot_config.joint_limits,
            expand_dist=args.expand_dist, path_resolution=0.1,
            max_iter=args.max_iter, connect_circle_dist=25, goal_sample_rate=0.2,
            collision_fn=collision_fn, use_goal_biased_sampling=True,
            goal_noise_sigma=0.4, rng=np.random.default_rng(trial),
        )
        began = time.perf_counter()
        path = planner.plan()
        elapsed = time.perf_counter() - began
        if path is None:
            print(f"  trial {trial}: no solution in {elapsed:.1f} s")
            continue
        if not all(collision_fn(_node(np.array(q))) for q in path):
            print(f"  trial {trial}: returned path contains an invalid waypoint")
            continue
        if kept is None:
            kept = [[float(v) for v in q] for q in path]
        solved += 1
        times.append(elapsed)
        costs.append(planner.compute_path_cost(path))
        counts.append(len(path))
        print(f"  trial {trial}: {elapsed:.1f} s, cost {costs[-1]:.3f} rad, {len(path)} waypoints")

    row = {
        "planner": "RRT* (kinematic_planner)",
        "pipeline": "kinematic_planner",
        "successes": solved,
        "trials": args.trials,
        "plan_time_s": statistics.median(times) if solved else None,
        "path_length_rad": statistics.median(costs) if solved else None,
        "waypoints": int(statistics.median(counts)) if solved else None,
        "note": None if solved else f"no solution in {args.max_iter} iterations",
        "max_iter": args.max_iter,
        "implementation": "Python",
        "straight_line_in_collision": blocked,
        "trajectory": kept,
    }
    data["local_planner"] = row
    with open(args.results, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\n{row['planner']}: {solved}/{args.trials} solved"
          + (f", median {row['plan_time_s']:.1f} s, cost {row['path_length_rad']:.3f} rad"
             if solved else ""))
    print(f"merged into {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
