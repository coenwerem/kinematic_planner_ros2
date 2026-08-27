#!/usr/bin/env python3
"""Compare the MoveIt planning pipelines with the kinematic_planner RRT*.

tools/benchmark_planners.py compares the in-repo RRT* and Informed RRT*
implementations against each other. This script plans one blocked start-to-goal
problem on the xArm7 through MoveIt's OMPL, CHOMP, STOMP, and Pilz pipelines, so
the custom planners can be read against the ecosystem baselines under one robot
model, one planning scene, and one set of joint limits.

Two guards keep the comparison meaningful. Before planning, the straight-line
joint interpolation between the endpoints must itself be in collision, otherwise
the problem is trivial and the script refuses to report numbers. After planning,
every returned waypoint is re-checked against the planning scene, so a pipeline
only scores a success when its path is verifiably collision-free.

The script never executes a trajectory. It plans and inspects, so it is safe to
run without hardware.

Usage:
    ros2 launch xarm7_moveit_config move_group.launch.py     # terminal 1
    python3 tools/benchmark_moveit_planners.py               # terminal 2
"""

import argparse
import json
import statistics
import sys
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningScene,
    PlanningSceneComponents,
    RobotState,
    WorkspaceParameters,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetMotionPlan,
    GetPlanningScene,
    GetStateValidity,
)
from shape_msgs.msg import SolidPrimitive

JOINTS = [f"joint{i}" for i in range(1, 8)]
GROUP = "xarm7"
BASE_FRAME = "link_base"

# The dense scene from tools/render_xarm7_demo.py: six obstacles around the arm
# plus the mounting table it stands on. Each entry is a box given by its center
# position and full side lengths in the base frame.
OBSTACLES = [
    {"pos": (0.00, 0.00, -0.20), "size": (1.10, 1.10, 0.40)},   # mounting table
    {"pos": (0.35, 0.15, 0.20), "size": (0.08, 0.08, 0.40)},
    {"pos": (0.35, -0.15, 0.20), "size": (0.08, 0.08, 0.40)},
    {"pos": (0.50, 0.00, 0.15), "size": (0.08, 0.08, 0.30)},
    {"pos": (0.45, 0.28, 0.20), "size": (0.08, 0.08, 0.40)},
    {"pos": (0.45, -0.28, 0.20), "size": (0.08, 0.08, 0.40)},
    {"pos": (0.20, 0.30, 0.20), "size": (0.08, 0.08, 0.30)},
]

# Endpoints selected by sampling collision-free configurations and keeping the
# pair whose end effector crosses the obstacle field. The end effector travels
# 1.09 m, from (0.33, -0.53, 0.09) to (0.38, 0.54, 0.32), so a solution has to
# route around the obstacles rather than rotate in place.
START = [-0.967, 1.548, 0.493, 2.546, -1.747, 2.349, 1.417]
GOAL = [1.670, 0.885, -1.081, 1.630, -1.096, 0.300, 1.678]

# Pilz PTP is a deterministic industrial motion generator and runs no
# collision-aware search, so it is reported for contrast rather than as a
# competing search planner.
PLANNERS = [
    ("OMPL/RRTConnect", "ompl", "RRTConnectkConfigDefault"),
    ("OMPL/RRT*", "ompl", "RRTstarkConfigDefault"),
    ("OMPL/PRM", "ompl", "PRMkConfigDefault"),
    ("CHOMP", "chomp", ""),
    ("STOMP", "stomp", ""),
    ("Pilz/PTP", "pilz_industrial_motion_planner", "PTP"),
]


def path_length(trajectory):
    """C-space path length: summed joint-space distance between waypoints."""
    pts = trajectory.joint_trajectory.points
    return sum(
        sum((b - a) ** 2 for a, b in zip(p.positions, q.positions)) ** 0.5
        for p, q in zip(pts, pts[1:])
    )


class MoveItBenchmark(Node):
    def __init__(self, budget, trials):
        super().__init__("moveit_planner_benchmark")
        self.budget = budget
        self.trials = trials
        self.apply_scene = self._client(ApplyPlanningScene, "/apply_planning_scene")
        self.validity = self._client(GetStateValidity, "/check_state_validity")
        self.planner = self._client(GetMotionPlan, "/plan_kinematic_path")
        self.scene_query = self._client(GetPlanningScene, "/get_planning_scene")

    def _client(self, srv_type, name):
        cli = self.create_client(srv_type, name)
        if not cli.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(f"{name} unavailable; is move_group running?")
        return cli

    def _call(self, cli, request, timeout):
        fut = cli.call_async(request)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def allowed_collision_entries(self):
        comp = PlanningSceneComponents(
            components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        )
        res = self._call(self.scene_query, GetPlanningScene.Request(components=comp), 25.0)
        return len(res.scene.allowed_collision_matrix.entry_names) if res else 0

    def populate_scene(self):
        """Add the obstacles as a scene diff.

        The diff matters. Applying a scene with is_diff False overwrites the
        whole scene including the allowed-collision matrix built from the SRDF,
        after which every adjacent link pair reports a false collision.
        """
        scene = PlanningScene(is_diff=True)
        for i, box in enumerate(OBSTACLES):
            obj = CollisionObject(id=f"obstacle_{i}", operation=CollisionObject.ADD)
            obj.header.frame_id = BASE_FRAME
            obj.primitives = [
                SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(box["size"]))
            ]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = box["pos"]
            pose.orientation.w = 1.0
            obj.primitive_poses = [pose]
            scene.world.collision_objects.append(obj)
        self._call(self.apply_scene, ApplyPlanningScene.Request(scene=scene), 15.0)
        return len(scene.world.collision_objects)

    def state_valid(self, q):
        state = RobotState()
        state.joint_state.name = JOINTS
        state.joint_state.position = [float(v) for v in q]
        state.is_diff = False
        res = self._call(
            self.validity,
            GetStateValidity.Request(robot_state=state, group_name=GROUP),
            15.0,
        )
        return None if res is None else res.valid

    def straight_line_collisions(self, steps=25):
        blocked = 0
        for k in range(steps + 1):
            a = k / steps
            q = [(1 - a) * s + a * g for s, g in zip(START, GOAL)]
            if self.state_valid(q) is False:
                blocked += 1
        return blocked, steps + 1

    def build_request(self, pipeline, planner_id):
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.pipeline_id = pipeline
        req.planner_id = planner_id
        req.num_planning_attempts = 1
        req.allowed_planning_time = self.budget
        req.max_velocity_scaling_factor = 1.0
        req.max_acceleration_scaling_factor = 1.0
        bounds = WorkspaceParameters()
        bounds.header.frame_id = BASE_FRAME
        bounds.min_corner.x = bounds.min_corner.y = bounds.min_corner.z = -2.0
        bounds.max_corner.x = bounds.max_corner.y = bounds.max_corner.z = 2.0
        req.workspace_parameters = bounds
        req.start_state.joint_state.name = JOINTS
        req.start_state.joint_state.position = [float(v) for v in START]
        req.start_state.is_diff = False
        goal = Constraints()
        for name, value in zip(JOINTS, GOAL):
            goal.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=float(value),
                    tolerance_above=1e-3,
                    tolerance_below=1e-3,
                    weight=1.0,
                )
            )
        req.goal_constraints = [goal]
        return req

    def trajectory_invalid_waypoints(self, trajectory):
        names = trajectory.joint_trajectory.joint_names
        bad = 0
        for wp in trajectory.joint_trajectory.points:
            q = [wp.positions[names.index(j)] for j in JOINTS]
            if self.state_valid(q) is False:
                bad += 1
        return bad

    def run(self, straight_line):
        rows = []
        header = (
            f"{'planner':<18}{'success':<10}{'plan_time[s]':<15}"
            f"{'path_len[rad]':<16}{'waypoints'}"
        )
        print("\n" + header)
        print("-" * len(header))
        for label, pipeline, planner_id in PLANNERS:
            ok, times, lengths, counts, note = 0, [], [], [], None
            kept = None
            for _ in range(self.trials):
                started = time.perf_counter()
                res = self._call(
                    self.planner,
                    GetMotionPlan.Request(motion_plan_request=self.build_request(pipeline, planner_id)),
                    self.budget + 35.0,
                )
                elapsed = time.perf_counter() - started
                if res is None:
                    note = "no response"
                    continue
                code = res.motion_plan_response.error_code.val
                if code != 1:
                    note = f"error code {code}"
                    continue
                traj = res.motion_plan_response.trajectory
                if not traj.joint_trajectory.points:
                    note = "empty trajectory"
                    continue
                bad = self.trajectory_invalid_waypoints(traj)
                if bad:
                    note = f"{bad} invalid waypoints"
                    continue
                if kept is None:
                    names = traj.joint_trajectory.joint_names
                    kept = [[wp.positions[names.index(j)] for j in JOINTS]
                            for wp in traj.joint_trajectory.points]
                ok += 1
                times.append(elapsed)
                lengths.append(path_length(traj))
                counts.append(len(traj.joint_trajectory.points))
            rate = f"{ok}/{self.trials}"
            rows.append({
                "planner": label,
                "pipeline": pipeline,
                "successes": ok,
                "trials": self.trials,
                "plan_time_s": statistics.median(times) if ok else None,
                "path_length_rad": statistics.median(lengths) if ok else None,
                "waypoints": int(statistics.median(counts)) if ok else None,
                "note": None if ok else note,
                "trajectory": kept,
            })
            if ok:
                print(
                    f"{label:<18}{rate:<10}{statistics.median(times):<15.3f}"
                    f"{statistics.median(lengths):<16.3f}{int(statistics.median(counts))}"
                )
            else:
                print(f"{label:<18}{rate:<10}{'--':<15}{'--':<16}--   ({note})")
        print()
        return {
            "budget_s": self.budget,
            "trials": self.trials,
            "start": START,
            "goal": GOAL,
            "obstacles": OBSTACLES,
            "straight_line_in_collision": straight_line[0],
            "straight_line_samples": straight_line[1],
            "results": rows,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--budget", type=float, default=10.0,
                    help="allowed_planning_time per request, seconds")
    ap.add_argument("--out", default=None,
                    help="write results to this JSON file for plotting")
    args = ap.parse_args()

    rclpy.init()
    try:
        node = MoveItBenchmark(args.budget, args.trials)
        entries = node.allowed_collision_entries()
        if entries == 0:
            print("error: allowed-collision matrix is empty; move_group lost its SRDF "
                  "semantics (a scene applied with is_diff False will do this)",
                  file=sys.stderr)
            return 1
        print(f"allowed-collision matrix: {entries} entries")
        print(f"planning scene: {node.populate_scene()} obstacles")
        blocked, total = node.straight_line_collisions()
        print(f"straight-line interpolation: {blocked}/{total} waypoints in collision")
        if blocked == 0:
            print("error: the direct path is collision-free, so this is not a "
                  "planning problem; adjust START/GOAL or OBSTACLES", file=sys.stderr)
            return 2
        summary = node.run((blocked, total))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"results written to {args.out}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
