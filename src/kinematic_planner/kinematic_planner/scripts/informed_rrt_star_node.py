#!/usr/bin/env python3

"""
ROS 2 node: collision-free trajectory planning via Informed RRT*.

Informed RRT* focuses sampling inside an admissible ellipsoidal heuristic once a first
solution is found, yielding faster path-cost convergence than plain RRT*.

References:
    [1] Gammell, Srinivasa & Barfoot, "Informed RRT*: Optimal Sampling-based Path Planning
        Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic," IROS 2014.
    [2] https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/
           InformedRRTStar/informed_rrt_star.py

Author: Clinton Enwerem
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Point
from rclpy.duration import Duration
import numpy as np
import math
import random
import tempfile
import os

from robot_3r_interfaces.msg import JointWaypoint, JointSpacePath
from robot_3r_interfaces.msg import SceneObstacles, RigidBodyGeom
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray
from rcl_interfaces.msg import ParameterDescriptor
import transforms3d as t3d
import spatialmath as sm
from typing import List, Dict
from numpy.typing import NDArray

import fcl
from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.collision.collision_utils import obstacle_to_fclobj, create_fcl_object, se3_to_pose_stamped

try:
    from roboticstoolbox.robot.ERobot import ERobot
    _RTB_AVAILABLE = True
except ImportError:
    _RTB_AVAILABLE = False


def _build_rtb_model(urdf_path: str):
    class _RobotModel(ERobot):
        def __init__(self, path):
            links, name, urdf_string, urdf_fp = super().URDF_read(path)
            super().__init__(links, name=name.upper(), manufacturer="Custom",
                             urdf_string=urdf_string, urdf_filepath=urdf_fp)
    return _RobotModel(urdf_path)


class InformedRRTStarPlanner(Node):
    """ROS2 node wrapper around the Informed RRT* planner."""

    class InformedRRTStar:
        """
        Informed RRT* planning in joint space.

        After a first solution is found, sampling is focused inside the smallest ellipsoid
        in C-space that can contain any path with the same or lower cost, accelerating
        convergence to the optimum.
        """

        class InformedRRTStarConfigNode:
            def __init__(self, q):
                self.q = np.array(q, dtype=float)
                self.parent = None
                self.cost = 0.0
                self.path_q = []

        def __init__(self, outer_instance: Node, start, goal, robot_geom, expand_dist,
                     obstacle_geom, rand_area, path_resolution, max_iter,
                     connect_circle_dist, goal_sample_rate, check_collision_param=True):
            self.outer = outer_instance
            self.start = self.InformedRRTStarConfigNode(start)
            self.start.path_q = [start]
            self.end = self.InformedRRTStarConfigNode(goal)
            self.end.path_q = [goal]
            self.dimension = len(start)
            self.robot_geom = robot_geom
            self.min_rand = rand_area[0]
            self.max_rand = rand_area[1]
            self.expand_dist = expand_dist
            self.path_resolution = path_resolution
            self.goal_sample_rate = goal_sample_rate
            self.max_iter = max_iter
            self.obstacle_geom = obstacle_geom
            self.connect_circle_dist = connect_circle_dist
            self.config_tree = []
            self.check_collision_param = check_collision_param
            self.rewire_total = 0
            # informed sampling state
            self.c_best = float("inf")
            self.q_center = None
            self.R_align = None
            self.b = None
            self.c_min = None
            self.L_scale = np.zeros((self.dimension, self.dimension))
            self.q_ellipse = None
            self.q_unit = None

        # ---- sampling helpers -------------------------------------------

        def sample_free(self):
            """Uniform random sampling from joint limits (fallback sampler)."""
            try:
                samp_q = [np.random.uniform(lo, hi) for lo, hi in self.outer.joint_limits]
                return self.InformedRRTStarConfigNode(np.array(samp_q))
            except Exception as e:
                self.outer.get_logger().error(f"sample_free error: {e}")

        def calc_dist_to_goal(self, q):
            return np.linalg.norm(np.array(q) - np.array(self.end.q))

        def compute_ellipse_params(self):
            """Compute c_min and semi-minor axis b from the current best cost."""
            try:
                self.c_min = float(np.linalg.norm(
                    np.array(self.start.q, dtype=float) - np.array(self.end.q, dtype=float)))
            except Exception:
                self.c_min = float("inf")
            if self.c_best < float("inf"):
                val = self.c_best ** 2 - self.c_min ** 2
                self.b = float(np.sqrt(max(val, 0.0))) / 2.0
                if not np.isfinite(self.b) or self.b <= 0.0:
                    self.b = 1e-6
            else:
                self.b = None

        def compute_ellipse_center(self):
            center = (np.array(self.start.q, float) + np.array(self.end.q, float)) / 2.0
            return center.reshape((-1, 1))

        def compute_rotation_matrix(self):
            """SVD-based rotation matrix aligning the first axis with start→goal direction."""
            try:
                diff = np.array(self.end.q, float) - np.array(self.start.q, float)
                norm_diff = np.linalg.norm(diff)
                if norm_diff == 0.0:
                    return np.eye(self.dimension)
                a1 = (diff / norm_diff).reshape((-1, 1))
                e1 = np.zeros((self.dimension, 1), float)
                e1[0, 0] = 1.0
                U, _, Vt = np.linalg.svd(a1 @ e1.T, full_matrices=True)
                S = np.eye(self.dimension)
                S[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt)
                self.R_align = U @ S @ Vt
                return self.R_align
            except Exception:
                return np.eye(self.dimension)

        def sample_unit_hypersphere(self):
            """Uniformly sample from the unit n-ball."""
            n = self.dimension
            x = np.random.normal(size=(n,))
            norm = np.linalg.norm(x)
            if norm == 0.0:
                x = np.zeros(n)
                x[0] = 1.0
                norm = 1.0
            x = x / norm
            r = np.random.rand() ** (1.0 / n)
            return (x * r).reshape((n, 1))

        def informed_sample(self):
            """
            Sample from the ellipsoidal informed distribution.
            Falls back to uniform sampling when no solution exists yet.

            The ellipsoid is defined by:
                q_ellipse = R_align @ L_scale @ q_unit + q_center
            where L_scale = diag(c_best/2, b, b, ..., b).
            """
            if self.c_best == float("inf"):
                return np.array([np.random.uniform(lo, hi) for lo, hi in self.outer.joint_limits])

            self.compute_ellipse_params()
            if self.c_min is None or self.c_min == float("inf"):
                return np.array([np.random.uniform(lo, hi) for lo, hi in self.outer.joint_limits])

            self.q_center = self.compute_ellipse_center()
            self.R_align = self.compute_rotation_matrix()
            semi_minor = self.b if self.b is not None else float(max(self.c_best ** 2 - self.c_min ** 2, 0.0) ** 0.5) / 2.0
            L_vec = np.zeros(self.dimension)
            L_vec[0] = self.c_best / 2.0
            L_vec[1:] = semi_minor
            self.L_scale = np.diag(L_vec)

            low = np.array([lo for lo, _ in self.outer.joint_limits])
            high = np.array([hi for _, hi in self.outer.joint_limits])
            rnd = None
            for _ in range(10):
                self.q_unit = self.sample_unit_hypersphere()
                # equation (5): q_ellipse = R_align @ L_scale @ q_unit + q_center
                self.q_ellipse = (self.R_align @ self.L_scale @ self.q_unit) + self.q_center
                rnd = self.q_ellipse.flatten()
                if np.all(rnd >= low) and np.all(rnd <= high):
                    return rnd
            return np.clip(rnd, low, high)

        # ---- tree operations --------------------------------------------

        def get_nearby_neighbors(self, x_new) -> List[int]:
            assert self.connect_circle_dist > 2 * (1 + 1 / self.dimension) ** (1 / self.dimension)
            n = len(self.config_tree)
            near_radius = self.expand_dist if n <= 1 else min(
                self.connect_circle_dist * (math.log(n) / n) ** (1.0 / self.dimension),
                self.expand_dist
            )
            dists = [np.sum((nd.q - x_new.q) ** 2) for nd in self.config_tree]
            return [i for i, d in enumerate(dists) if d <= near_radius ** 2]

        def choose_best_parent(self, new_node, near_inds):
            if not near_inds:
                nearest_idx = self.get_nearest_node_index(self.config_tree, new_node)
                cand = self.steer(self.config_tree[nearest_idx], new_node)
                if cand and self.collision_free(cand, self.robot_geom, self.obstacle_geom, self.check_collision_param):
                    cand.parent = self.config_tree[nearest_idx]
                    cand.cost = self.config_tree[nearest_idx].cost + self.calc_new_cost(self.config_tree[nearest_idx], cand)
                    return cand
                return None
            costs = []
            for i in near_inds:
                cand = self.steer(self.config_tree[i], new_node)
                if cand and self.collision_free(cand, self.robot_geom, self.obstacle_geom, self.check_collision_param):
                    costs.append(self.calc_new_cost(self.config_tree[i], cand))
                else:
                    costs.append(float("inf"))
            min_cost = min(costs)
            if min_cost == float("inf"):
                return None
            min_ind = near_inds[costs.index(min_cost)]
            new_node = self.steer(self.config_tree[min_ind], new_node)
            if not new_node:
                return None
            new_node.parent = self.config_tree[min_ind]
            new_node.cost = min_cost
            return new_node

        def steer(self, x_nearest, x_random):
            start = np.array(x_nearest.q, float)
            goal = np.array(x_random.q, float)
            new_node = self.InformedRRTStarConfigNode(start.copy())
            d, _, _ = self.calc_distance_and_angle(x_nearest, x_random)
            new_node.path_q = [start]
            if d == 0.0:
                return None
            extend_length = min(self.expand_dist, d)
            n_expand = max(1, math.floor(extend_length / self.path_resolution))
            unit_vec = (goal - start) / np.linalg.norm(goal - start)
            for _ in range(n_expand):
                new_node.q = new_node.q + unit_vec * self.path_resolution
                new_node.path_q.append(new_node.q.copy())
            if self.calc_distance_and_angle(new_node, x_random)[0] <= self.path_resolution:
                new_node.q = goal.copy()
                new_node.path_q.append(goal)
            new_node.parent = x_nearest
            return new_node

        def rewire(self, new_node, near_inds):
            for i in near_inds:
                near_node = self.config_tree[i]
                cand = self.steer(new_node, near_node)
                if not cand:
                    continue
                if not self.collision_free(cand, self.robot_geom, self.obstacle_geom, self.check_collision_param):
                    continue
                new_cost = new_node.cost + self.calc_new_cost(new_node, near_node)
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.rewire_total += 1
                    self.propagate_cost_to_leaves(near_node)

        def find_best_goal_node(self):
            dists = [self.calc_dist_to_goal(nd.q) for nd in self.config_tree]
            goal_inds = [dists.index(i) for i in dists if i <= self.expand_dist]
            cf_inds = [gi for gi in goal_inds
                       if (c := self.steer(self.config_tree[gi], self.end)) and
                       self.collision_free(c, self.robot_geom, self.obstacle_geom, self.check_collision_param)]
            if not cf_inds:
                return None
            min_cost = min(self.config_tree[i].cost for i in cf_inds)
            return next(i for i in cf_inds if self.config_tree[i].cost == min_cost)

        def propagate_cost_to_leaves(self, parent_node):
            for node in self.config_tree:
                if node.parent == parent_node:
                    node.cost = self.calc_new_cost(parent_node, node)
                    self.propagate_cost_to_leaves(node)

        def calc_new_cost(self, from_node, to_node):
            d, _, _ = self.calc_distance_and_angle(from_node, to_node)
            return from_node.cost + d

        def generate_final_course(self, goal_ind):
            path = [self.end.q]
            node = self.config_tree[goal_ind]
            while node.parent is not None:
                path.append(node.q)
                node = node.parent
            path.append(node.q)
            path.reverse()
            return path

        def plan(self):
            start_free = self.collision_free(self.start, self.robot_geom, self.obstacle_geom, self.check_collision_param)
            goal_node = self.InformedRRTStarConfigNode(self.outer.goal_config)
            goal_node.path_q = [self.outer.goal_config]
            end_free = self.collision_free(goal_node, self.robot_geom, self.obstacle_geom, self.check_collision_param)
            if start_free is not None and end_free is not None:
                if not start_free or not end_free:
                    if not start_free:
                        self.outer.get_logger().warning("Start configuration is in collision!")
                        self.outer.start_goal_collision = "start"
                    if not end_free:
                        self.outer.get_logger().warning("Goal configuration is in collision!")
                        self.outer.start_goal_collision = "goal"
                    return None

            best_path = None
            self.config_tree = [self.start]
            informed_sample_count = 0
            c_best_updates = 0

            for i in range(self.max_iter):
                if self.outer.verbose:
                    self.outer.get_logger().info(f"Iter {i}, tree: {len(self.config_tree)}")

                informed_active = self.c_best < float("inf")
                rnd_arr = self.informed_sample()
                if informed_active:
                    informed_sample_count += 1

                rnd_node = self.InformedRRTStarConfigNode(rnd_arr)
                nearest_ind = self.get_nearest_node_index(self.config_tree, rnd_node)
                new_node = self.steer(self.config_tree[nearest_ind], rnd_node)

                collision_ok = new_node is not None and self.collision_free(
                    new_node, self.robot_geom, self.obstacle_geom, self.check_collision_param)

                if new_node and collision_ok:
                    near_inds = self.get_nearby_neighbors(new_node)
                    new_node = self.choose_best_parent(new_node, near_inds)
                    if new_node:
                        self.config_tree.append(new_node)
                        self.rewire(new_node, near_inds)

                        last_index = self.find_best_goal_node()
                        if last_index is not None:
                            temp_path = self.generate_final_course(last_index)
                            temp_cost = self.compute_path_cost(temp_path)
                            if temp_cost < self.c_best:
                                if self.c_best == float("inf"):
                                    self.compute_ellipse_params()
                                    self.outer.get_logger().info(
                                        f"First solution! c_min={self.c_min:.3f}, b={self.b}")
                                else:
                                    self.outer.get_logger().info(
                                        f"Improved: {self.c_best:.3f} → {temp_cost:.3f}")
                                self.c_best = temp_cost
                                best_path = temp_path
                                self.compute_ellipse_params()
                                c_best_updates += 1

                if not self.outer.rrts_search_until_max_iter and best_path is not None:
                    break

            if self.outer.verbose:
                self.outer.get_logger().info(
                    f"Planning done. c_best={self.c_best:.3f}, "
                    f"informed_samples={informed_sample_count}, "
                    f"c_best_updates={c_best_updates}")
            return best_path

        def collision_free(self, candidate_node, robot_geom: RigidBodyGeom,
                           obstacles: SceneObstacles, check_collision: bool = True) -> bool:
            if not check_collision:
                return False
            try:
                nodes_to_check = [self.InformedRRTStarConfigNode(n) for n in candidate_node.path_q]
            except Exception as e:
                self.outer.get_logger().error(f"collision check setup error: {e}")
                return False

            obs_fcl_objects = list(obstacle_to_fclobj(obstacles=obstacles))
            rtb_model = self.outer.rtb_model
            collision_detected = False

            for c_node in nodes_to_check:
                q = c_node.q
                rtb_model.q = q
                for link in rtb_model.links:
                    link_name = link.name
                    if link_name in [self.outer.robot_config.base_link_name, "link0",
                                     self.outer.robot_config.world_frame]:
                        continue
                    try:
                        idx = robot_geom.link_names.index(link_name)
                    except ValueError:
                        continue
                    try:
                        T_fk = rtb_model.fkine(q, end=link.name, include_base=True)
                        lg = robot_geom.link_geometries[idx]
                        if lg.type == SolidPrimitive.BOX:
                            geom = fcl.Box(*lg.dimensions)
                        elif lg.type == SolidPrimitive.SPHERE:
                            geom = fcl.Sphere(lg.dimensions[0])
                        elif lg.type == SolidPrimitive.CYLINDER:
                            geom = fcl.Cylinder(lg.dimensions[0], lg.dimensions[1])
                        else:
                            continue
                        rob_obj = create_fcl_object(
                            se3_to_pose_stamped(T_fk, self.outer,
                                                frame_id=self.outer.robot_config.base_link_name),
                            geom)
                        for obs_obj in obs_fcl_objects:
                            if self.outer.collision_checker == "bvol":
                                creq = fcl.CollisionRequest()
                                creq.enable_contact = True
                                cres = fcl.CollisionResult()
                                ret = fcl.collide(rob_obj, obs_obj, creq, cres)
                                if cres.is_collision or ret > 0:
                                    collision_detected = True
                                    break
                            if self.outer.collision_checker == "proximity":
                                dreq = fcl.DistanceRequest(enable_signed_distance=True)
                                dres = fcl.DistanceResult()
                                fcl.distance(rob_obj, obs_obj, dreq, dres)
                                min_dist = dres.min_distance
                                if min_dist < 0:
                                    self.outer.proximity_alert = True
                                    collision_detected = True
                                    break
                                if min_dist < self.outer.min_obs_dist:
                                    self.outer.proximity_alert = True
                                    collision_detected = True
                                    break
                        if collision_detected:
                            break
                    except Exception as e:
                        self.outer.get_logger().error(f"Error checking {link_name}: {e}")
                        collision_detected = True
                        break
                if collision_detected:
                    break
            return not collision_detected

        @staticmethod
        def get_nearest_node_index(config_tree, rnd_node):
            dlist = [np.sum((nd.q - rnd_node.q) ** 2) for nd in config_tree]
            return dlist.index(min(dlist))

        @staticmethod
        def calc_distance_and_angle(from_node, to_node):
            diff = to_node.q - from_node.q
            d = np.linalg.norm(diff)
            dx, dy = diff[0], diff[1] if len(diff) > 1 else 0.0
            dz = diff[2] if len(diff) > 2 else 0.0
            return d, math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))

        @staticmethod
        def compute_path_cost(path: List[NDArray]) -> float:
            if len(path) < 2:
                return 0.0
            arr = np.array(path)
            return float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))

    # ------------------------------------------------------------------
    # Node __init__
    # ------------------------------------------------------------------
    def __init__(self, node_name: str = "informed_rrt_star", queue_size=10):
        self.node_name = node_name
        self.queue_size = queue_size
        super().__init__(self.node_name)
        self.get_logger().info(
            f"\n--------------------------------------------------\n"
            f"Initializing {self.node_name} node...\n"
            f"--------------------------------------------------"
        )

        self.declare_parameter("planning_algorithm", "rrt_star")
        self.declare_parameter("stop_if_plan_found", False)
        self.declare_parameter("verbose", False)
        self.declare_parameter("rrts_expand_dist", 0.3)
        self.declare_parameter("rrts_path_resolution", 0.1)
        self.declare_parameter("rrts_max_iter", 300)
        self.declare_parameter("rrts_connect_circle_dist", 20)
        self.declare_parameter("rrts_search_until_max_iter", True)
        self.declare_parameter("rrts_goal_sample_rate", 0.3)
        self.declare_parameter("use_goal_biased_sampling", False)
        self.declare_parameter("goal_noise_sigma", 0.05)
        self.declare_parameter("min_obs_dist", 0.1)
        self.declare_parameter("collision_checker", "proximity")
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("show_jsp_waypoints", True)
        self.declare_parameter("show_ee_path", False)
        self.declare_parameter("check_collision", True)
        self.declare_parameter("print_metrics", True)
        self.declare_parameter("max_planning_attempts", 1)
        self.declare_parameter("stop_on_failure", True)
        self.declare_parameter("proximity_alert", False)
        self.declare_parameter("start_goal_collision", "")
        self.declare_parameter("disabled_collision_pairs", [""])
        self.declare_parameter("world_frame", "world")
        if not self.has_parameter("robot_description"):
            self.declare_parameter("robot_description", "")

        p = self.get_parameter
        self.verbose = p("verbose").get_parameter_value().bool_value
        self.print_metrics = p("print_metrics").get_parameter_value().bool_value
        self.pl_alg = p("planning_algorithm").get_parameter_value().string_value
        self.rrts_expand_dist = p("rrts_expand_dist").get_parameter_value().double_value
        self.rrts_path_resolution = p("rrts_path_resolution").get_parameter_value().double_value
        self.rrts_max_iter = p("rrts_max_iter").get_parameter_value().integer_value
        self.rrts_connect_circle_dist = p("rrts_connect_circle_dist").get_parameter_value().integer_value
        self.rrts_goal_sample_rate = p("rrts_goal_sample_rate").get_parameter_value().double_value
        self.rrts_search_until_max_iter = p("rrts_search_until_max_iter").get_parameter_value().bool_value
        self.show_jsp_waypoints = p("show_jsp_waypoints").get_parameter_value().bool_value
        self.show_ee_path = p("show_ee_path").get_parameter_value().bool_value
        self.check_collision = p("check_collision").get_parameter_value().bool_value
        self.stop_if_plan_found = p("stop_if_plan_found").get_parameter_value().bool_value
        self.min_obs_dist = p("min_obs_dist").get_parameter_value().double_value
        self.collision_checker = p("collision_checker").get_parameter_value().string_value
        self.max_planning_attempts = p("max_planning_attempts").get_parameter_value().integer_value
        self.stop_on_failure = p("stop_on_failure").get_parameter_value().bool_value
        self.proximity_alert = p("proximity_alert").get_parameter_value().bool_value
        self.start_goal_collision = p("start_goal_collision").get_parameter_value().string_value
        self.use_goal_biased_sampling = p("use_goal_biased_sampling").get_parameter_value().bool_value
        self.goal_noise_sigma = p("goal_noise_sigma").get_parameter_value().double_value
        self.random_seed = p("random_seed").get_parameter_value().integer_value
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # ---- load robot config from URDF --------------------------------
        urdf_str = p("robot_description").get_parameter_value().string_value
        if not urdf_str:
            raise RuntimeError("robot_description parameter is empty!")

        raw_disabled = p("disabled_collision_pairs").get_parameter_value().string_array_value
        disabled_pairs = [
            tuple(s.split(":", 1)) for s in raw_disabled if ":" in s
        ]
        world_frame = p("world_frame").get_parameter_value().string_value
        self.robot_config = RobotConfig.from_urdf(urdf_str, disabled_pairs=disabled_pairs,
                                                   world_frame=world_frame)
        self.joint_limits = self.robot_config.joint_limits

        # ---- RTB model --------------------------------------------------
        if not _RTB_AVAILABLE:
            raise RuntimeError("roboticstoolbox-python is required.")
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w") as f:
            f.write(urdf_str)
            tmp_urdf = f.name
        self.rtb_model = _build_rtb_model(tmp_urdf)
        os.unlink(tmp_urdf)

        midpoint = [(lo + hi) / 2.0 for lo, hi in self.joint_limits]
        self.declare_parameter("goal_config", midpoint)
        self.goal_config = p("goal_config").get_parameter_value().double_array_value

        # ---- planning state ---------------------------------------------
        self.planning_done = False
        self.planning_attempts = 0
        self.planning_failed = False
        self.start_config = None
        self.obstacle_geom = None
        self.robot_geom = None
        self.rand_area = [(lo, hi) for lo, hi in self.joint_limits]
        self.planner = None
        self.computed_path = None

        # ---- subs/pubs --------------------------------------------------
        self.create_subscription(JointState, "/joint_states", self.compute_plan, queue_size)
        self.create_subscription(SceneObstacles, "/scene_obstacles", self.get_scene_obs_cb, queue_size)
        self.create_subscription(RigidBodyGeom, "robot_geometry", self.robot_geom_cb, queue_size)
        self.plan_pub = self.create_publisher(JointSpacePath, "informed_rrts/jsp_path", queue_size)
        self.ee_path_pub = self.create_publisher(PoseStamped, "informed_rrts/ee_path", queue_size)
        self.jsp_path_marker_pub = self.create_publisher(MarkerArray, "informed_rrts/path_markers", 10)
        if self.stop_if_plan_found:
            self.create_timer(1.0, self.stop_node_cb)
        if self.show_jsp_waypoints:
            self.create_timer(1.0, self.publish_jsp_path_markers_cb)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def stop_node_cb(self):
        if self.planning_done:
            return
        if self.planning_failed and self.stop_on_failure:
            raise SystemExit

    def robot_geom_cb(self, msg: RigidBodyGeom):
        if self.robot_geom is None:
            self.robot_geom = msg

    def get_scene_obs_cb(self, msg: SceneObstacles):
        self.obstacle_geom = msg

    def compute_plan(self, msg: JointState):
        if self.planning_done or self.planning_failed:
            return
        self.start_config = msg.position
        if self.start_config is None or self.obstacle_geom is None or self.robot_geom is None:
            return

        self.planner = self.InformedRRTStar(
            self,
            start=self.start_config,
            goal=self.goal_config,
            robot_geom=self.robot_geom,
            obstacle_geom=self.obstacle_geom,
            rand_area=self.rand_area,
            expand_dist=self.rrts_expand_dist,
            path_resolution=self.rrts_path_resolution,
            max_iter=self.rrts_max_iter,
            connect_circle_dist=self.rrts_connect_circle_dist,
            goal_sample_rate=self.rrts_goal_sample_rate,
            check_collision_param=self.check_collision,
        )

        self.planning_attempts += 1
        self.get_logger().info(f"Informed RRT* attempt {self.planning_attempts}/{self.max_planning_attempts}")
        computed_path = self.planner.plan()

        if computed_path is None:
            self.get_logger().warning("No path found.")
            if self.planning_attempts >= self.max_planning_attempts:
                self.planning_failed = True
                if self.stop_on_failure:
                    self.create_timer(1.0, lambda: rclpy.shutdown())
        else:
            self.get_logger().info(f"Path found on attempt {self.planning_attempts}!")
            plan_msg = JointSpacePath()
            plan_msg.joint_names = list(self.robot_config.joint_names)
            for q in computed_path:
                wp = JointWaypoint()
                self.get_logger().info(f"\033[92mWaypoint: {np.round(q, 2)}\033[0m")
                wp.positions = np.array(q).tolist()
                plan_msg.waypoints.append(wp)
            if self.print_metrics:
                cost = self.InformedRRTStar.compute_path_cost(computed_path)
                self.get_logger().info(f"Path cost: {cost:.3f} rad")
            self.plan_pub.publish(plan_msg)
            self.computed_path = computed_path
            if self.stop_if_plan_found:
                self.planning_done = True

    def publish_jsp_path_markers_cb(self):
        if self.computed_path is not None and self.show_jsp_waypoints:
            self._publish_markers(self.computed_path)

    def _publish_markers(self, path):
        if self.robot_geom is None:
            return
        markers = MarkerArray()
        n = len(path)
        for idx, q in enumerate(path):
            rtb_poses: Dict[str, PoseStamped] = {}
            try:
                for link in self.rtb_model.links:
                    T = self.rtb_model.fkine(q, end=link.name, include_base=True)
                    rtb_poses[link.name] = se3_to_pose_stamped(
                        T, self, frame_id=self.robot_config.base_link_name)
            except Exception:
                continue
            for i, link_name in enumerate(self.robot_geom.link_names):
                if link_name not in rtb_poses:
                    continue
                try:
                    lg = self.robot_geom.link_geometries[i]
                    lg_orig = self.robot_geom.link_geom_origins[i]
                except IndexError:
                    continue
                lp = rtb_poses[link_name].pose
                link_T = sm.SE3(lp.position.x, lp.position.y, lp.position.z) * sm.SE3.RPY(
                    *t3d.euler.quat2euler([lp.orientation.w, lp.orientation.x,
                                          lp.orientation.y, lp.orientation.z], axes="sxyz"))
                orig_T = sm.SE3(lg_orig.position.x, lg_orig.position.y, lg_orig.position.z) * sm.SE3.RPY(
                    *t3d.euler.quat2euler([lg_orig.orientation.w, lg_orig.orientation.x,
                                          lg_orig.orientation.y, lg_orig.orientation.z], axes="sxyz"))
                ps = se3_to_pose_stamped(link_T * orig_T, self,
                                         frame_id=self.robot_config.base_link_name)
                m = Marker()
                m.header.frame_id = self.robot_config.base_link_name
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = f"informed_wp_{idx}"
                m.id = i
                if lg.type == SolidPrimitive.BOX:
                    m.type = Marker.CUBE
                    m.scale.x, m.scale.y, m.scale.z = lg.dimensions
                elif lg.type == SolidPrimitive.SPHERE:
                    m.type = Marker.SPHERE
                    m.scale.x = m.scale.y = m.scale.z = 2 * lg.dimensions[0]
                elif lg.type == SolidPrimitive.CYLINDER:
                    m.type = Marker.CYLINDER
                    m.scale.x = m.scale.y = 2 * lg.dimensions[0]
                    m.scale.z = lg.dimensions[1]
                else:
                    m.type = Marker.SPHERE
                    m.scale.x = m.scale.y = m.scale.z = 0.02
                m.pose.position = ps.pose.position
                m.pose.orientation = ps.pose.orientation
                m.color.r, m.color.g, m.color.b = (0.616, 0.540, 0.993) if idx == 0 else \
                    (0.122, 0.467, 0.706) if idx == n - 1 else (0.122, 0.467, 0.706)
                m.color.a = 1.0 if idx in (0, n - 1) else 0.16
                m.lifetime = Duration(seconds=0).to_msg()
                markers.markers.append(m)
        self.jsp_path_marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = InformedRRTStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
