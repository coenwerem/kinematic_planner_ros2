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
from typing import Dict

from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.collision.collision_utils import se3_to_pose_stamped
from kinematic_planner.planning.informed_rrt_star import InformedRRTStar
from kinematic_planner.scripts.planner_node import build_collision_fn

try:
    from roboticstoolbox.robot.ERobot import ERobot
    _RTB_AVAILABLE = True
except ImportError:
    _RTB_AVAILABLE = False

__all__ = ["InformedRRTStarPlanner", "build_collision_fn"]


def _build_rtb_model(urdf_path: str):
    class _RobotModel(ERobot):
        def __init__(self, path):
            links, name, urdf_string, urdf_fp = super().URDF_read(path)
            super().__init__(links, name=name.upper(), manufacturer="Custom",
                             urdf_string=urdf_string, urdf_filepath=urdf_fp)
    return _RobotModel(urdf_path)


class InformedRRTStarPlanner(Node):
    """ROS2 node wrapper around the Informed RRT* planner."""

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

        collision_fn = build_collision_fn(
            robot_config=self.robot_config,
            robot_geom=self.robot_geom,
            obstacle_geom=self.obstacle_geom,
            rtb_model=self.rtb_model,
            collision_checker=self.collision_checker,
            min_obs_dist=self.min_obs_dist,
            check_collision=self.check_collision,
            get_logger=self.get_logger,
        )
        self.planner = InformedRRTStar(
            start=self.start_config,
            goal=self.goal_config,
            joint_limits=self.joint_limits,
            expand_dist=self.rrts_expand_dist,
            path_resolution=self.rrts_path_resolution,
            max_iter=self.rrts_max_iter,
            connect_circle_dist=self.rrts_connect_circle_dist,
            collision_fn=collision_fn,
            search_until_max_iter=self.rrts_search_until_max_iter,
            rng=np.random.default_rng(self.random_seed),
        )

        self.planning_attempts += 1
        self.get_logger().info(f"Informed RRT* attempt {self.planning_attempts}/{self.max_planning_attempts}")
        computed_path = self.planner.plan()
        if self.planner.start_goal_collision:
            self.start_goal_collision = self.planner.start_goal_collision

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
                cost = InformedRRTStar.compute_path_cost(computed_path)
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
