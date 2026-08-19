#!/usr/bin/env python3

"""
ROS 2 node: collision-free trajectory planning via RRT*.

Accepts any N-DOF robot whose URDF uses convex collision primitives (box, cylinder, sphere).
Robot model is supplied entirely through the `robot_description` ROS2 parameter
package required.

Usage:
    ros2 run kinematic_planner planner_node --ros-args \
        -p goal_config:="[1.5093, 0.6072, 1.4052]" \
        -p check_collision:=True

Algorithm reference:
    [1] Karaman & Frazzoli, "Sampling-based algorithms for optimal motion planning," IJRR 2011.
    [2] AtsushiSakai/PythonRobotics — rrt_star_seven_joint_arm_control

Author: Clinton Enwerem
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Point
from rclpy.duration import Duration
import numpy as np
import random
import subprocess
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

import fcl
from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.collision.collision_utils import obstacle_to_fclobj, create_fcl_object, se3_to_pose_stamped
from kinematic_planner.planning.rrt_star import RRTStar
from kinematic_planner.planning.tree import TreeNode

try:
    from roboticstoolbox.robot.ERobot import ERobot
    _RTB_AVAILABLE = True
except ImportError:
    _RTB_AVAILABLE = False


def _build_rtb_model(urdf_path: str):
    """Build a Robotics Toolbox ERobot from a URDF file path."""
    class _RobotModel(ERobot):
        def __init__(self, path):
            links, name, urdf_string, urdf_fp = super().URDF_read(path)
            super().__init__(links, name=name.upper(), manufacturer="Custom",
                             urdf_string=urdf_string, urdf_filepath=urdf_fp)
    return _RobotModel(urdf_path)


class _NullClockNode:
    """Supplies get_clock() for se3_to_pose_stamped() outside an rclpy.Node."""

    def get_clock(self):
        from rclpy.clock import Clock
        return Clock()


def build_collision_fn(robot_config, robot_geom, obstacle_geom, rtb_model,
                        collision_checker: str, min_obs_dist: float,
                        check_collision: bool, get_logger=lambda: None):
    """Build a collision_fn closure for kinematic_planner.planning.

    check_collision=False returns a closure reporting collision-free
    unconditionally, before any FCL/robot_geom code runs. RRTPlannerBase
    (kinematic_planner.planning.tree) never reads check_collision itself;
    build_collision_fn is the single call site consuming check_collision.
    """
    if not check_collision:
        return lambda _node: True

    def collision_fn(candidate_node: TreeNode) -> bool:
        obs_fcl_objects = list(obstacle_to_fclobj(obstacles=obstacle_geom))
        for q in candidate_node.path_q:
            rtb_model.q = q
            for link in rtb_model.links:
                link_name = link.name
                skip_names = [robot_config.base_link_name, "link0", robot_config.world_frame]
                if link_name in skip_names:
                    continue
                if link_name not in robot_geom.link_names:
                    continue
                idx = robot_geom.link_names.index(link_name)
                T_fk_se3 = rtb_model.fkine(q, end=link.name, include_base=True)
                link_geometry = robot_geom.link_geometries[idx]
                if link_geometry.type == SolidPrimitive.BOX:
                    geom = fcl.Box(*link_geometry.dimensions)
                elif link_geometry.type == SolidPrimitive.SPHERE:
                    geom = fcl.Sphere(link_geometry.dimensions[0])
                elif link_geometry.type == SolidPrimitive.CYLINDER:
                    geom = fcl.Cylinder(link_geometry.dimensions[0], link_geometry.dimensions[1])
                else:
                    continue
                rob_pose = se3_to_pose_stamped(
                    T_fk_se3, _NullClockNode(), frame_id=robot_config.base_link_name,
                )
                rob_obj = create_fcl_object(rob_pose, geom)
                for obs_obj in obs_fcl_objects:
                    if collision_checker == "bvol":
                        creq = fcl.CollisionRequest()
                        creq.enable_contact = True
                        cres = fcl.CollisionResult()
                        ret = fcl.collide(rob_obj, obs_obj, creq, cres)
                        collision_found = cres.is_collision or ret > 0
                        if collision_found:
                            return False
                    elif collision_checker == "proximity":
                        dreq = fcl.DistanceRequest(enable_signed_distance=True)
                        dres = fcl.DistanceResult()
                        fcl.distance(rob_obj, obs_obj, dreq, dres)
                        # single clearance rule, meters, shared with
                        # informed_rrt_star_node.py through build_collision_fn:
                        # reject whenever signed distance drops below min_obs_dist.
                        below_clearance = dres.min_distance < min_obs_dist
                        if below_clearance:
                            return False
        return True

    return collision_fn


class SamplingBasedJSPlanner(Node):
    """ROS2 node wrapper around the RRT* planner."""

    # ------------------------------------------------------------------
    # Node __init__
    # ------------------------------------------------------------------
    def __init__(self, node_name: str = "sampling_based_planner", queue_size=10):
        self.node_name = node_name
        self.queue_size = queue_size
        super().__init__(self.node_name)
        self.get_logger().info(
            f"\n--------------------------------------------------\n"
            f"Initializing {self.node_name} node...\n"
            f"--------------------------------------------------"
        )

        # ---- parameters --------------------------------------------------
        self.declare_parameter("planning_algorithm", "rrt_star",
            ParameterDescriptor(description="Planning algorithm; only 'rrt_star' supported."))
        self.declare_parameter("stop_if_plan_found", True)
        self.declare_parameter("verbose", False)
        self.declare_parameter("rrts_expand_dist", 0.3)
        self.declare_parameter("rrts_path_resolution", 0.1)
        self.declare_parameter("rrts_max_iter", 300)
        self.declare_parameter("rrts_connect_circle_dist", 20)
        self.declare_parameter("rrts_search_until_max_iter", False)
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
        # disabled_collision_pairs: list of "link_a:link_b" strings
        self.declare_parameter("disabled_collision_pairs", [""])
        self.declare_parameter("world_frame", "world")
        # robot_description loaded from URDF (set by launch file or robot_state_publisher)
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

        # ---- load robot config from URDF --------------------
        urdf_str = p("robot_description").get_parameter_value().string_value
        if not urdf_str:
            raise RuntimeError("robot_description parameter is empty! Pass it via the launch file.")

        raw_disabled = p("disabled_collision_pairs").get_parameter_value().string_array_value
        disabled_pairs = []
        for pair_str in raw_disabled:
            if ":" in pair_str:
                a, b = pair_str.split(":", 1)
                disabled_pairs.append((a.strip(), b.strip()))

        world_frame = p("world_frame").get_parameter_value().string_value
        self.robot_config = RobotConfig.from_urdf(urdf_str, disabled_pairs=disabled_pairs,
                                                   world_frame=world_frame)
        self.joint_limits = self.robot_config.joint_limits

        # ---- load RTB model for FK (writes URDF to a temp file) ---------
        if not _RTB_AVAILABLE:
            raise RuntimeError("roboticstoolbox-python is required. Install it with: pip install roboticstoolbox-python")
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w") as f:
            f.write(urdf_str)
            tmp_urdf = f.name
        self.rtb_model = _build_rtb_model(tmp_urdf)
        os.unlink(tmp_urdf)

        # goal configuration
        midpoint = [(lo + hi) / 2.0 for lo, hi in self.joint_limits]
        self.declare_parameter("goal_config", midpoint)
        self.goal_config = p("goal_config").get_parameter_value().double_array_value
        if self.verbose:
            self.get_logger().info(f"Planning to goal: {self.goal_config} via {self.pl_alg}")

        # ---- planning state ---------------------------------------------
        if self.stop_if_plan_found:
            self.planning_done = False
        self.planning_attempts = 0
        self.planning_failed = False
        self.start_config = None
        self.obstacle_geom = None
        self.robot_geom = None
        self.rand_area = [(lo, hi) for lo, hi in self.joint_limits]
        self.rrt_star = None
        self.computed_path = None

        # ---- subscribers ------------------------------------------------
        self.create_subscription(JointState, "/joint_states", self.compute_plan, queue_size)
        self.create_subscription(SceneObstacles, "/scene_obstacles", self.get_scene_obs_cb, queue_size)
        self.create_subscription(RigidBodyGeom, "robot_geometry", self.robot_geom_cb, queue_size)

        # ---- publishers -------------------------------------------------
        self.plan_pub = self.create_publisher(JointSpacePath, "smpb_planner/jsp_path", queue_size)
        self.ee_path_pub = self.create_publisher(PoseStamped, "smpb_planner/ee_path", queue_size)
        self.jsp_path_marker_pub = self.create_publisher(MarkerArray, "planned_jsp_path_markers", 10)
        self.ee_path_marker_pub = self.create_publisher(MarkerArray, "ee_path_marker", 10)

        if self.stop_if_plan_found:
            self.stop_timer = self.create_timer(1.0, self.stop_node_cb)
        if self.show_jsp_waypoints:
            self.create_timer(1.0, self.publish_jsp_path_markers_cb)
        if self.show_ee_path:
            self.create_timer(1.0, self.publish_ee_path_markers_cb)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def stop_node_cb(self):
        if getattr(self, "planning_done", False):
            return
        if getattr(self, "planning_failed", False) and self.stop_on_failure:
            self.get_logger().info("Stopping node due to planning failure.")
            raise SystemExit

    def robot_geom_cb(self, msg: RigidBodyGeom):
        if self.robot_geom is None:
            self.robot_geom = msg
            if self.verbose:
                self.get_logger().info(f"Received robot geometry. Links: {self.robot_geom.link_names}")

    def get_scene_obs_cb(self, msg: SceneObstacles):
        self.obstacle_geom = msg
        if self.verbose:
            self.get_logger().info(f"Received {len(msg.scene_obstacles)} obstacles.")

    def compute_plan(self, msg: JointState):
        if getattr(self, "planning_done", False) or getattr(self, "planning_failed", False):
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
        self.rrt_star = RRTStar(
            start=self.start_config,
            goal=self.goal_config,
            joint_limits=self.joint_limits,
            expand_dist=self.rrts_expand_dist,
            path_resolution=self.rrts_path_resolution,
            max_iter=self.rrts_max_iter,
            connect_circle_dist=self.rrts_connect_circle_dist,
            goal_sample_rate=self.rrts_goal_sample_rate,
            collision_fn=collision_fn,
            use_goal_biased_sampling=self.use_goal_biased_sampling,
            goal_noise_sigma=self.goal_noise_sigma,
            search_until_max_iter=self.rrts_search_until_max_iter,
            rng=np.random.default_rng(self.random_seed),
        )

        self.planning_attempts += 1
        self.get_logger().info(f"Planning attempt {self.planning_attempts}/{self.max_planning_attempts}")
        computed_path = self.rrt_star.plan()
        if self.rrt_star.start_goal_collision:
            self.start_goal_collision = self.rrt_star.start_goal_collision

        if computed_path is None:
            if self.start_goal_collision == "start":
                self.get_logger().warning("Start configuration is in collision.")
            elif self.start_goal_collision == "goal":
                self.get_logger().warning("Goal configuration is in collision.")
            elif self.proximity_alert:
                self.get_logger().warning(f"Obstacles too close (attempt {self.planning_attempts})")
            else:
                self.get_logger().warning(f"Planning failed (attempt {self.planning_attempts})")
            if self.planning_attempts >= self.max_planning_attempts:
                self.planning_failed = True
                if self.stop_on_failure:
                    self.create_timer(1.0, lambda: rclpy.shutdown())
        else:
            self.get_logger().info(f"Path found on attempt {self.planning_attempts}!")
            self.planning_attempts = 0
            plan_msg = JointSpacePath()
            plan_msg.joint_names = list(self.robot_config.joint_names)
            for q in computed_path:
                wp = JointWaypoint()
                self.get_logger().info(f"\033[92mWaypoint: {np.round(q, 2)}\033[0m")
                wp.positions = np.array(q).tolist()
                plan_msg.waypoints.append(wp)
            self.get_logger().info(f"Waypoints: {len(plan_msg.waypoints)}")
            if self.print_metrics:
                path_cost = self.rrt_star.compute_path_cost(computed_path)
                self.get_logger().info(f"Path cost: {path_cost:.3f} rad")
            self.plan_pub.publish(plan_msg)
            self.computed_path = computed_path
            if self.stop_if_plan_found:
                self.planning_done = True

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def publish_ee_path_markers_cb(self):
        if self.computed_path is not None and self.show_ee_path:
            self.publish_ee_path(self.computed_path)

    def publish_ee_path(self, computed_path):
        points = []
        last_pose_stamped = None
        for q in computed_path:
            try:
                self.rtb_model.q = q
                T = self.rtb_model.fkine(q).A
            except Exception:
                continue
            trans = T[:3, 3].tolist()
            R = T[:3, :3]
            quat = t3d.quaternions.mat2quat(R)
            ps = PoseStamped()
            ps.header.frame_id = self.robot_config.base_link_name
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = trans[0]
            ps.pose.position.y = trans[1]
            ps.pose.position.z = trans[2]
            ps.pose.orientation.w = float(quat[0])
            ps.pose.orientation.x = float(quat[1])
            ps.pose.orientation.y = float(quat[2])
            ps.pose.orientation.z = float(quat[3])
            last_pose_stamped = ps
            points.append(Point(x=trans[0], y=trans[1], z=trans[2]))

        if points:
            marker = Marker()
            marker.header.frame_id = self.robot_config.base_link_name
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "ee_path"
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.015
            marker.color.a = 1.0
            marker.color.g = 1.0
            marker.points = points
            marker.lifetime = Duration(seconds=0).to_msg()
            self.ee_path_marker_pub.publish(MarkerArray(markers=[marker]))
        if last_pose_stamped is not None:
            self.ee_path_pub.publish(last_pose_stamped)

    def publish_jsp_path_markers_cb(self):
        if self.computed_path is not None and self.show_jsp_waypoints:
            self.publish_jsp_path_markers(self.computed_path)

    def publish_jsp_path_markers(self, computed_path):
        if self.robot_geom is None:
            return

        markers = MarkerArray()
        n_waypoints = len(computed_path)
        default_color = (0.122, 0.467, 0.706)  # matplotlib blue

        for idx, q in enumerate(computed_path):
            marker_id = 0
            rtb_link_poses: Dict[str, PoseStamped] = {}
            try:
                for link in self.rtb_model.links:
                    T = self.rtb_model.fkine(q, end=link.name, include_base=True)
                    rtb_link_poses[link.name] = se3_to_pose_stamped(
                        T, self, frame_id=self.robot_config.base_link_name)
            except Exception as e:
                self.get_logger().warning(f"FK failed for waypoint {idx}: {e}")
                continue

            for i, link_name in enumerate(self.robot_geom.link_names):
                try:
                    link_geom = self.robot_geom.link_geometries[i]
                    link_geom_orig = self.robot_geom.link_geom_origins[i]
                except IndexError:
                    continue

                if link_name not in rtb_link_poses:
                    continue

                link_pose = rtb_link_poses[link_name].pose
                link_T = sm.SE3(link_pose.position.x, link_pose.position.y, link_pose.position.z) \
                    * sm.SE3.RPY(*t3d.euler.quat2euler(
                        [link_pose.orientation.w, link_pose.orientation.x,
                         link_pose.orientation.y, link_pose.orientation.z], axes="sxyz"))
                origin_T = sm.SE3(link_geom_orig.position.x, link_geom_orig.position.y, link_geom_orig.position.z) \
                    * sm.SE3.RPY(*t3d.euler.quat2euler(
                        [link_geom_orig.orientation.w, link_geom_orig.orientation.x,
                         link_geom_orig.orientation.y, link_geom_orig.orientation.z], axes="sxyz"))
                geom_T = link_T * origin_T
                ps = se3_to_pose_stamped(geom_T, self, frame_id=self.robot_config.base_link_name)

                p = Pose()
                p.position = ps.pose.position
                p.orientation = ps.pose.orientation

                m = Marker()
                m.header.frame_id = self.robot_config.base_link_name
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = f"path_wp_{idx}"
                m.id = marker_id
                marker_id += 1

                if link_geom.type == SolidPrimitive.BOX:
                    m.type = Marker.CUBE
                    sx, sy, sz = link_geom.dimensions
                    m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
                elif link_geom.type == SolidPrimitive.SPHERE:
                    m.type = Marker.SPHERE
                    r = link_geom.dimensions[0]
                    m.scale.x = m.scale.y = m.scale.z = 2 * r
                elif link_geom.type == SolidPrimitive.CYLINDER:
                    m.type = Marker.CYLINDER
                    m.scale.x = m.scale.y = 2 * link_geom.dimensions[0]
                    m.scale.z = link_geom.dimensions[1]
                else:
                    m.type = Marker.SPHERE
                    m.scale.x = m.scale.y = m.scale.z = 0.02

                m.pose = p
                # start waypoint: purple; goal waypoint: blue; others: faded
                if idx == 0:
                    m.color.r, m.color.g, m.color.b, m.color.a = 0.616, 0.540, 0.993, 1.0
                elif idx == n_waypoints - 1:
                    m.color.r, m.color.g, m.color.b, m.color.a = *default_color, 1.0
                else:
                    m.color.r, m.color.g, m.color.b, m.color.a = *default_color, 0.16
                m.lifetime = Duration(seconds=0).to_msg()
                markers.markers.append(m)

        self.jsp_path_marker_pub.publish(markers)
        if self.verbose:
            self.get_logger().info(f"Published {len(markers.markers)} markers for {n_waypoints} waypoints.")


def main(args=None):
    rclpy.init(args=args)
    node = SamplingBasedJSPlanner()
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
