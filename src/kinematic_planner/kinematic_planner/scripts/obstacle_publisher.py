#!/usr/bin/env python3

"""
Simple obstacle publisher for RViz and FCL collision checking.

Publishes box obstacles on /scene_obstacles and visualisation markers on /obstacle_markers.
All robot-specific constants (base link name, world frame, dense-mode ring dimensions)
are ROS2 parameters — no hardcoded robot geometry.

Author: Clinton Enwerem
"""

import rclpy
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose
from kinematic_planner_interfaces.msg import SceneObstacles
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA
import math
import itertools


def default_obstacle_scene(is_dense, platform_height, num_obstacles=10, radius=0.45, center_z=0.5):
    """Positions and sizes for the built-in demo obstacle scenes, both
    resting on top of the robot's mounting platform (bottom face at
    platform_height) rather than floating through it at platform_height
    + center_z * height. center_z is the fraction of each obstacle's own
    height between its bottom face and its center pose (0.5 for centered).

    Returns (positions, sizes), each a flat list of 3 floats per obstacle,
    matching ObstaclePublisher's obstacle_positions/obstacle_sizes
    parameter layout.
    """
    if is_dense:
        obs_heights = [1.0, 0.8]
        positions, sizes = [], []
        for i in range(num_obstacles):
            angle = 2 * math.pi * i / num_obstacles
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            height = obs_heights[i % 2]
            side = 0.1 if i % 2 == 0 else 0.15
            sizes.append([side, side, height])
            positions.extend([x, y, platform_height + center_z * height])
        return positions, list(itertools.chain.from_iterable(sizes))

    height = 1.0
    z = platform_height + center_z * height
    return [0.0, 0.6, z, 0.6, 0.0, z], [0.1, 0.1, height]


class ObstaclePublisher(Node):
    def __init__(self, **kwargs):
        super().__init__("obstacle_publisher", **kwargs)

        # ---- robot-frame parameters ------
        self.declare_parameter("base_link_name", "base_link")
        self.declare_parameter("world_frame", "world")

        # ---- obstacle type & scene mode ----------------------------------------
        self.declare_parameter("obstacle_type", "box")
        self.declare_parameter("is_dense", False)

        # ---- obstacle-scene geometry — platform_height is the only value
        # you need to set per robot (its mounting platform's top height,
        # e.g. 0.755 for 3R); the rest default to a generic scene ----------
        self.declare_parameter("dense_num_obstacles", 10)
        self.declare_parameter("dense_ring_radius", 0.45)
        self.declare_parameter("center_z", 0.5)
        self.declare_parameter("platform_height", 0.5)

        self.base_link_name = self.get_parameter("base_link_name").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.is_dense = self.get_parameter("is_dense").get_parameter_value().bool_value
        self.obstacle_type = self.get_parameter("obstacle_type").get_parameter_value().string_value

        center_z = self.get_parameter("center_z").get_parameter_value().double_value
        platform_height = self.get_parameter("platform_height").get_parameter_value().double_value
        if self.is_dense:
            num_obstacles = self.get_parameter("dense_num_obstacles").get_parameter_value().integer_value
            radius = self.get_parameter("dense_ring_radius").get_parameter_value().double_value
            obstacle_positions, obstacle_sizes = default_obstacle_scene(
                is_dense=True, platform_height=platform_height,
                num_obstacles=num_obstacles, radius=radius, center_z=center_z,
            )
            self.declare_parameter("num_obstacles", num_obstacles)
            self.declare_parameter("obstacle_positions", obstacle_positions)
            self.declare_parameter("obstacle_sizes", obstacle_sizes)
        else:
            obstacle_positions, obstacle_sizes = default_obstacle_scene(
                is_dense=False, platform_height=platform_height, center_z=center_z,
            )
            self.declare_parameter("num_obstacles", 2)
            self.declare_parameter("obstacle_sizes", obstacle_sizes)
            self.declare_parameter("obstacle_positions", obstacle_positions)

        self.num_obstacles = self.get_parameter("num_obstacles").get_parameter_value().integer_value
        self.obstacle_sizes = self.get_parameter("obstacle_sizes").get_parameter_value().double_array_value
        self.obstacle_positions = self.get_parameter("obstacle_positions").get_parameter_value().double_array_value

        self.scene_obstacle_publisher_ = self.create_publisher(SceneObstacles, "scene_obstacles", 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.obstacle_marker_pub = self.create_publisher(MarkerArray, "obstacle_markers", 10)
        self.obstacle_marker_pub_timer = self.create_timer(0.1, self.publish_obstacle_markers)
        self.obstacle_dict = {}
        self.first_publish = True
        self.first_marker_publish = True
        self.first_nonbox_marker_warning = True

    def timer_callback(self):
        self.publish_obstacles()

    def build_obstacles_message(self) -> SceneObstacles:
        msg = SceneObstacles()
        poses = []
        obstacles = []
        obstacle_ids = []

        stride = {"box": 3, "sphere": 1, "cylinder": 2}[self.obstacle_type]

        for i in range(self.num_obstacles):
            pose = PoseStamped()
            pose.header.frame_id = self.world_frame
            pose.pose.position.x = self.obstacle_positions[3 * i]
            pose.pose.position.y = self.obstacle_positions[3 * i + 1]
            pose.pose.position.z = self.obstacle_positions[3 * i + 2]
            pose.pose.orientation.w = 1.0
            poses.append(pose)
            obstacle_ids.append(i + 1)

            dims = self.obstacle_sizes[stride * i:stride * i + stride] if self.is_dense else self.obstacle_sizes[:stride]

            obstacle = SolidPrimitive()
            if self.obstacle_type == "box":
                obstacle.type = SolidPrimitive.BOX
            elif self.obstacle_type == "sphere":
                obstacle.type = SolidPrimitive.SPHERE
            elif self.obstacle_type == "cylinder":
                obstacle.type = SolidPrimitive.CYLINDER
            else:
                self.get_logger().error(f"Unsupported obstacle_type: {self.obstacle_type}")
                continue
            obstacle.dimensions = list(dims)
            obstacles.append(obstacle)
            self.obstacle_dict[i + 1] = (obstacle, pose)

        msg.scene_obstacles = obstacles
        msg.obstacle_poses = poses
        msg.obstacle_ids = obstacle_ids
        return msg

    def publish_obstacles(self):
        msg = self.build_obstacles_message()
        self.scene_obstacle_publisher_.publish(msg)
        if self.first_publish:
            self.get_logger().info(f"Published {len(msg.scene_obstacles)} obstacles.")
            self.first_publish = False

    def publish_obstacle_markers(self):
        if not self.obstacle_dict:
            return
        obstacles = []
        obstacle_poses = []
        for obstacle, pose in self.obstacle_dict.values():
            obstacles.append(obstacle)
            obstacle_poses.append(pose)

        marker_array = MarkerArray()
        for i, (obstacle, pose) in enumerate(zip(obstacles, obstacle_poses)):
            if self.obstacle_type != "box":
                if self.first_nonbox_marker_warning:
                    self.get_logger().warning(
                        f"Marker rendering is unimplemented for obstacle_type={self.obstacle_type!r} "
                        "obstacles; skipping their markers."
                    )
                    self.first_nonbox_marker_warning = False
                continue
            marker = Marker()
            marker_pose = Pose()
            marker_pose.position = pose.pose.position
            marker_pose.orientation = pose.pose.orientation
            marker.header.frame_id = self.world_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i + 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = marker_pose
            marker.scale = Vector3()
            if self.is_dense:
                marker.scale.x = self.obstacle_sizes[3 * i]
                marker.scale.y = self.obstacle_sizes[3 * i + 1]
                marker.scale.z = self.obstacle_sizes[3 * i + 2]
            else:
                marker.scale.x = self.obstacle_sizes[0]
                marker.scale.y = self.obstacle_sizes[1]
                marker.scale.z = self.obstacle_sizes[2]
            marker.color = ColorRGBA(r=1.0, g=0.64, b=0.0, a=1.0)
            marker_array.markers.append(marker)

        self.obstacle_marker_pub.publish(marker_array)
        if self.first_marker_publish:
            self.get_logger().info(f"Published {len(marker_array.markers)} obstacle markers.")
            self.first_marker_publish = False


def main(args=None):
    rclpy.init(args=args)
    node = ObstaclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
