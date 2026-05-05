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
from robot_3r_interfaces.msg import SceneObstacles
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA
import math
import itertools


class ObstaclePublisher(Node):
    def __init__(self):
        super().__init__("obstacle_publisher")

        # ---- robot-frame parameters ------
        self.declare_parameter("base_link_name", "base_link")
        self.declare_parameter("world_frame", "world")

        # ---- obstacle type & scene mode ----------------------------------------
        self.declare_parameter("obstacle_type", "box")
        self.declare_parameter("is_dense", False)

        # ---- dense-mode ring geometry — no 3R assumptions ----------------------
        self.declare_parameter("dense_num_obstacles", 10)
        self.declare_parameter("dense_ring_radius", 0.45)
        self.declare_parameter("dense_center_z", 0.5)
        # platform_height is now a required parameter — set it to match your robot's
        # first link height (e.g. 0.755 for 3R).  Defaults to 0.5 (generic).
        self.declare_parameter("dense_platform_height", 0.5)

        self.base_link_name = self.get_parameter("base_link_name").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.is_dense = self.get_parameter("is_dense").get_parameter_value().bool_value
        self.obstacle_type = self.get_parameter("obstacle_type").get_parameter_value().string_value

        if self.is_dense:
            num_obstacles = self.get_parameter("dense_num_obstacles").get_parameter_value().integer_value
            radius = self.get_parameter("dense_ring_radius").get_parameter_value().double_value
            center_z = self.get_parameter("dense_center_z").get_parameter_value().double_value
            platform_height = self.get_parameter("dense_platform_height").get_parameter_value().double_value
            obs_z = [1.0, 0.8]
            obstacle_positions = []
            obstacle_sizes = []
            for i in range(num_obstacles):
                angle = 2 * math.pi * i / num_obstacles
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                if i % 2 == 0:
                    obstacle_sizes.append([0.1, 0.1, obs_z[0]])
                    obstacle_positions.extend([x, y, center_z * obs_z[i % 2]])
                else:
                    obstacle_sizes.append([0.15, 0.15, obs_z[1]])
                    obstacle_positions.extend([x, y, platform_height + center_z * obs_z[i % 2]])
            self.declare_parameter("num_obstacles", num_obstacles)
            self.declare_parameter("obstacle_positions", obstacle_positions)
            self.declare_parameter("obstacle_sizes", list(itertools.chain.from_iterable(obstacle_sizes)))
        else:
            self.declare_parameter("num_obstacles", 2)
            self.declare_parameter("obstacle_sizes", [0.1, 0.1, 1.0])
            self.declare_parameter("obstacle_positions", [0.0, 0.6, 0.5, 0.6, 0.0, 0.5])

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

    def timer_callback(self):
        self.publish_obstacles()

    def publish_obstacles(self):
        msg = SceneObstacles()
        poses = []
        obstacles = []
        obstacle_ids = []

        for i in range(self.num_obstacles):
            pose = PoseStamped()
            pose.header.frame_id = self.world_frame
            pose.pose.position.x = self.obstacle_positions[3 * i]
            pose.pose.position.y = self.obstacle_positions[3 * i + 1]
            pose.pose.position.z = self.obstacle_positions[3 * i + 2]
            pose.pose.orientation.w = 1.0
            poses.append(pose)
            obstacle_ids.append(i + 1)

            if self.obstacle_type == "box":
                obstacle = SolidPrimitive()
                obstacle.type = SolidPrimitive.BOX
                obstacle.dimensions = (
                    self.obstacle_sizes[3 * i:3 * i + 3] if self.is_dense else self.obstacle_sizes
                )
                obstacles.append(obstacle)
            self.obstacle_dict[i + 1] = (obstacle, pose)

        msg.scene_obstacles = obstacles
        msg.obstacle_poses = poses
        msg.obstacle_ids = obstacle_ids
        self.scene_obstacle_publisher_.publish(msg)

        if self.first_publish:
            self.get_logger().info(f"Published {len(obstacles)} obstacles.")
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
