#!/usr/bin/env python3

"""
Robot geometry publisher — parses URDF collision geometry and publishes RigidBodyGeom messages.

All robot metadata (base link name, world frame, joint map) is derived directly from the
`robot_description` ROS2 parameter. No external robot module required.

Author: Clinton Enwerem
"""

import rclpy
from rclpy.node import Node
import xml.etree.ElementTree as ET
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped
import transforms3d as t3d
from kinematic_planner_interfaces.msg import RigidBodyGeom
from rclpy.duration import Duration
from kinematic_planner.collision.collision_utils import get_link_transform
import spatialmath as sm
import numpy as np


class RobotGeomPublisher(Node):
    def __init__(self):
        super().__init__("robot_geom_publisher")

        self.declare_parameter("base_link_name", "base_link")
        self.declare_parameter("world_frame", "world")
        if not self.has_parameter("robot_description"):
            self.declare_parameter("robot_description", "")

        urdf_str = self.get_parameter("robot_description").get_parameter_value().string_value
        if not urdf_str:
            raise RuntimeError("robot_description parameter is empty!")

        self.base_link_name = self.get_parameter("base_link_name").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value

        self.tree = ET.fromstring(urdf_str)
        robot_name_attr = self.tree.attrib.get("name")
        self.robot_name = robot_name_attr.lower().strip().replace(" ", "_") if robot_name_attr else "robot"

        self.robot_geom = None
        self.pub = self.create_publisher(RigidBodyGeom, "robot_geometry", 10)
        self.timer = self.create_timer(6.0, self.publish_robot_geometry_once)
        self.first_publish = True

    def publish_robot_geometry_once(self):
        self.publish_robot_geometry()
        self.create_timer(1.0, self.publish_robot_geometry)

    def publish_robot_geometry(self):
        robot_geom = RigidBodyGeom()
        link_poses = []
        link_geometries = []
        link_names = []
        link_geom_origins = []

        # build joint map: child_link -> (parent_link, T_joint_in_parent)
        joint_map = {}
        for joint in self.tree.findall("joint"):
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            if parent_el is None or child_el is None:
                continue
            parent = parent_el.attrib.get("link", "")
            child = child_el.attrib.get("link", "")
            joint_origin = joint.find("origin")
            if joint_origin is not None:
                xyz_origin = [float(x) for x in joint_origin.attrib.get("xyz", "0 0 0").split()]
                rpy_origin = [float(x) for x in joint_origin.attrib.get("rpy", "0 0 0").split()]
            else:
                xyz_origin = [0, 0, 0]
                rpy_origin = [0, 0, 0]
            T_joint = sm.SE3(xyz_origin) * sm.SE3.RPY(rpy_origin, order="xyz", unit="rad")
            joint_map[child] = (parent, T_joint)

        for link in self.tree.findall("link"):
            link_name = link.attrib.get("name")

            for col in link.findall("collision"):
                geom = col.find("geometry")
                if geom is None:
                    continue
                origin = col.find("origin")
                if origin is not None:
                    xyz = [float(x) for x in origin.attrib.get("xyz", "0 0 0").split()]
                    rpy = [float(x) for x in origin.attrib.get("rpy", "0 0 0").split()]
                else:
                    xyz = [0, 0, 0]
                    rpy = [0, 0, 0]

                quat_offset = t3d.euler.euler2quat(rpy[0], rpy[1], rpy[2])  # [w,x,y,z]

                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = xyz
                pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quat_offset
                link_geom_origins.append(pose)

                try:
                    T_link = get_link_transform(link_name, joint_map).A
                except Exception as e:
                    self.get_logger().error(f"Transform error for '{link_name}': {e}")
                    continue

                try:
                    T_final = T_link @ t3d.affines.compose(
                        T=xyz, R=t3d.quaternions.quat2mat(quat_offset), Z=[1, 1, 1]
                    )
                except Exception as e:
                    self.get_logger().error(f"Compose transform error for '{link_name}': {e}")
                    continue

                trans_final, rot_final, _, _ = t3d.affines.decompose44(T_final)
                quat_final = t3d.quaternions.mat2quat(rot_final)

                prim = SolidPrimitive()
                if geom.find("box") is not None:
                    size = [float(x) for x in geom.find("box").attrib.get("size").split()]
                    prim.type = SolidPrimitive.BOX
                    prim.dimensions = size
                elif geom.find("sphere") is not None:
                    r = float(geom.find("sphere").attrib.get("radius"))
                    prim.type = SolidPrimitive.SPHERE
                    prim.dimensions = [r]
                elif geom.find("cylinder") is not None:
                    r = float(geom.find("cylinder").attrib.get("radius"))
                    length = float(geom.find("cylinder").attrib.get("length"))
                    prim.type = SolidPrimitive.CYLINDER
                    prim.dimensions = [r, length]
                else:
                    continue

                link_geometries.append(prim)

                link_pose = PoseStamped()
                link_pose.header.frame_id = self.base_link_name
                link_pose.header.stamp = self.get_clock().now().to_msg()
                link_pose.pose.position.x = trans_final[0]
                link_pose.pose.position.y = trans_final[1]
                link_pose.pose.position.z = trans_final[2]
                link_pose.pose.orientation.w = quat_final[0]
                link_pose.pose.orientation.x = quat_final[1]
                link_pose.pose.orientation.y = quat_final[2]
                link_pose.pose.orientation.z = quat_final[3]
                link_poses.append(link_pose)
                link_names.append(link_name)

        robot_geom.link_poses = link_poses
        robot_geom.link_geometries = link_geometries
        robot_geom.link_names = link_names
        robot_geom.link_geom_origins = link_geom_origins

        if self.first_publish:
            self.get_logger().info(
                f"Publishing geometry for {self.robot_name}: {len(link_names)} links: {link_names}"
            )
            self.first_publish = False
        self.pub.publish(robot_geom)


def main(args=None):
    rclpy.init(args=args)
    node = RobotGeomPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
