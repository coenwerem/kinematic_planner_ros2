#!/usr/bin/env python3

from robot_3r_interfaces.msg import RigidBodyGeom, SceneObstacles
from shape_msgs.msg import SolidPrimitive
import fcl
from geometry_msgs.msg import PoseStamped
import transforms3d as t3d
import spatialmath as sm
import numpy as np


def robotgeom_to_fclobj(robot_geom: RigidBodyGeom):
    fcl_objects = []
    for i in range(len(robot_geom.link_names)):
        link_pose = robot_geom.link_poses[i]
        link_geometry = robot_geom.link_geometries[i]
        link_geometry_type = link_geometry.type
        if link_geometry_type == SolidPrimitive.BOX:
            fcl_geom = fcl.Box(*link_geometry.dimensions)
        elif link_geometry_type == SolidPrimitive.SPHERE:
            fcl_geom = fcl.Sphere(link_geometry.dimensions[SolidPrimitive.SPHERE_RADIUS])
        elif link_geometry_type == SolidPrimitive.CYLINDER:
            fcl_geom = fcl.Cylinder(
                link_geometry.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                link_geometry.dimensions[SolidPrimitive.CYLINDER_HEIGHT],
            )
        else:
            print(f"Unsupported geometry type: {link_geometry_type}")
            continue
        fcl_obj = create_fcl_object(link_pose, fcl_geom)
        fcl_objects.append(fcl_obj)
    return fcl_objects


def obstacle_to_fclobj(obstacles: SceneObstacles):
    fcl_objects = []
    for i in range(len(obstacles.obstacle_ids)):
        obstacle_pose = obstacles.obstacle_poses[i]
        obstacle_geometry = obstacles.scene_obstacles[i]
        obstacle_geometry_type = obstacle_geometry.type
        if obstacle_geometry_type == SolidPrimitive.BOX:
            fcl_geom = fcl.Box(*obstacle_geometry.dimensions)
        elif obstacle_geometry_type == SolidPrimitive.SPHERE:
            fcl_geom = fcl.Sphere(obstacle_geometry.dimensions[SolidPrimitive.SPHERE_RADIUS])
        elif obstacle_geometry_type == SolidPrimitive.CYLINDER:
            fcl_geom = fcl.Cylinder(
                obstacle_geometry.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                obstacle_geometry.dimensions[SolidPrimitive.CYLINDER_HEIGHT],
            )
        else:
            print(f"Unsupported geometry type: {obstacle_geometry_type}")
            continue
        fcl_obj = create_fcl_object(obstacle_pose, fcl_geom)
        fcl_objects.append(fcl_obj)
    return fcl_objects


def create_fcl_object(pose: PoseStamped, geometry: fcl.CollisionGeometry) -> fcl.CollisionObject:
    q = pose.pose.orientation
    t = pose.pose.position
    R = t3d.quaternions.quat2mat([q.w, q.x, q.y, q.z])
    translation = [t.x, t.y, t.z]
    return fcl.CollisionObject(geometry, fcl.Transform(R, translation))


def check_collision(fcl_obj1, fcl_obj2) -> bool:
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    dist_request = fcl.DistanceRequest()
    dist_result = fcl.DistanceResult()
    fcl.distance(fcl_obj1, fcl_obj2, dist_request, dist_result)
    fcl.collide(fcl_obj1, fcl_obj2, request, result)
    return dist_result.min_distance < 0


def get_link_transform(link_name: str, joint_map: dict) -> sm.SE3:
    """Return transform from base to link by chaining all parent joints."""
    T = sm.SE3()
    current = link_name
    while current in joint_map:
        parent, T_joint = joint_map[current]
        T = T_joint * T
        current = parent
    return T


def se3_to_pose_stamped(se3: sm.SE3, node_obj, frame_id: str = "base_link") -> PoseStamped:
    """Convert a spatialmath SE3 object to a ROS PoseStamped message."""
    import transforms3d as t3d
    from std_msgs.msg import Header
    pose_msg = PoseStamped()
    pose_msg.header = Header()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.stamp = node_obj.get_clock().now().to_msg()
    T = se3.A
    pose_msg.pose.position.x = T[0, 3]
    pose_msg.pose.position.y = T[1, 3]
    pose_msg.pose.position.z = T[2, 3]
    quat = t3d.quaternions.mat2quat(T[:3, :3])
    pose_msg.pose.orientation.w = quat[0]
    pose_msg.pose.orientation.x = quat[1]
    pose_msg.pose.orientation.y = quat[2]
    pose_msg.pose.orientation.z = quat[3]
    return pose_msg
