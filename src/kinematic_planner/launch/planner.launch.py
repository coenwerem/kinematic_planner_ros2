#!/usr/bin/env python3

"""
Launch file for the standalone kinematic planner.

Brings up the nodes needed for collision-free RRT* planning:
  1. robot_state_publisher — publishes /robot_description and TF from URDF
  2. robot_geom_publisher  — publishes robot link geometry from URDF
  3. obstacle_publisher     — publishes scene obstacles
  4. joint_state_publisher — publishes the robot's start joint configuration
  5. planner_node           — runs RRT* and publishes the planned path

The robot is described entirely by the URDF xacro in
robot_3r_description.  Swap in a different URDF to use a different robot.

Usage:
    ros2 launch kinematic_planner planner.launch.py
    ros2 launch kinematic_planner planner.launch.py goal_config:="[1.5093, 0.6072, 1.4052]"
    ros2 launch kinematic_planner planner.launch.py is_dense:=true algorithm:=informed_rrt_star
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_description = FindPackageShare("robot_3r_description")
    urdf_file = PathJoinSubstitution([pkg_description, "urdf", "robot_3r.urdf.xacro"])
    robot_description = Command(["xacro ", urdf_file])

    declared_args = [
        DeclareLaunchArgument(
            "goal_config",
            default_value="[1.5093, 0.6072, 1.4052]",
            description="Goal joint configuration [q1, q2, q3] in radians.",
        ),
        DeclareLaunchArgument(
            "is_dense",
            default_value="false",
            choices=["true", "false"],
            description="Dense obstacle ring (10 obstacles) vs. sparse (2 obstacles).",
        ),
        DeclareLaunchArgument(
            "check_collision",
            default_value="true",
            choices=["true", "false"],
            description="Enable FCL collision checking during planning.",
        ),
        DeclareLaunchArgument(
            "collision_checker",
            default_value="proximity",
            choices=["proximity", "bvol"],
            description="Collision checker mode: proximity (signed distance) or bvol (bounding volume).",
        ),
        DeclareLaunchArgument(
            "rrts_max_iter",
            default_value="2000",
            description="Maximum RRT* iterations.",
        ),
        DeclareLaunchArgument(
            "verbose",
            default_value="false",
            choices=["true", "false"],
            description="Print per-iteration planning logs.",
        ),
        DeclareLaunchArgument(
            "algorithm",
            default_value="rrt_star",
            choices=["rrt_star", "informed_rrt_star"],
            description="Which planner node to launch.",
        ),
        DeclareLaunchArgument(
            "world_frame",
            default_value="world",
            description="World / fixed frame name.",
        ),
        DeclareLaunchArgument(
            "base_link_name",
            default_value="base_link",
            description="Robot base link name.",
        ),
        # Dense-mode obstacle ring parameters (override to match your robot)
        DeclareLaunchArgument(
            "dense_platform_height",
            default_value="0.755",
            description="Height of the robot's first link (used for dense obstacle ring placement).",
        ),
        DeclareLaunchArgument(
            "dense_ring_radius",
            default_value="0.45",
            description="Radius of the dense obstacle ring in metres.",
        ),
    ]

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen",
    )

    robot_geom_publisher = Node(
        package="kinematic_planner",
        executable="robot_geom_publisher",
        name="robot_geom_publisher",
        parameters=[{
            "robot_description": robot_description,
            "base_link_name": LaunchConfiguration("base_link_name"),
            "world_frame": LaunchConfiguration("world_frame"),
        }],
        output="screen",
    )

    obstacle_publisher = Node(
        package="kinematic_planner",
        executable="obstacle_publisher",
        name="obstacle_publisher",
        parameters=[{
            "is_dense": LaunchConfiguration("is_dense"),
            "world_frame": LaunchConfiguration("world_frame"),
            "base_link_name": LaunchConfiguration("base_link_name"),
            "dense_platform_height": LaunchConfiguration("dense_platform_height"),
            "dense_ring_radius": LaunchConfiguration("dense_ring_radius"),
        }],
        output="screen",
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen",
    )

    planner_node = Node(
        package="kinematic_planner",
        executable="planner_node",
        name="sampling_based_planner",
        parameters=[{
            "robot_description": robot_description,
            "goal_config": LaunchConfiguration("goal_config"),
            "check_collision": LaunchConfiguration("check_collision"),
            "collision_checker": LaunchConfiguration("collision_checker"),
            "rrts_max_iter": LaunchConfiguration("rrts_max_iter"),
            "verbose": LaunchConfiguration("verbose"),
            "world_frame": LaunchConfiguration("world_frame"),
            "base_link_name": LaunchConfiguration("base_link_name"),
        }],
        output="screen",
        condition=LaunchConfigurationEquals("algorithm", "rrt_star"),
    )

    informed_rrt_star_node = Node(
        package="kinematic_planner",
        executable="informed_rrt_star_node",
        name="informed_rrt_star",
        parameters=[{
            "robot_description": robot_description,
            "goal_config": LaunchConfiguration("goal_config"),
            "check_collision": LaunchConfiguration("check_collision"),
            "collision_checker": LaunchConfiguration("collision_checker"),
            "rrts_max_iter": LaunchConfiguration("rrts_max_iter"),
            "verbose": LaunchConfiguration("verbose"),
            "world_frame": LaunchConfiguration("world_frame"),
            "base_link_name": LaunchConfiguration("base_link_name"),
        }],
        output="screen",
        condition=LaunchConfigurationEquals("algorithm", "informed_rrt_star"),
    )

    return LaunchDescription(declared_args + [
        robot_state_publisher,
        robot_geom_publisher,
        obstacle_publisher,
        joint_state_publisher,
        planner_node,
        informed_rrt_star_node,
    ])
