#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution
)
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
import os
from ament_index_python import get_package_share_directory

def generate_launch_description():
    launch_rviz = LaunchConfiguration("launch_rviz")
    log_level = LaunchConfiguration("log_level")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    description_package = LaunchConfiguration("description_package")
    description_filepath = LaunchConfiguration("description_filepath")

    # URDF via xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([FindPackageShare(description_package), description_filepath])
    ])

    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }


    declared_arguments = [
        DeclareLaunchArgument(
            "description_package",
            default_value="robot_3r_description",
            description="Package with robot description.",
        ),
        DeclareLaunchArgument(
            "description_filepath",
            default_value=os.path.join("urdf", "robot_3r.urdf.xacro"),
            description="Path to URDF/Xacro file, relative to share of `description_package`.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulated time.",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="warn",
            description="ROS 2 log level.",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz2 with the robot model and TFs."
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=os.path.join(
                get_package_share_directory("robot_3r_description"),
                "rviz",
                "view_3r.rviz",
            ),
            description="Path to configuration for RViz2.",
        )]
    
    return LaunchDescription(declared_arguments + [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                robot_description,
                {"publish_frequency": 100.0, "frame_prefix": "", "use_sim_time": use_sim_time},
            ],
        ),
        # rviz2
        Node(
            package="rviz2",
            condition=IfCondition(launch_rviz),
            executable="rviz2",
            output="log",
            arguments=[
                "--display-config",
                rviz_config,
                "--ros-args",
                "--log-level",
                log_level,
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ])