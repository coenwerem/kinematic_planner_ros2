"""Bring up move_group for the xArm7 with all four planning pipelines loaded.

The benchmark node talks to this move_group over the /plan_kinematic_path
service, selecting a pipeline and planner per request, so every planner runs
against one robot model, one planning scene, and one set of joint limits.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PIPELINES = ["ompl", "chomp", "stomp", "pilz_industrial_motion_planner"]


def generate_launch_description():
    urdf = os.path.join(
        get_package_share_directory("xarm7_description"), "urdf", "xarm7.urdf"
    )

    moveit_config = (
        MoveItConfigsBuilder("xarm7", package_name="xarm7_moveit_config")
        .robot_description(file_path=urdf)
        .robot_description_semantic(file_path="config/xarm7.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .planning_pipelines(pipelines=PIPELINES, default_planning_pipeline="ompl")
        .to_moveit_configs()
    )

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="log",
                parameters=[moveit_config.robot_description],
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                output="log",
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {"publish_robot_description_semantic": True,
                     "moveit_manage_controllers": False},
                ],
            ),
        ]
    )
