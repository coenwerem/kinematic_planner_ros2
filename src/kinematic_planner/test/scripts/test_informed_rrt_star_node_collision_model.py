# src/kinematic_planner/test/scripts/test_informed_rrt_star_node_collision_model.py
"""Exercises build_collision_fn through the informed_rrt_star_node import
path, without constructing an rclpy.Node."""
import numpy as np

from robot_3r_interfaces.msg import SceneObstacles
from kinematic_planner.planning.tree import TreeNode
from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.scripts.informed_rrt_star_node import build_collision_fn


def _empty_obstacle_scene():
    scene = SceneObstacles()
    scene.scene_obstacles = []
    scene.obstacle_poses = []
    scene.obstacle_ids = []
    return scene


def test_check_collision_false_accepts_every_candidate():
    fn = build_collision_fn(
        robot_config=RobotConfig.from_urdf(
            '<robot name="r"><link name="base_link"/></robot>'
        ),
        link_shapes={},
        obstacle_geom=_empty_obstacle_scene(),
        rtb_model=None,
        collision_checker="proximity",
        min_obs_dist=0.1,
        check_collision=False,
    )
    node = TreeNode(np.array([0.0]))
    node.path_q = [node.q]
    assert fn(node) is True
