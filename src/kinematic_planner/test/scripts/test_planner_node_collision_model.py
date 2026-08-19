# src/kinematic_planner/test/scripts/test_planner_node_collision_model.py
"""Exercises build_collision_fn directly, without constructing an
rclpy.Node, matching the release spec's requirement that planner-
algorithm tests not require an rclpy graph."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive

from kinematic_planner_interfaces.msg import SceneObstacles
from kinematic_planner.collision.robot_collision_model import build_link_collision_shapes
from kinematic_planner.planning.tree import TreeNode
from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.scripts.planner_node import build_collision_fn


TWO_LINK_SELF_COLLIDING_URDF = """
<robot name="two_link_test_robot">
  <link name="base_link"/>
  <link name="link_1">
    <collision><geometry><box size="0.5 0.1 0.1"/></geometry></collision>
  </link>
  <link name="link_2">
    <collision><origin xyz="0.25 0 0"/><geometry><box size="0.5 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/>
  </joint>
  <joint name="joint_2" type="fixed">
    <origin xyz="0 0 0"/>
    <parent link="link_1"/><child link="link_2"/>
  </joint>
</robot>
"""


class _FakeLink:
    def __init__(self, name):
        self.name = name


class _FakeRTBModel:
    """Two coincident links at the origin. link_2's fixed joint origin
    (xyz="0 0 0") places link_2's collision box exactly on top of
    link_1's, so the self-collision test below always detects an overlap
    regardless of q."""

    links = [_FakeLink("link_1"), _FakeLink("link_2")]

    def __init__(self):
        self.q = None

    def fkine(self, q, end, include_base=True):
        class _SE3:
            A = np.eye(4)
        return _SE3()


def _empty_obstacle_scene():
    scene = SceneObstacles()
    scene.scene_obstacles = []
    scene.obstacle_poses = []
    scene.obstacle_ids = []
    return scene


def test_check_collision_false_accepts_every_candidate():
    fn = build_collision_fn(
        robot_config=RobotConfig.from_urdf(TWO_LINK_SELF_COLLIDING_URDF),
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


def test_self_collision_between_non_adjacent_overlapping_links_is_detected():
    robot_config = RobotConfig.from_urdf(TWO_LINK_SELF_COLLIDING_URDF)
    # joint_2 is a fixed joint (no actuated DOF between link_1/link_2), and a
    # fixed-joint parent/child pair is still adjacent, so RobotConfig auto-
    # disables link_1/link_2 by adjacency. Overriding disabled_collision_pairs
    # here isolates the self-collision detection path under test from the
    # adjacency auto-disable Task 2 already covers on its own.
    robot_config.disabled_collision_pairs = []
    link_shapes = build_link_collision_shapes(ET.fromstring(TWO_LINK_SELF_COLLIDING_URDF))
    fn = build_collision_fn(
        robot_config=robot_config,
        link_shapes=link_shapes,
        obstacle_geom=_empty_obstacle_scene(),
        rtb_model=_FakeRTBModel(),
        collision_checker="bvol",
        min_obs_dist=0.0,
        check_collision=True,
    )
    node = TreeNode(np.array([0.0]))
    node.path_q = [node.q]
    assert fn(node) is False
