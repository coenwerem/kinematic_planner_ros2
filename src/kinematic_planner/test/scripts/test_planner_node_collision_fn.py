# src/kinematic_planner/test/scripts/test_planner_node_collision_fn.py
"""Exercises build_collision_fn directly, without constructing an
rclpy.Node, matching the release spec's requirement that planner-
algorithm tests not require an rclpy graph."""
from kinematic_planner.scripts.planner_node import build_collision_fn
from kinematic_planner.planning.tree import TreeNode
import numpy as np


class _FakeRobotConfig:
    base_link_name = "base_link"
    world_frame = "world"


def test_check_collision_false_accepts_every_candidate():
    fn = build_collision_fn(
        robot_config=_FakeRobotConfig(),
        robot_geom=None,
        obstacle_geom=None,
        rtb_model=None,
        collision_checker="proximity",
        min_obs_dist=0.1,
        check_collision=False,
    )
    node = TreeNode(np.array([0.0, 0.0, 0.0]))
    node.path_q = [node.q]
    assert fn(node) is True


class _RaisingLink:
    name = "link1"


class _RaisingRTBModel:
    """Stands in for an rtb_model whose fkine() raises, exercising the
    per-link fail-safe: an exception during FK/geometry/FCL work must
    make collision_fn return False rather than propagate."""

    links = [_RaisingLink()]

    def fkine(self, q, end=None, include_base=True):
        raise RuntimeError("fk failed")


class _FakeRobotGeom:
    link_names = ["link1"]
    link_geometries = [None]


class _FakeObstacleGeom:
    obstacle_ids = []


class _LoggerSpy:
    def __init__(self):
        self.errors = []

    def error(self, msg):
        self.errors.append(msg)


def test_collision_fn_returns_false_and_logs_on_per_link_exception():
    logger = _LoggerSpy()
    fn = build_collision_fn(
        robot_config=_FakeRobotConfig(),
        robot_geom=_FakeRobotGeom(),
        obstacle_geom=_FakeObstacleGeom(),
        rtb_model=_RaisingRTBModel(),
        collision_checker="proximity",
        min_obs_dist=0.1,
        check_collision=True,
        get_logger=lambda: logger,
    )
    node = TreeNode(np.array([0.0, 0.0, 0.0]))
    node.path_q = [node.q]
    assert fn(node) is False
    assert logger.errors
