# src/kinematic_planner/test/scripts/test_informed_rrt_star_node_collision_fn.py
from kinematic_planner.scripts.informed_rrt_star_node import build_collision_fn
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
