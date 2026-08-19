import fcl
import numpy as np
import pytest

from kinematic_planner.collision.self_collision import check_self_collision
from kinematic_planner.robot.robot_config import RobotConfig


ADJACENT_LINKS_URDF = """
<robot name="test_robot">
  <link name="base_link"/>
  <link name="link_1"><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="link_2"><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <limit lower="-3.14" upper="3.14"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link_1"/><child link="link_2"/>
    <limit lower="-3.14" upper="3.14"/>
  </joint>
</robot>
"""


def test_directly_connected_links_are_auto_disabled():
    config = RobotConfig.from_urdf(ADJACENT_LINKS_URDF)
    pairs = config.get_collision_pairs()
    assert ("link_1", "link_2") not in pairs
    assert ("link_2", "link_1") not in pairs


def test_explicit_disabled_pairs_merge_with_adjacency_auto_disable():
    config = RobotConfig.from_urdf(ADJACENT_LINKS_URDF, disabled_pairs=[("base_link", "link_2")])
    pairs = config.get_collision_pairs()
    assert ("base_link", "link_2") not in pairs
    assert ("link_1", "link_2") not in pairs  # adjacency auto-disable still applies


def _boxes_at(position):
    obj = fcl.CollisionObject(fcl.Box(0.2, 0.2, 0.2), fcl.Transform(np.eye(3), position))
    return [obj]


def test_check_self_collision_flags_overlapping_non_adjacent_links():
    link_objs = {
        "link_a": _boxes_at([0.0, 0.0, 0.0]),
        "link_b": _boxes_at([0.1, 0.0, 0.0]),  # 0.2-side boxes 0.1 apart: overlapping
    }
    assert check_self_collision(link_objs, collision_pairs=[("link_a", "link_b")]) is True


def test_check_self_collision_clears_separated_links():
    link_objs = {
        "link_a": _boxes_at([0.0, 0.0, 0.0]),
        "link_b": _boxes_at([5.0, 0.0, 0.0]),
    }
    assert check_self_collision(link_objs, collision_pairs=[("link_a", "link_b")]) is False


def test_check_self_collision_ignores_disabled_pair_even_when_overlapping():
    link_objs = {
        "link_a": _boxes_at([0.0, 0.0, 0.0]),
        "link_b": _boxes_at([0.1, 0.0, 0.0]),
    }
    # link_a/link_b overlap, but the pair is simply absent from collision_pairs,
    # matching how RobotConfig.get_collision_pairs() excludes disabled pairs upstream
    assert check_self_collision(link_objs, collision_pairs=[]) is False
