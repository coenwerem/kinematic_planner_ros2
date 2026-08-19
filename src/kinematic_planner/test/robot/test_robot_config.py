# src/kinematic_planner/test/robot/test_robot_config.py
"""RobotConfig.from_urdf's base_link_name resolution, both the auto-derive
default and the explicit override needed for URDFs with a separate fixed
root link above the robot's own base."""
from kinematic_planner.robot.robot_config import RobotConfig


WORLD_ROOTED_URDF = """
<robot name="world_rooted_test_robot">
  <link name="world"/>
  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="link_1">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <joint name="base_joint" type="fixed">
    <parent link="world"/><child link="base_link"/>
  </joint>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/>
  </joint>
</robot>
"""

SELF_ROOTED_URDF = """
<robot name="self_rooted_test_robot">
  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="link_1">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/>
  </joint>
</robot>
"""


def test_auto_derived_base_link_name_resolves_to_world_root_when_present():
    # Documents the known limitation of the no-parent-joint heuristic: a
    # URDF with a separate fixed root link above the robot's own base
    # resolves base_link_name to that root, not the robot's base, unless
    # the caller passes base_link_name explicitly.
    config = RobotConfig.from_urdf(WORLD_ROOTED_URDF)
    assert config.base_link_name == "world"


def test_explicit_base_link_name_overrides_the_auto_derive_heuristic():
    config = RobotConfig.from_urdf(WORLD_ROOTED_URDF, base_link_name="base_link")
    assert config.base_link_name == "base_link"


def test_auto_derived_base_link_name_is_correct_when_robot_base_is_the_root():
    config = RobotConfig.from_urdf(SELF_ROOTED_URDF)
    assert config.base_link_name == "base_link"
