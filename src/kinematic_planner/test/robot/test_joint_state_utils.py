import numpy as np
import pytest
from sensor_msgs.msg import JointState

from kinematic_planner.robot.joint_state_utils import remap_joint_state


def test_remap_orders_positions_by_requested_joint_names_regardless_of_message_order():
    msg = JointState()
    msg.name = ["joint_3", "joint_1", "joint_2"]
    msg.position = [30.0, 10.0, 20.0]
    result = remap_joint_state(msg, joint_names=["joint_1", "joint_2", "joint_3"])
    assert np.allclose(result, [10.0, 20.0, 30.0])


def test_remap_ignores_extra_joints_not_requested():
    msg = JointState()
    msg.name = ["gripper_finger", "joint_1", "joint_2"]
    msg.position = [0.0, 1.5, 2.5]
    result = remap_joint_state(msg, joint_names=["joint_1", "joint_2"])
    assert np.allclose(result, [1.5, 2.5])


def test_remap_raises_value_error_naming_every_missing_joint():
    msg = JointState()
    msg.name = ["joint_1"]
    msg.position = [1.5]
    with pytest.raises(ValueError, match="joint_2"):
        remap_joint_state(msg, joint_names=["joint_1", "joint_2", "joint_3"])


def test_remap_raises_value_error_on_name_position_length_mismatch():
    msg = JointState()
    msg.name = ["joint_1", "joint_2"]
    msg.position = [1.0]
    with pytest.raises(ValueError, match="2 names but 1 positions"):
        remap_joint_state(msg, joint_names=["joint_1", "joint_2"])
