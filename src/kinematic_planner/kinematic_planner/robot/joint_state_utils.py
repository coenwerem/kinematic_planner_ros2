#!/usr/bin/env python3

"""JointState name-based remapping. A JointState publisher can order
`position` in any sequence. The planner requires `RobotConfig.joint_names`
order. remap_joint_state converts between the two by matching each requested
joint name to its position in the message.
"""

from typing import List

import numpy as np
from sensor_msgs.msg import JointState


def remap_joint_state(msg: JointState, joint_names: List[str]) -> np.ndarray:
    if len(msg.name) != len(msg.position):
        raise ValueError(
            f"JointState message has {len(msg.name)} names but {len(msg.position)} positions; "
            f"the two arrays must be the same length."
        )
    name_to_position = dict(zip(msg.name, msg.position))
    missing = [name for name in joint_names if name not in name_to_position]
    if missing:
        raise ValueError(
            f"JointState message is missing required joints: {missing}. "
            f"Message carried names: {list(msg.name)}."
        )
    return np.array([name_to_position[name] for name in joint_names], dtype=float)
