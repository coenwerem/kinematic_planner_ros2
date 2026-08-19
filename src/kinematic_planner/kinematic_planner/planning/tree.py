"""RRT*/Informed RRT* tree bookkeeping. Imports none of rclpy, fcl, or
ROS message types.

Invariant: node.cost equals the accumulated cost from the tree root to
node along the parent chain. calc_new_cost returns the full accumulated
value (from_node.cost + edge_distance). Every call site assigns
calc_new_cost's return value directly to a node's .cost field.
"""

from typing import List, Optional

import numpy as np


class TreeNode:
    def __init__(self, q):
        self.q = np.array(q, dtype=float)
        self.parent: Optional["TreeNode"] = None
        self.cost: float = 0.0
        self.path_q: List[np.ndarray] = []


def edge_distance(from_node: TreeNode, to_node: TreeNode) -> float:
    return float(np.linalg.norm(to_node.q - from_node.q))


def calc_new_cost(from_node: TreeNode, to_node: TreeNode) -> float:
    """Accumulated cost of to_node under the parent assignment to_node.parent = from_node."""
    return from_node.cost + edge_distance(from_node, to_node)
