"""RRT*/Informed RRT* tree bookkeeping. Imports none of rclpy, fcl, or
ROS message types.

Invariant: node.cost equals the accumulated cost from the tree root to
node along the parent chain. calc_new_cost returns the full accumulated
value (from_node.cost + edge_distance). Every call site assigns
calc_new_cost's return value directly to a node's .cost field.
"""

import math
from typing import Callable, List, Optional, Tuple

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


class RRTPlannerBase:
    """Shared tree bookkeeping for RRT*-family planners: steer,
    nearest/near-neighbor queries, cost propagation. A subclass supplies
    sample_free()/informed_sample() plus a plan() loop.
    """

    def __init__(
        self,
        dimension: int,
        joint_limits: List[Tuple[float, float]],
        expand_dist: float,
        path_resolution: float,
        connect_circle_dist: float,
        collision_fn: Callable[[TreeNode], bool],
        rng: Optional[np.random.Generator] = None,
    ):
        self.dimension = dimension
        self.joint_limits = joint_limits
        self.expand_dist = expand_dist
        self.path_resolution = path_resolution
        self.connect_circle_dist = connect_circle_dist
        self.collision_fn = collision_fn
        self.rng = rng if rng is not None else np.random.default_rng()
        self.config_tree: List[TreeNode] = []

    @staticmethod
    def get_nearest_node_index(config_tree: List[TreeNode], rnd_node: TreeNode) -> int:
        dists = [np.sum((node.q - rnd_node.q) ** 2) for node in config_tree]
        return int(np.argmin(dists))

    def get_nearby_neighbors(self, x_new: TreeNode) -> List[int]:
        assert self.connect_circle_dist > 2 * (1 + 1 / self.dimension) ** (1 / self.dimension), \
            "connect_circle_dist too small for the RRT* near-radius formula"
        n = len(self.config_tree)
        if n <= 1:
            near_radius = self.expand_dist
        else:
            near_radius = self.connect_circle_dist * (math.log(n) / n) ** (1.0 / self.dimension)
            if self.expand_dist:
                near_radius = min(near_radius, self.expand_dist)
        dists = [np.sum((nd.q - x_new.q) ** 2) for nd in self.config_tree]
        return [i for i, d in enumerate(dists) if d <= near_radius ** 2]

    def steer(self, x_nearest: TreeNode, x_random: TreeNode) -> Optional[TreeNode]:
        start = np.array(x_nearest.q, dtype=float)
        goal = np.array(x_random.q, dtype=float)
        full_dist = float(np.linalg.norm(goal - start))
        if full_dist == 0.0:
            return None
        new_node = TreeNode(start.copy())
        new_node.path_q = [start.copy()]
        extend_length = min(self.expand_dist, full_dist)
        n_expand = max(1, math.floor(extend_length / self.path_resolution))
        unit_vec = (goal - start) / full_dist
        for _ in range(n_expand):
            new_node.q = new_node.q + unit_vec * self.path_resolution
            new_node.path_q.append(new_node.q.copy())
        remaining_dist = float(np.linalg.norm(goal - new_node.q))
        if remaining_dist <= self.path_resolution:
            new_node.q = goal.copy()
            new_node.path_q.append(goal.copy())
        new_node.parent = x_nearest
        return new_node

    def choose_best_parent(self, new_node: TreeNode, near_inds: List[int]) -> Optional[TreeNode]:
        if not near_inds:
            nearest_idx = self.get_nearest_node_index(self.config_tree, new_node)
            nearest = self.config_tree[nearest_idx]
            cand = self.steer(nearest, new_node)
            if cand is not None and self.collision_fn(cand):
                cand.parent = nearest
                cand.cost = calc_new_cost(nearest, cand)
                return cand
            return None

        candidates = []
        for i in near_inds:
            near_node = self.config_tree[i]
            cand = self.steer(near_node, new_node)
            if cand is not None and self.collision_fn(cand):
                candidates.append((calc_new_cost(near_node, cand), cand))
            else:
                candidates.append((float("inf"), None))

        min_cost, best_cand = min(candidates, key=lambda x: x[0])
        if min_cost == float("inf"):
            return None
        best_cand.cost = min_cost
        return best_cand
