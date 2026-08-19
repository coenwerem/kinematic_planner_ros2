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

    def rewire(self, new_node: TreeNode, near_inds: List[int]) -> None:
        for i in near_inds:
            near_node = self.config_tree[i]
            if near_node is new_node:
                continue
            cand = self.steer(new_node, near_node)
            if cand is None or not self.collision_fn(cand):
                continue
            new_cost = calc_new_cost(new_node, near_node)
            if new_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = new_cost
                near_node.path_q = cand.path_q
                self.propagate_cost_to_leaves(near_node)

    def propagate_cost_to_leaves(self, parent_node: TreeNode) -> None:
        for node in self.config_tree:
            if node.parent is parent_node:
                node.cost = calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def find_best_goal_node(self, end_node: TreeNode) -> Optional[int]:
        dist_to_goal = [edge_distance(nd, end_node) for nd in self.config_tree]
        goal_inds = [i for i, d in enumerate(dist_to_goal) if d <= self.expand_dist]
        collision_free_goal_inds = []
        for gi in goal_inds:
            cand = self.steer(self.config_tree[gi], end_node)
            if cand is not None and self.collision_fn(cand):
                collision_free_goal_inds.append(gi)
        if not collision_free_goal_inds:
            return None
        return min(collision_free_goal_inds, key=lambda i: self.config_tree[i].cost)

    def generate_final_course(self, goal_ind: int, end_node: TreeNode) -> List[np.ndarray]:
        path = [end_node.q]
        node = self.config_tree[goal_ind]
        while node.parent is not None:
            path.append(node.q)
            node = node.parent
        path.append(node.q)
        path.reverse()
        return path

    @staticmethod
    def compute_path_cost(path: List[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0
        arr = np.array(path)
        return float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
