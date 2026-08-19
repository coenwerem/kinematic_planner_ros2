"""Informed RRT* over joint space, ROS-independent.

Reference: Gammell, Srinivasa & Barfoot, "Informed RRT*: Optimal
Sampling-based Path Planning Focused via Direct Sampling of an
Admissible Ellipsoidal Heuristic," IROS 2014.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from kinematic_planner.planning.tree import RRTPlannerBase, TreeNode, calc_new_cost


class InformedRRTStar(RRTPlannerBase):
    def __init__(
        self,
        start,
        goal,
        joint_limits: List[Tuple[float, float]],
        expand_dist: float,
        path_resolution: float,
        max_iter: int,
        connect_circle_dist: float,
        collision_fn: Callable[[TreeNode], bool],
        search_until_max_iter: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(
            dimension=len(start),
            joint_limits=joint_limits,
            expand_dist=expand_dist,
            path_resolution=path_resolution,
            connect_circle_dist=connect_circle_dist,
            collision_fn=collision_fn,
            rng=rng,
        )
        self.start = TreeNode(start)
        self.start.path_q = [np.array(start, dtype=float)]
        self.end = TreeNode(goal)
        self.end.path_q = [np.array(goal, dtype=float)]
        self.max_iter = max_iter
        self.search_until_max_iter = search_until_max_iter
        self.start_goal_collision: Optional[str] = None

        self.c_best = float("inf")
        self.c_min: Optional[float] = None
        self.q_center: Optional[np.ndarray] = None
        self.R_align: Optional[np.ndarray] = None
        self.b: Optional[float] = None

    # ---- informed sampling -----------------------------------------

    def compute_ellipse_params(self) -> None:
        self.c_min = float(np.linalg.norm(np.array(self.start.q) - np.array(self.end.q)))
        if self.c_best < float("inf"):
            val = max(self.c_best ** 2 - self.c_min ** 2, 0.0)
            self.b = float(np.sqrt(val)) / 2.0
            if not np.isfinite(self.b) or self.b <= 0.0:
                self.b = 1e-6
        else:
            self.b = None

    def _ellipse_center(self) -> np.ndarray:
        return ((np.array(self.start.q) + np.array(self.end.q)) / 2.0).reshape((-1, 1))

    def _rotation_matrix(self) -> np.ndarray:
        diff = np.array(self.end.q) - np.array(self.start.q)
        norm_diff = np.linalg.norm(diff)
        if norm_diff == 0.0:
            return np.eye(self.dimension)
        a1 = (diff / norm_diff).reshape((-1, 1))
        e1 = np.zeros((self.dimension, 1))
        e1[0, 0] = 1.0
        U, _, Vt = np.linalg.svd(a1 @ e1.T, full_matrices=True)
        S = np.eye(self.dimension)
        S[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt)
        return U @ S @ Vt

    def _sample_unit_hypersphere(self) -> np.ndarray:
        n = self.dimension
        x = self.rng.normal(size=(n,))
        norm = np.linalg.norm(x)
        if norm == 0.0:
            x = np.zeros(n)
            x[0] = 1.0
            norm = 1.0
        x = x / norm
        r = self.rng.random() ** (1.0 / n)
        return (x * r).reshape((n, 1))

    def informed_sample(self) -> np.ndarray:
        no_solution_yet = self.c_best == float("inf")
        c_min_unusable = self.c_min is None or self.c_min == float("inf")
        if no_solution_yet or c_min_unusable:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.joint_limits])

        self.q_center = self._ellipse_center()
        self.R_align = self._rotation_matrix()
        semi_minor = self.b if self.b is not None else 1e-6
        L_vec = np.zeros(self.dimension)
        L_vec[0] = self.c_best / 2.0
        L_vec[1:] = semi_minor
        L_scale = np.diag(L_vec)

        low = np.array([lo for lo, _ in self.joint_limits])
        high = np.array([hi for _, hi in self.joint_limits])
        rnd = None
        for _ in range(10):
            q_unit = self._sample_unit_hypersphere()
            q_ellipse = (self.R_align @ L_scale @ q_unit) + self.q_center
            rnd = q_ellipse.flatten()
            in_bounds = np.all(rnd >= low) and np.all(rnd <= high)
            if in_bounds:
                return rnd
        return np.clip(rnd, low, high)

    # ---- tree-specific overrides -------------------------------------

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

    def plan(self) -> Optional[List[np.ndarray]]:
        start_ok = self.collision_fn(self.start)
        end_ok = self.collision_fn(self.end)
        if not start_ok:
            self.start_goal_collision = "start"
            return None
        if not end_ok:
            self.start_goal_collision = "goal"
            return None

        best_path = None
        self.config_tree = [self.start]

        for _ in range(self.max_iter):
            rnd_arr = self.informed_sample()
            rnd_node = TreeNode(rnd_arr)
            nearest_ind = self.get_nearest_node_index(self.config_tree, rnd_node)
            new_node = self.steer(self.config_tree[nearest_ind], rnd_node)

            if new_node is not None and self.collision_fn(new_node):
                near_inds = self.get_nearby_neighbors(new_node)
                new_node = self.choose_best_parent(new_node, near_inds)
                if new_node is not None:
                    self.config_tree.append(new_node)
                    self.rewire(new_node, near_inds)

                    last_index = self.find_best_goal_node(self.end)
                    if last_index is not None:
                        temp_path = self.generate_final_course(last_index, self.end)
                        temp_cost = self.compute_path_cost(temp_path)
                        if temp_cost < self.c_best:
                            self.c_best = temp_cost
                            best_path = temp_path
                            self.compute_ellipse_params()

            found_and_can_stop = (not self.search_until_max_iter) and (best_path is not None)
            if found_and_can_stop:
                break

        return best_path
