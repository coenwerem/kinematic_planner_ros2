"""RRT* over joint space, ROS-independent.

Reference: Karaman & Frazzoli, "Sampling-based algorithms for optimal
motion planning," IJRR 2011.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from kinematic_planner.planning.tree import RRTPlannerBase, TreeNode


class RRTStar(RRTPlannerBase):
    def __init__(
        self,
        start,
        goal,
        joint_limits: List[Tuple[float, float]],
        expand_dist: float,
        path_resolution: float,
        max_iter: int,
        connect_circle_dist: float,
        goal_sample_rate: float,
        collision_fn: Callable[[TreeNode], bool],
        use_goal_biased_sampling: bool = False,
        goal_noise_sigma: float = 0.05,
        search_until_max_iter: bool = False,
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
        self.goal_sample_rate = goal_sample_rate
        self.use_goal_biased_sampling = use_goal_biased_sampling
        self.goal_noise_sigma = goal_noise_sigma
        self.search_until_max_iter = search_until_max_iter
        self.start_goal_collision: Optional[str] = None

    def sample_free(self) -> TreeNode:
        if self.use_goal_biased_sampling and self.rng.random() <= self.goal_sample_rate:
            goal = np.array(self.end.q)
            noise = self.rng.normal(scale=self.goal_noise_sigma, size=goal.shape)
            lo = np.array([jl[0] for jl in self.joint_limits])
            hi = np.array([jl[1] for jl in self.joint_limits])
            return TreeNode(np.clip(goal + noise, lo, hi))
        samp = [self.rng.uniform(lo, hi) for lo, hi in self.joint_limits]
        return TreeNode(np.array(samp))

    def plan(
        self,
        on_iteration: Optional[Callable[[int, float], None]] = None,
    ) -> Optional[List[np.ndarray]]:
        start_ok = self.collision_fn(self.start)
        end_ok = self.collision_fn(self.end)
        if not start_ok:
            self.start_goal_collision = "start"
            return None
        if not end_ok:
            self.start_goal_collision = "goal"
            return None

        self.config_tree = [self.start]
        best_cost = float("inf")
        for i in range(self.max_iter):
            rnd_node = self.sample_free()
            nearest_ind = self.get_nearest_node_index(self.config_tree, rnd_node)
            new_node = self.steer(self.config_tree[nearest_ind], rnd_node)
            added = False
            if new_node is not None and self.collision_fn(new_node):
                near_inds = self.get_nearby_neighbors(new_node)
                new_node = self.choose_best_parent(new_node, near_inds)
                if new_node is not None:
                    self.config_tree.append(new_node)
                    self.rewire(new_node, near_inds)
                    added = True

            if on_iteration is not None:
                if added:
                    last_index = self.find_best_goal_node(self.end)
                    if last_index is not None:
                        path = self.generate_final_course(last_index, self.end)
                        best_cost = min(best_cost, self.compute_path_cost(path))
                on_iteration(i, best_cost)

            if not self.search_until_max_iter:
                last_index = self.find_best_goal_node(self.end)
                if last_index is not None:
                    return self.generate_final_course(last_index, self.end)

        last_index = self.find_best_goal_node(self.end)
        if last_index is not None:
            return self.generate_final_course(last_index, self.end)
        return None
