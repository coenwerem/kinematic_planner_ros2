import numpy as np
from kinematic_planner.planning.tree import TreeNode, edge_distance, calc_new_cost


def test_edge_distance_is_euclidean_norm():
    a = TreeNode(np.array([0.0, 0.0, 0.0]))
    b = TreeNode(np.array([3.0, 4.0, 0.0]))
    assert edge_distance(a, b) == 5.0


def test_calc_new_cost_adds_edge_distance_to_from_node_cost():
    a = TreeNode(np.array([0.0, 0.0]))
    a.cost = 10.0
    b = TreeNode(np.array([3.0, 4.0]))
    # calc_new_cost must return from_node.cost (10) + edge_distance (5) = 15,
    # not edge_distance (5) alone.
    assert calc_new_cost(a, b) == 15.0
