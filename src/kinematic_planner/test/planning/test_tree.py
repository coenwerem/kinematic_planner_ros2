import numpy as np
from kinematic_planner.planning.tree import TreeNode, edge_distance, calc_new_cost, RRTPlannerBase


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


def _always_free(_node):
    return True


def _make_base(dimension=2, expand_dist=1.0, path_resolution=0.25):
    return RRTPlannerBase(
        dimension=dimension,
        joint_limits=[(-10.0, 10.0)] * dimension,
        expand_dist=expand_dist,
        path_resolution=path_resolution,
        connect_circle_dist=50.0,
        collision_fn=_always_free,
    )


def test_steer_stops_at_expand_dist_when_target_is_farther():
    base = _make_base(expand_dist=1.0, path_resolution=0.25)
    start = TreeNode(np.array([0.0, 0.0]))
    target = TreeNode(np.array([10.0, 0.0]))
    steered = base.steer(start, target)
    assert steered is not None
    assert np.isclose(edge_distance(start, steered), 1.0, atol=1e-9)
    assert steered.parent is start


def test_steer_reaches_target_exactly_when_within_expand_dist():
    base = _make_base(expand_dist=5.0, path_resolution=0.25)
    start = TreeNode(np.array([0.0, 0.0]))
    target = TreeNode(np.array([1.0, 0.0]))
    steered = base.steer(start, target)
    assert steered is not None
    assert np.allclose(steered.q, target.q)


def test_get_nearest_node_index_picks_closest():
    base = _make_base()
    node_a = TreeNode(np.array([0.0, 0.0]))
    node_b = TreeNode(np.array([5.0, 5.0]))
    base.config_tree = [node_a, node_b]
    rnd = TreeNode(np.array([4.9, 4.9]))
    assert base.get_nearest_node_index(base.config_tree, rnd) == 1


def test_get_nearby_neighbors_respects_expand_dist_cap():
    base = _make_base(expand_dist=2.0)
    node_near_1 = TreeNode(np.array([0.0, 0.0]))
    node_near_2 = TreeNode(np.array([1.0, 0.0]))
    node_far = TreeNode(np.array([9.0, 0.0]))
    base.config_tree = [node_near_1, node_near_2, node_far]
    x_new = TreeNode(np.array([0.0, 0.0]))
    near = base.get_nearby_neighbors(x_new)
    assert 0 in near
    assert 1 in near
    assert 2 not in near
