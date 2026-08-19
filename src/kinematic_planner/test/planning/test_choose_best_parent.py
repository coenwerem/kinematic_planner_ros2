import numpy as np
from kinematic_planner.planning.tree import RRTPlannerBase, TreeNode


def _always_free(_node):
    return True


def test_fallback_branch_does_not_double_count_parent_cost():
    base = RRTPlannerBase(
        dimension=1,
        joint_limits=[(-100.0, 100.0)],
        expand_dist=10.0,
        path_resolution=1.0,
        connect_circle_dist=50.0,
        collision_fn=_always_free,
    )
    root = TreeNode(np.array([0.0]))
    root.cost = 10.0
    base.config_tree = [root]

    rnd = TreeNode(np.array([3.0]))
    new_node = base.steer(root, rnd)
    result = base.choose_best_parent(new_node, near_inds=[])

    assert result is not None
    assert result.parent is root
    # correct: root.cost (10) plus edge distance (3) equals 13.
    # a double-counted result would read 10 + (10 + 3) = 23.
    assert result.cost == 13.0


def test_near_inds_branch_picks_minimum_cost_parent():
    base = RRTPlannerBase(
        dimension=1,
        joint_limits=[(-100.0, 100.0)],
        expand_dist=10.0,
        path_resolution=1.0,
        connect_circle_dist=50.0,
        collision_fn=_always_free,
    )
    cheap_node = TreeNode(np.array([0.0]))
    cheap_node.cost = 1.0
    expensive_node = TreeNode(np.array([0.0]))
    expensive_node.cost = 100.0
    base.config_tree = [cheap_node, expensive_node]

    rnd = TreeNode(np.array([3.0]))
    new_node = base.steer(cheap_node, rnd)
    result = base.choose_best_parent(new_node, near_inds=[0, 1])

    assert result.parent is cheap_node
    assert result.cost == 4.0  # cheap_node.cost (1) plus edge distance (3)
