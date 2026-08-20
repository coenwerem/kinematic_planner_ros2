import numpy as np
from kinematic_planner.planning.tree import RRTPlannerBase, TreeNode


def _always_free(_node):
    return True


def _make_base(expand_dist=10.0, path_resolution=1.0):
    return RRTPlannerBase(
        dimension=1,
        joint_limits=[(-100.0, 100.0)],
        expand_dist=expand_dist,
        path_resolution=path_resolution,
        connect_circle_dist=50.0,
        collision_fn=_always_free,
    )


def test_rewire_does_not_double_count_new_node_cost():
    base = _make_base()
    new_node = TreeNode(np.array([0.0]))
    new_node.cost = 5.0
    near_node = TreeNode(np.array([1.0]))
    near_node.cost = 100.0  # deliberately expensive so rewiring is the correct choice
    base.config_tree = [new_node, near_node]

    base.rewire(new_node, near_inds=[1])

    # correct: new_node.cost (5) plus edge distance (1) equals 6.
    # planner_node.py's original bug left near_node.cost at 0.0 (never assigned).
    # informed_rrt_star_node.py's original bug produced 5 + (5 + 1) = 11.
    assert near_node.cost == 6.0
    assert near_node.parent is new_node


def test_rewire_leaves_cost_unchanged_when_not_improving():
    base = _make_base()
    new_node = TreeNode(np.array([0.0]))
    new_node.cost = 50.0
    near_node = TreeNode(np.array([1.0]))
    near_node.cost = 2.0  # already cheaper than rewiring through new_node would produce
    base.config_tree = [new_node, near_node]

    base.rewire(new_node, near_inds=[1])

    assert near_node.cost == 2.0
    assert near_node.parent is None


def test_rewire_propagates_cost_to_preexisting_grandchildren():
    base = _make_base()
    new_node = TreeNode(np.array([0.0]))
    new_node.cost = 1.0

    near_node = TreeNode(np.array([1.0]))
    near_node.cost = 100.0

    grandchild = TreeNode(np.array([2.0]))
    grandchild.parent = near_node
    near_node.children.append(grandchild)
    grandchild.cost = 101.0  # stale near_node.cost (100) plus edge distance (1)

    base.config_tree = [new_node, near_node, grandchild]

    base.rewire(new_node, near_inds=[1])

    # near_node.cost becomes new_node.cost (1) plus edge distance (1) = 2.
    # in-place mutation keeps grandchild.parent pointed at the same
    # near_node object, so propagate_cost_to_leaves recomputes
    # grandchild.cost from the updated near_node.cost.
    assert near_node.cost == 2.0
    assert grandchild.parent is near_node
    assert grandchild.cost == 3.0  # near_node.cost (2) plus edge distance (1)
