import numpy as np
from kinematic_planner.planning.tree import RRTPlannerBase, TreeNode


def _always_free(_node):
    return True


def _make_base():
    return RRTPlannerBase(
        dimension=1,
        joint_limits=[(-100.0, 100.0)],
        expand_dist=5.0,
        path_resolution=1.0,
        connect_circle_dist=50.0,
        collision_fn=_always_free,
    )


def test_find_best_goal_node_does_not_collapse_duplicate_distances():
    base = _make_base()
    end_node = TreeNode(np.array([10.0]))

    # two distinct nodes, both at distance 4 from end_node (within expand_dist=5),
    # so the per-node distance-to-goal list holds a duplicate value; list.index()
    # would collapse both candidates onto whichever appears first.
    cheap_node = TreeNode(np.array([6.0]))
    cheap_node.cost = 1.0
    expensive_node = TreeNode(np.array([6.0]))
    expensive_node.cost = 50.0
    base.config_tree = [expensive_node, cheap_node]  # expensive_node listed first, on purpose

    best_ind = base.find_best_goal_node(end_node)

    assert best_ind == 1  # must resolve to cheap_node, the actually-cheapest candidate


def test_find_best_goal_node_returns_none_when_nothing_in_range():
    base = _make_base()
    end_node = TreeNode(np.array([1000.0]))
    far_node = TreeNode(np.array([0.0]))
    base.config_tree = [far_node]
    assert base.find_best_goal_node(end_node) is None


def test_generate_final_course_and_compute_path_cost():
    base = _make_base()
    root = TreeNode(np.array([0.0]))
    root.cost = 0.0
    mid_node = TreeNode(np.array([3.0]))
    mid_node.parent = root
    mid_node.cost = 3.0
    base.config_tree = [root, mid_node]

    end_node = TreeNode(np.array([7.0]))
    path = base.generate_final_course(goal_ind=1, end_node=end_node)

    assert [p.tolist() for p in path] == [[0.0], [3.0], [7.0]]
    assert base.compute_path_cost(path) == 7.0  # edge lengths 3 plus 4
