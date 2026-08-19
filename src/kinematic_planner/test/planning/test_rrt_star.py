import numpy as np
from kinematic_planner.planning.rrt_star import RRTStar


def _always_free(_node):
    return True


def test_plan_finds_a_path_in_open_space_with_correct_cost_invariant():
    rng = np.random.default_rng(0)
    planner = RRTStar(
        start=[0.0, 0.0],
        goal=[5.0, 0.0],
        joint_limits=[(-10.0, 10.0), (-10.0, 10.0)],
        expand_dist=1.0,
        path_resolution=0.25,
        max_iter=500,
        connect_circle_dist=20.0,
        goal_sample_rate=0.3,
        collision_fn=_always_free,
        use_goal_biased_sampling=True,
        search_until_max_iter=False,
        rng=rng,
    )
    path = planner.plan()
    assert path is not None
    assert np.allclose(path[0], [0.0, 0.0])
    assert np.allclose(path[-1], [5.0, 0.0])

    # cost invariant: every tree node's cost equals the sum of edge
    # lengths from the tree root to the node, walked via node.parent.
    for node in planner.config_tree:
        chain_cost = 0.0
        cur = node
        while cur.parent is not None:
            chain_cost += float(np.linalg.norm(cur.q - cur.parent.q))
            cur = cur.parent
        assert np.isclose(node.cost, chain_cost, atol=1e-9), \
            f"node at {node.q} has cost {node.cost}, expected {chain_cost}"


def test_plan_returns_none_and_flags_start_collision():
    def _reject_start(node):
        return not np.allclose(node.q, [0.0, 0.0])

    planner = RRTStar(
        start=[0.0, 0.0],
        goal=[5.0, 0.0],
        joint_limits=[(-10.0, 10.0), (-10.0, 10.0)],
        expand_dist=1.0,
        path_resolution=0.25,
        max_iter=10,
        connect_circle_dist=20.0,
        goal_sample_rate=0.3,
        collision_fn=_reject_start,
        rng=np.random.default_rng(0),
    )
    assert planner.plan() is None
    assert planner.start_goal_collision == "start"


def test_collision_fn_always_true_produces_a_plan():
    # RRTStar takes only collision_fn, with no separate check_collision
    # boolean to invert. A collision_fn returning True unconditionally
    # must produce a plan.
    planner = RRTStar(
        start=[0.0, 0.0],
        goal=[5.0, 0.0],
        joint_limits=[(-10.0, 10.0), (-10.0, 10.0)],
        expand_dist=1.0,
        path_resolution=0.25,
        max_iter=500,
        connect_circle_dist=20.0,
        goal_sample_rate=0.3,
        collision_fn=lambda _node: True,
        use_goal_biased_sampling=True,
        rng=np.random.default_rng(1),
    )
    assert planner.plan() is not None
