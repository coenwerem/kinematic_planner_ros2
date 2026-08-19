import numpy as np
from kinematic_planner.planning.informed_rrt_star import InformedRRTStar


def _always_free(_node):
    return True


def test_plan_finds_a_path_with_correct_cost_invariant():
    planner = InformedRRTStar(
        start=[0.0, 0.0],
        goal=[5.0, 0.0],
        joint_limits=[(-10.0, 10.0), (-10.0, 10.0)],
        expand_dist=1.0,
        path_resolution=0.25,
        max_iter=800,
        connect_circle_dist=20.0,
        collision_fn=_always_free,
        search_until_max_iter=True,
        rng=np.random.default_rng(0),
    )
    path = planner.plan()
    assert path is not None
    assert np.isfinite(planner.c_best)

    for node in planner.config_tree:
        chain_cost = 0.0
        cur = node
        while cur.parent is not None:
            chain_cost += float(np.linalg.norm(cur.q - cur.parent.q))
            cur = cur.parent
        assert np.isclose(node.cost, chain_cost, atol=1e-9), \
            f"node at {node.q} has cost {node.cost}, expected {chain_cost}"


def test_informed_samples_stay_inside_the_ellipsoid_once_a_solution_exists():
    planner = InformedRRTStar(
        start=[0.0, 0.0],
        goal=[5.0, 0.0],
        joint_limits=[(-10.0, 10.0), (-10.0, 10.0)],
        expand_dist=1.0,
        path_resolution=0.25,
        max_iter=1,
        connect_circle_dist=20.0,
        collision_fn=_always_free,
        rng=np.random.default_rng(0),
    )
    planner.c_best = 6.0
    planner.compute_ellipse_params()
    samples = [planner.informed_sample() for _ in range(200)]
    for sample in samples:
        # sample must satisfy the defining property of the Informed RRT*
        # prolate-spheroid ellipse (Gammell, Srinivasa & Barfoot, IROS
        # 2014, eq. 1-2): the sum of distances from sample to the two
        # foci (start.q, end.q) does not exceed c_best.
        d_start = np.linalg.norm(sample - np.array([0.0, 0.0]))
        d_goal = np.linalg.norm(sample - np.array([5.0, 0.0]))
        assert d_start + d_goal <= planner.c_best + 1e-6


def test_collision_fn_always_true_produces_a_plan():
    planner = InformedRRTStar(
        start=[0.0, 0.0],
        goal=[5.0, 0.0],
        joint_limits=[(-10.0, 10.0), (-10.0, 10.0)],
        expand_dist=1.0,
        path_resolution=0.25,
        max_iter=800,
        connect_circle_dist=20.0,
        collision_fn=lambda _node: True,
        search_until_max_iter=False,
        rng=np.random.default_rng(1),
    )
    assert planner.plan() is not None
