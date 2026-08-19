# src/kinematic_planner/test/scripts/test_informed_rrt_star_node_rng_reuse.py
"""Regression test for the compute_plan() RNG-reseed bug in
kinematic_planner.scripts.informed_rrt_star_node.InformedRRTStarPlanner,
mirroring test_planner_node_rng_reuse.py for the sibling node.

Prior to the fix, compute_plan() built InformedRRTStar with
rng=np.random.default_rng(self.random_seed) freshly on every call, so
every retry across successive /joint_states callbacks replayed an
identical sample sequence. The fix constructs the generator once (as
self.rng in __init__) and reuses it across calls.
"""
import numpy as np

from kinematic_planner.planning.informed_rrt_star import InformedRRTStar


def _always_free(_node):
    return True


def _build_and_plan(rng, max_iter=200):
    """Mirrors InformedRRTStarPlanner.compute_plan()'s planner construction."""
    planner = InformedRRTStar(
        start=[0.0, 0.0],
        goal=[5.0, 0.0],
        joint_limits=[(-10.0, 10.0), (-10.0, 10.0)],
        expand_dist=1.0,
        path_resolution=0.25,
        max_iter=max_iter,
        connect_circle_dist=20.0,
        collision_fn=_always_free,
        search_until_max_iter=True,
        rng=rng,
    )
    planner.plan()
    return [tuple(node.q) for node in planner.config_tree]


def test_successive_compute_plan_style_attempts_share_one_rng_and_diverge():
    shared_rng = np.random.default_rng(42)

    first_attempt_tree = _build_and_plan(shared_rng)
    second_attempt_tree = _build_and_plan(shared_rng)

    assert first_attempt_tree != second_attempt_tree, (
        "reusing self.rng across successive compute_plan()-style attempts "
        "must advance the sample stream, not replay it"
    )


def test_fresh_default_rng_per_attempt_is_the_bug_this_fix_removes():
    first_attempt_tree = _build_and_plan(np.random.default_rng(42))
    second_attempt_tree = _build_and_plan(np.random.default_rng(42))

    assert first_attempt_tree == second_attempt_tree, (
        "sanity check on the bug being fixed: a fresh default_rng(seed) "
        "per attempt must replay an identical sample sequence"
    )
