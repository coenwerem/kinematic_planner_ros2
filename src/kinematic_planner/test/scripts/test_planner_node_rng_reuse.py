# src/kinematic_planner/test/scripts/test_planner_node_rng_reuse.py
"""Regression test for the compute_plan() RNG-reseed bug in
kinematic_planner.scripts.planner_node.SamplingBasedJSPlanner.

Prior to the fix, SamplingBasedJSPlanner.compute_plan() built the RRTStar
planner with rng=np.random.default_rng(self.random_seed) freshly on every
call. Because /joint_states can trigger compute_plan() more than once, and
max_planning_attempts > 1 exists precisely so a retry samples differently,
every retry replayed an identical sample sequence and the retry loop did
nothing useful.

The fix constructs a single np.random.default_rng(self.random_seed) once
(mirrored here as a stand-in for SamplingBasedJSPlanner.__init__'s
self.rng) and reuses it across every RRTStar construction, exactly as
compute_plan() now does via rng=self.rng. This test builds two RRTStar
planners the way two successive compute_plan() invocations on the same
node instance would, and proves their sampled trees diverge -- while
also showing, as a control, that the pre-fix pattern (a fresh
default_rng(seed) per construction) would have produced identical trees.
"""
import numpy as np

from kinematic_planner.planning.rrt_star import RRTStar


def _always_free(_node):
    return True


def _build_and_plan(rng, seed=None, max_iter=200):
    """Mirrors SamplingBasedJSPlanner.compute_plan()'s RRTStar construction."""
    planner = RRTStar(
        start=[0.0, 0.0],
        goal=[5.0, 0.0],
        joint_limits=[(-10.0, 10.0), (-10.0, 10.0)],
        expand_dist=1.0,
        path_resolution=0.25,
        max_iter=max_iter,
        connect_circle_dist=20.0,
        goal_sample_rate=0.3,
        collision_fn=_always_free,
        use_goal_biased_sampling=True,
        search_until_max_iter=True,
        rng=rng,
    )
    planner.plan()
    return [tuple(node.q) for node in planner.config_tree]


def test_successive_compute_plan_style_attempts_share_one_rng_and_diverge():
    # Stand-in for self.rng constructed once in __init__ and reused by
    # every compute_plan() call, per the fix.
    shared_rng = np.random.default_rng(42)

    first_attempt_tree = _build_and_plan(shared_rng)
    second_attempt_tree = _build_and_plan(shared_rng)

    assert first_attempt_tree != second_attempt_tree, (
        "reusing self.rng across successive compute_plan()-style attempts "
        "must advance the sample stream, not replay it"
    )


def test_fresh_default_rng_per_attempt_is_the_bug_this_fix_removes():
    # Control: this is exactly the pre-fix behavior
    # (rng=np.random.default_rng(self.random_seed) built fresh inside
    # compute_plan on every call). It must reproduce an identical tree
    # every time, which is precisely why the retry loop was broken.
    first_attempt_tree = _build_and_plan(np.random.default_rng(42))
    second_attempt_tree = _build_and_plan(np.random.default_rng(42))

    assert first_attempt_tree == second_attempt_tree, (
        "sanity check on the bug being fixed: a fresh default_rng(seed) "
        "per attempt must replay an identical sample sequence"
    )
