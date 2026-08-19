# src/kinematic_planner/test/planning/test_interpolate.py
import numpy as np
import pytest

from kinematic_planner.planning.interpolate import interpolate_waypoints


def test_single_waypoint_returns_it_unchanged():
    frames = interpolate_waypoints([np.array([1.0, 2.0])], steps_per_segment=5)
    assert len(frames) == 1
    assert np.allclose(frames[0], [1.0, 2.0])


def test_two_waypoints_interpolate_linearly_and_end_exactly_at_the_target():
    waypoints = [np.array([0.0, 0.0]), np.array([1.0, 2.0])]
    frames = interpolate_waypoints(waypoints, steps_per_segment=4)
    # 4 intermediate steps (t=0, 0.25, 0.5, 0.75) plus the final waypoint itself.
    assert len(frames) == 5
    assert np.allclose(frames[0], [0.0, 0.0])
    assert np.allclose(frames[2], [0.5, 1.0])
    assert np.allclose(frames[-1], [1.0, 2.0])


def test_three_waypoints_chain_segments_without_duplicating_the_middle_one():
    waypoints = [np.array([0.0]), np.array([1.0]), np.array([3.0])]
    frames = interpolate_waypoints(waypoints, steps_per_segment=2)
    values = [f[0] for f in frames]
    assert values == pytest.approx([0.0, 0.5, 1.0, 2.0, 3.0])


def test_zero_steps_per_segment_raises():
    with pytest.raises(ValueError):
        interpolate_waypoints([np.array([0.0]), np.array([1.0])], steps_per_segment=0)


def test_empty_waypoints_returns_empty():
    assert interpolate_waypoints([], steps_per_segment=5) == []
