"""Linear joint-space interpolation between RRT* waypoints, for playback/
rendering only. This is kinematic interpolation, not a dynamics-validated
trajectory: no velocity or acceleration limits are applied, and no timing
model beyond a fixed step count per segment."""
from typing import List

import numpy as np


def interpolate_waypoints(waypoints: List[np.ndarray], steps_per_segment: int) -> List[np.ndarray]:
    """Linearly interpolate steps_per_segment intermediate configurations
    between each consecutive pair of waypoints, then append the final
    waypoint once. Returns a single flat list of configurations tracing
    the whole path start to goal."""
    if steps_per_segment < 1:
        raise ValueError(f"steps_per_segment must be >= 1, got {steps_per_segment}")
    if len(waypoints) == 0:
        return []
    frames = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        for step in range(steps_per_segment):
            t = step / steps_per_segment
            frames.append(a + t * (b - a))
    frames.append(np.asarray(waypoints[-1], dtype=float))
    return frames
