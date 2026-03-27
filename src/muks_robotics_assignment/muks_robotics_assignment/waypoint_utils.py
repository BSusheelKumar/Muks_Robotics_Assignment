"""Waypoint parsing and validation helpers."""

from typing import Iterable, List

from muks_robotics_assignment.common_types import Point2D


def parse_flat_waypoints(flat_values: Iterable[float]) -> List[Point2D]:
    """
    Convert flattened waypoint values into [(x, y), ...] pairs.

    Why this helper exists:
    Keeping validation outside ROS node classes makes waypoint parsing reusable
    and easy to unit-test independently.
    """
    vals = [float(v) for v in flat_values]
    if len(vals) % 2 != 0:
        raise ValueError('Waypoints must contain an even number of numeric values.')

    waypoints = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    if len(waypoints) < 2:
        raise ValueError('At least two waypoints are required.')
    return waypoints
