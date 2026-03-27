"""Reusable geometry helpers for 2D planning and control."""

import math
from typing import Sequence

from muks_robotics_assignment.common_types import Point2D


def distance(p1: Point2D, p2: Point2D) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract planar yaw from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def path_length(points: Sequence[Point2D]) -> float:
    """Total polyline length."""
    if len(points) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(points)):
        total += distance(points[i - 1], points[i])
    return total
