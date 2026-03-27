"""Shared data types used across planning and control modules."""

from dataclasses import dataclass
from typing import Tuple


Point2D = Tuple[float, float]
TrajectoryPoint2D = Tuple[float, float, float]


@dataclass
class ReferenceState:
    """Time-sampled reference state for trajectory tracking."""

    x: float
    y: float
    yaw: float
    v: float
    w: float
