"""Path smoothing algorithms."""

from typing import List, Sequence

from muks_robotics_assignment.common_types import Point2D
from muks_robotics_assignment.geometry_utils import distance


def _safe_interp(p1: Point2D, p2: Point2D, t1: float, t2: float, t: float) -> Point2D:
    denom = t2 - t1
    if abs(denom) < 1e-9:
        return p1

    a = (t2 - t) / denom
    b = (t - t1) / denom
    return (a * p1[0] + b * p2[0], a * p1[1] + b * p2[1])


def smooth_path_catmull_rom(
    waypoints: Sequence[Point2D],
    samples_per_segment: int = 25,
    alpha: float = 0.5,
) -> List[Point2D]:
    """
    Convert discrete waypoints into a smooth continuous path using
    centripetal Catmull-Rom spline interpolation.

    Why centripetal parameterization:
    It reduces self-intersections and overshoot around sharp turns compared
    to uniform parameterization, making tracking more stable.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    if samples_per_segment < 2:
        samples_per_segment = 2

    pts = [waypoints[0], *waypoints, waypoints[-1]]
    smoothed: List[Point2D] = []

    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]

        t0 = 0.0
        t1 = t0 + max(distance(p0, p1) ** alpha, 1e-6)
        t2 = t1 + max(distance(p1, p2) ** alpha, 1e-6)
        t3 = t2 + max(distance(p2, p3) ** alpha, 1e-6)

        for s in range(samples_per_segment):
            t = t1 + (t2 - t1) * (s / float(samples_per_segment))
            a1 = _safe_interp(p0, p1, t0, t1, t)
            a2 = _safe_interp(p1, p2, t1, t2, t)
            a3 = _safe_interp(p2, p3, t2, t3, t)

            b1 = _safe_interp(a1, a2, t0, t2, t)
            b2 = _safe_interp(a2, a3, t1, t3, t)

            c = _safe_interp(b1, b2, t1, t2, t)
            smoothed.append(c)

    smoothed.append(waypoints[-1])
    return smoothed
