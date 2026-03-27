"""Trajectory generation: spatial resampling + time parameterization."""

import math
from typing import List, Sequence, Tuple

from muks_robotics_assignment.common_types import Point2D, TrajectoryPoint2D
from muks_robotics_assignment.geometry_utils import distance


def _cumulative_arclength(points: Sequence[Point2D]) -> List[float]:
    if not points:
        return []

    s_vals = [0.0]
    for i in range(1, len(points)):
        s_vals.append(s_vals[-1] + distance(points[i - 1], points[i]))
    return s_vals


def resample_path_uniform_distance(
    path_points: Sequence[Point2D],
    sample_distance: float,
) -> Tuple[List[Point2D], List[float]]:
    """
    Resample a polyline at approximately uniform arc-length intervals.

    Returns:
    - sampled_points: (x, y)
    - sampled_s: cumulative distance at each sampled point
    """
    if not path_points:
        return [], []
    if len(path_points) == 1:
        return [path_points[0]], [0.0]

    if float(sample_distance) <= 0.0:
        raise ValueError('sample_distance must be > 0.')
    ds = float(sample_distance)
    s_vals = _cumulative_arclength(path_points)
    total_length = s_vals[-1]

    if total_length < 1e-9:
        return [path_points[0]], [0.0]

    sampled_s: List[float] = []
    current_s = 0.0
    while current_s < total_length:
        sampled_s.append(current_s)
        current_s += ds
    sampled_s.append(total_length)

    sampled_points: List[Point2D] = []
    seg_idx = 0
    for s_target in sampled_s:
        while seg_idx < len(s_vals) - 2 and s_target > s_vals[seg_idx + 1]:
            seg_idx += 1

        s0 = s_vals[seg_idx]
        s1 = s_vals[seg_idx + 1]
        p0 = path_points[seg_idx]
        p1 = path_points[seg_idx + 1]

        if abs(s1 - s0) < 1e-9:
            sampled_points.append(p0)
            continue

        r = (s_target - s0) / (s1 - s0)
        sampled_points.append((p0[0] + r * (p1[0] - p0[0]), p0[1] + r * (p1[1] - p0[1])))

    return sampled_points, sampled_s


def assign_time_constant_speed(sampled_s: Sequence[float], speed: float) -> List[float]:
    """Assign timestamps using constant forward speed."""
    if float(speed) <= 0.0:
        raise ValueError('speed must be > 0 for constant-speed timing.')
    v = float(speed)
    return [s / v for s in sampled_s]


def assign_time_trapezoidal(
    sampled_s: Sequence[float],
    max_speed: float,
    max_accel: float,
) -> List[float]:
    """Assign timestamps using a trapezoidal (or triangular) speed profile."""
    if not sampled_s:
        return []

    if float(max_speed) <= 0.0:
        raise ValueError('max_speed must be > 0 for trapezoidal timing.')
    if float(max_accel) <= 0.0:
        raise ValueError('max_accel must be > 0 for trapezoidal timing.')

    v_max = float(max_speed)
    a_max = float(max_accel)

    total_length = sampled_s[-1]
    if total_length < 1e-9:
        return [0.0 for _ in sampled_s]

    d_accel = (v_max * v_max) / (2.0 * a_max)

    if 2.0 * d_accel <= total_length:
        # Full trapezoid: accelerate, cruise, decelerate.
        t_accel = v_max / a_max
        d_cruise = total_length - 2.0 * d_accel
        t_cruise = d_cruise / v_max
        times: List[float] = []

        for s in sampled_s:
            if s <= d_accel:
                t = math.sqrt(2.0 * s / a_max)
            elif s <= d_accel + d_cruise:
                t = t_accel + (s - d_accel) / v_max
            else:
                s_decel = s - d_accel - d_cruise
                disc = max(v_max * v_max - 2.0 * a_max * s_decel, 0.0)
                t_decel = (v_max - math.sqrt(disc)) / a_max
                t = t_accel + t_cruise + t_decel
            times.append(t)
        return times

    # Triangular profile when path is too short to reach v_max.
    v_peak = math.sqrt(total_length * a_max)
    t_peak = v_peak / a_max
    half_length = 0.5 * total_length

    times = []
    for s in sampled_s:
        if s <= half_length:
            t = math.sqrt(2.0 * s / a_max)
        else:
            d_from_end = max(total_length - s, 0.0)
            t = 2.0 * t_peak - math.sqrt(2.0 * d_from_end / a_max)
        times.append(t)
    return times


def generate_time_parameterized_trajectory(
    smoothed_path: Sequence[Point2D],
    sample_distance: float,
    velocity_profile: str,
    constant_speed: float,
    max_speed: float,
    max_accel: float,
) -> List[TrajectoryPoint2D]:
    """Convert a smooth geometric path into (x, y, t) samples."""
    sampled_points, sampled_s = resample_path_uniform_distance(smoothed_path, sample_distance)

    profile = velocity_profile.strip().lower()
    if profile == 'trapezoidal':
        sampled_t = assign_time_trapezoidal(sampled_s, max_speed=max_speed, max_accel=max_accel)
    elif profile == 'constant':
        sampled_t = assign_time_constant_speed(sampled_s, speed=constant_speed)
    else:
        raise ValueError(f'Unsupported velocity_profile: {velocity_profile}')

    return [(p[0], p[1], t) for p, t in zip(sampled_points, sampled_t)]
