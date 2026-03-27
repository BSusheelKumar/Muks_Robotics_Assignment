"""Trajectory tracking utilities for differential-drive/unicycle robots."""

import math
from bisect import bisect_right
from typing import Sequence, Tuple

from muks_robotics_assignment.common_types import ReferenceState, TrajectoryPoint2D
from muks_robotics_assignment.geometry_utils import normalize_angle


def sample_reference_at_time(
    trajectory: Sequence[TrajectoryPoint2D],
    trajectory_times: Sequence[float],
    t_query: float,
) -> ReferenceState:
    """
    Sample active trajectory by time and estimate feedforward yaw, linear speed,
    and angular speed from local finite differences.
    """
    if not trajectory:
        raise ValueError('Trajectory is empty.')

    if len(trajectory) == 1:
        x, y, _ = trajectory[0]
        return ReferenceState(x=x, y=y, yaw=0.0, v=0.0, w=0.0)

    total_time = trajectory_times[-1]
    t = max(0.0, min(t_query, total_time))
    idx = bisect_right(trajectory_times, t) - 1
    idx = max(0, min(idx, len(trajectory) - 2))

    x0, y0, t0 = trajectory[idx]
    x1, y1, t1 = trajectory[idx + 1]
    dt = max(t1 - t0, 1e-6)

    r = (t - t0) / dt
    x = x0 + r * (x1 - x0)
    y = y0 + r * (y1 - y0)
    yaw = math.atan2(y1 - y0, x1 - x0)

    segment_dist = math.hypot(x1 - x0, y1 - y0)
    v = segment_dist / dt

    if idx < len(trajectory) - 2:
        x2, y2, t2 = trajectory[idx + 2]
        yaw_next = math.atan2(y2 - y1, x2 - x1)
        w = normalize_angle(yaw_next - yaw) / max(t2 - t1, 1e-6)
    else:
        w = 0.0

    if t_query >= total_time:
        x, y, _ = trajectory[-1]
        x_prev, y_prev, _ = trajectory[-2]
        yaw = math.atan2(y - y_prev, x - x_prev)
        v = 0.0
        w = 0.0

    return ReferenceState(x=x, y=y, yaw=yaw, v=v, w=w)


def compute_tracking_control(
    current_x: float,
    current_y: float,
    current_yaw: float,
    reference: ReferenceState,
    kx: float,
    ky: float,
    kth: float,
    max_linear_speed: float,
    max_angular_speed: float,
    allow_reverse: bool,
) -> Tuple[float, float, float, float, float]:
    """
    Tracking law for a unicycle model with feedforward+feedback terms.

    Returns:
    - v_cmd, w_cmd
    - e_x, e_y, e_theta (errors in robot frame)
    """
    dx = reference.x - current_x
    dy = reference.y - current_y

    e_x = math.cos(current_yaw) * dx + math.sin(current_yaw) * dy
    e_y = -math.sin(current_yaw) * dx + math.cos(current_yaw) * dy
    e_theta = normalize_angle(reference.yaw - current_yaw)

    v_cmd = reference.v * math.cos(e_theta) + kx * e_x
    w_cmd = reference.w + ky * e_y + kth * math.sin(e_theta)

    if not allow_reverse:
        v_cmd = max(0.0, v_cmd)

    v_lim = max(float(max_linear_speed), 1e-5)
    w_lim = max(float(max_angular_speed), 1e-5)
    v_cmd = max(-v_lim, min(v_lim, v_cmd))
    w_cmd = max(-w_lim, min(w_lim, w_cmd))

    return v_cmd, w_cmd, e_x, e_y, e_theta
