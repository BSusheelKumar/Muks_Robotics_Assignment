import math

import pytest

from muks_robotics_assignment.common_types import ReferenceState
from muks_robotics_assignment.trajectory_tracking import compute_tracking_control, sample_reference_at_time


def test_sample_reference_empty_trajectory_raises() -> None:
    with pytest.raises(ValueError, match='Trajectory is empty'):
        sample_reference_at_time([], [], 0.0)


def test_sample_reference_returns_zero_speed_after_end() -> None:
    traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 2.0)]
    times = [0.0, 1.0, 2.0]
    ref = sample_reference_at_time(traj, times, t_query=5.0)
    assert math.isclose(ref.x, 2.0)
    assert math.isclose(ref.v, 0.0)
    assert math.isclose(ref.w, 0.0)


def test_compute_tracking_control_respects_limits_and_no_reverse() -> None:
    ref = ReferenceState(x=10.0, y=0.0, yaw=0.0, v=3.0, w=2.0)
    v_cmd, w_cmd, *_ = compute_tracking_control(
        current_x=0.0,
        current_y=0.0,
        current_yaw=0.0,
        reference=ref,
        kx=2.0,
        ky=4.0,
        kth=3.0,
        max_linear_speed=1.0,
        max_angular_speed=1.5,
        allow_reverse=False,
    )
    assert 0.0 <= v_cmd <= 1.0
    assert -1.5 <= w_cmd <= 1.5
