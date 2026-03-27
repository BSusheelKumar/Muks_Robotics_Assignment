import math

import pytest

from muks_robotics_assignment.trajectory_generation import (
    assign_time_constant_speed,
    assign_time_trapezoidal,
    generate_time_parameterized_trajectory,
    resample_path_uniform_distance,
)


def test_resample_uniform_distance_line() -> None:
    pts, s = resample_path_uniform_distance([(0.0, 0.0), (1.0, 0.0)], sample_distance=0.25)
    assert len(pts) == len(s)
    assert math.isclose(s[0], 0.0)
    assert math.isclose(s[-1], 1.0)


def test_assign_time_constant_speed_monotonic() -> None:
    t = assign_time_constant_speed([0.0, 0.5, 1.0], speed=0.5)
    assert t == [0.0, 1.0, 2.0]


def test_assign_time_trapezoidal_monotonic() -> None:
    t = assign_time_trapezoidal([0.0, 0.2, 0.4, 0.8, 1.0], max_speed=1.0, max_accel=0.8)
    assert all(t[i] <= t[i + 1] for i in range(len(t) - 1))
    assert math.isclose(t[0], 0.0)


def test_generate_time_parameterized_trajectory_returns_xyz_tuples() -> None:
    traj = generate_time_parameterized_trajectory(
        smoothed_path=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        sample_distance=0.2,
        velocity_profile='constant',
        constant_speed=0.5,
        max_speed=1.0,
        max_accel=0.8,
    )
    assert len(traj) > 2
    assert len(traj[0]) == 3
    assert traj[0][2] == 0.0


def test_generate_trajectory_invalid_profile_raises() -> None:
    with pytest.raises(ValueError, match='Unsupported velocity_profile'):
        generate_time_parameterized_trajectory(
            smoothed_path=[(0.0, 0.0), (1.0, 0.0)],
            sample_distance=0.2,
            velocity_profile='unknown',
            constant_speed=0.5,
            max_speed=1.0,
            max_accel=0.8,
        )


def test_resample_non_positive_distance_raises() -> None:
    with pytest.raises(ValueError, match='sample_distance'):
        resample_path_uniform_distance([(0.0, 0.0), (1.0, 0.0)], sample_distance=0.0)
