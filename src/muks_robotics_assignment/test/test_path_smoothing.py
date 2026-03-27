from muks_robotics_assignment.path_smoothing import smooth_path_catmull_rom


def test_smooth_path_with_two_points_produces_dense_path() -> None:
    waypoints = [(0.0, 0.0), (1.0, 0.0)]
    smoothed = smooth_path_catmull_rom(waypoints, samples_per_segment=10)
    assert len(smoothed) >= 10
    assert smoothed[0] == (0.0, 0.0)
    assert smoothed[-1] == (1.0, 0.0)


def test_smooth_path_short_input_returns_input() -> None:
    waypoints = [(0.0, 0.0)]
    assert smooth_path_catmull_rom(waypoints) == waypoints


def test_smooth_path_samples_per_segment_lower_bound() -> None:
    waypoints = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0)]
    smoothed = smooth_path_catmull_rom(waypoints, samples_per_segment=1)
    assert len(smoothed) >= 2
    assert smoothed[-1] == waypoints[-1]
