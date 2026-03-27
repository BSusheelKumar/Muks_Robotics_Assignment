import math

from muks_robotics_assignment.geometry_utils import distance, normalize_angle, path_length


def test_distance_basic() -> None:
    assert math.isclose(distance((0.0, 0.0), (3.0, 4.0)), 5.0)


def test_normalize_angle_range() -> None:
    angle = normalize_angle(4.0 * math.pi + 0.25)
    assert -math.pi <= angle <= math.pi
    assert math.isclose(angle, 0.25, rel_tol=1e-7, abs_tol=1e-7)


def test_path_length_polyline() -> None:
    pts = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]
    assert math.isclose(path_length(pts), 10.0)
