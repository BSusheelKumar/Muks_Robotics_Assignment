import pytest

from muks_robotics_assignment.waypoint_utils import parse_flat_waypoints


def test_parse_flat_waypoints_valid_input() -> None:
    pts = parse_flat_waypoints([0, 0, 1, 1, 2, 1.5])
    assert pts == [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)]


def test_parse_flat_waypoints_odd_count_raises() -> None:
    with pytest.raises(ValueError, match='even number'):
        parse_flat_waypoints([0, 0, 1])


def test_parse_flat_waypoints_too_few_raises() -> None:
    with pytest.raises(ValueError, match='At least two waypoints'):
        parse_flat_waypoints([0, 0])
