from sensor_msgs.msg import LaserScan

from muks_robotics_assignment.obstacle_avoidance import (
    find_blocked_index,
    nearest_obstacle_distance,
    path_collides,
    scan_to_world_points,
)


def _make_scan() -> LaserScan:
    scan = LaserScan()
    scan.angle_min = -0.2
    scan.angle_increment = 0.2
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.ranges = [1.0, 1.0, 1.0]
    return scan


def test_scan_to_world_points_projects_points() -> None:
    points = scan_to_world_points(
        scan_msg=_make_scan(),
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        detection_range=3.0,
        downsample_step=1,
        max_points=50,
    )
    assert len(points) == 3


def test_nearest_obstacle_distance() -> None:
    d = nearest_obstacle_distance((0.0, 0.0), [(1.0, 0.0), (2.0, 0.0)])
    assert abs(d - 1.0) < 1e-7


def test_path_collides_true_when_inside_clearance() -> None:
    assert path_collides([(0.0, 0.0), (1.0, 0.0)], [(1.05, 0.0)], clearance=0.1)


def test_find_blocked_index_returns_expected_index() -> None:
    trajectory_xy = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    obstacles = [(2.05, 0.0)]
    blocked_idx = find_blocked_index(
        trajectory_xy=trajectory_xy,
        obstacles=obstacles,
        start_idx=0,
        lookahead_points=3,
        clearance=0.1,
    )
    assert blocked_idx == 2
