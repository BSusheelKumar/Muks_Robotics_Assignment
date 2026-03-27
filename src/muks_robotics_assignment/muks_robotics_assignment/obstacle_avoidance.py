"""Local obstacle-aware replanning helpers using LaserScan point clouds."""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from sensor_msgs.msg import LaserScan

from muks_robotics_assignment.common_types import Point2D
from muks_robotics_assignment.geometry_utils import distance, path_length
from muks_robotics_assignment.path_smoothing import smooth_path_catmull_rom


@dataclass
class AvoidanceConfig:
    obstacle_detection_range: float
    obstacle_inflation_radius: float
    avoidance_margin: float
    avoidance_lookahead_points: int
    avoidance_forward_points: int
    avoidance_lateral_offsets: List[float]
    scan_downsample_step: int
    scan_max_points: int


def scan_to_world_points(
    scan_msg: LaserScan,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    detection_range: float,
    downsample_step: int,
    max_points: int,
) -> List[Point2D]:
    """Project scan ranges into world coordinates using current robot pose."""
    r_min = max(float(scan_msg.range_min), 0.01)
    r_max = min(float(scan_msg.range_max), detection_range)
    if r_max <= r_min:
        return []

    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)
    points_world: List[Point2D] = []

    step = max(1, int(downsample_step))
    limit = max(1, int(max_points))

    for i in range(0, len(scan_msg.ranges), step):
        r = float(scan_msg.ranges[i])
        if not math.isfinite(r) or r < r_min or r > r_max:
            continue

        angle = scan_msg.angle_min + i * scan_msg.angle_increment
        x_r = r * math.cos(angle)
        y_r = r * math.sin(angle)

        x_w = robot_x + cos_yaw * x_r - sin_yaw * y_r
        y_w = robot_y + sin_yaw * x_r + cos_yaw * y_r
        points_world.append((x_w, y_w))

        if len(points_world) >= limit:
            break

    return points_world


def nearest_obstacle_distance(point: Point2D, obstacles: Sequence[Point2D]) -> float:
    """Distance from point to nearest obstacle sample."""
    if not obstacles:
        return float('inf')

    px, py = point
    best_d2 = float('inf')
    for ox, oy in obstacles:
        dx = ox - px
        dy = oy - py
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
    return math.sqrt(best_d2)


def path_collides(path_xy: Sequence[Point2D], obstacles: Sequence[Point2D], clearance: float) -> bool:
    """True if any path point enters obstacle clearance radius."""
    if not path_xy or not obstacles:
        return False

    thresh_sq = clearance * clearance
    for px, py in path_xy:
        for ox, oy in obstacles:
            dx = ox - px
            dy = oy - py
            if dx * dx + dy * dy <= thresh_sq:
                return True
    return False


def find_blocked_index(
    trajectory_xy: Sequence[Point2D],
    obstacles: Sequence[Point2D],
    start_idx: int,
    lookahead_points: int,
    clearance: float,
) -> Optional[int]:
    """Find first blocked trajectory index in a local lookahead window."""
    if not trajectory_xy or not obstacles:
        return None

    s_idx = max(0, start_idx)
    e_idx = min(len(trajectory_xy) - 1, s_idx + max(1, lookahead_points))
    for idx in range(s_idx, e_idx + 1):
        if nearest_obstacle_distance(trajectory_xy[idx], obstacles) <= clearance:
            return idx
    return None


def build_detour_path(
    robot_xy: Point2D,
    trajectory_xy: Sequence[Point2D],
    current_idx: int,
    blocked_idx: int,
    obstacles: Sequence[Point2D],
    cfg: AvoidanceConfig,
    samples_per_segment: int,
) -> Optional[List[Point2D]]:
    """
    Build a local detour around nearest obstacle and splice it into the trajectory.

    Strategy:
    - Pick entry/exit anchors around blocked segment.
    - Generate multiple lateral bypass candidates.
    - Smooth each candidate and reject colliding ones.
    - Pick lowest-cost candidate (shorter and with more clearance).
    """
    if len(trajectory_xy) < 4 or not obstacles:
        return None

    entry_idx = max(current_idx + 1, blocked_idx - 5)
    exit_idx = min(len(trajectory_xy) - 1, blocked_idx + max(10, cfg.avoidance_forward_points))
    if exit_idx <= entry_idx:
        return None

    p_robot = robot_xy
    p_entry = trajectory_xy[entry_idx]
    p_exit = trajectory_xy[exit_idx]
    suffix = list(trajectory_xy[exit_idx:])
    if len(suffix) < 2:
        return None

    dx = p_exit[0] - p_entry[0]
    dy = p_exit[1] - p_entry[1]
    seg_norm = math.hypot(dx, dy)
    if seg_norm < 1e-6:
        dx = p_exit[0] - p_robot[0]
        dy = p_exit[1] - p_robot[1]
        seg_norm = math.hypot(dx, dy)
        if seg_norm < 1e-6:
            return None

    ux, uy = dx / seg_norm, dy / seg_norm
    nx, ny = -uy, ux

    bx, by = trajectory_xy[blocked_idx]
    nearest_obs: Optional[Point2D] = None
    best_obs_d2 = float('inf')
    for ox, oy in obstacles:
        odx = ox - bx
        ody = oy - by
        d2 = odx * odx + ody * ody
        if d2 < best_obs_d2:
            best_obs_d2 = d2
            nearest_obs = (ox, oy)
    if nearest_obs is None:
        return None

    ox, oy = nearest_obs
    clearance = cfg.obstacle_inflation_radius + cfg.avoidance_margin
    min_control_points = 4

    best_candidate: Optional[List[Point2D]] = None
    best_score = float('inf')

    for lateral_scale in cfg.avoidance_lateral_offsets:
        lateral = lateral_scale * clearance
        detour_mid = (ox + nx * lateral, oy + ny * lateral)
        detour_mid2 = (
            detour_mid[0] + ux * clearance * 0.8,
            detour_mid[1] + uy * clearance * 0.8,
        )

        control = [p_robot, p_entry, detour_mid, detour_mid2, p_exit]

        dedup: List[Point2D] = [control[0]]
        for pt in control[1:]:
            if distance(pt, dedup[-1]) > 0.05:
                dedup.append(pt)
        if len(dedup) < min_control_points:
            continue

        detour_segment = smooth_path_catmull_rom(
            dedup,
            samples_per_segment=max(6, samples_per_segment // 5),
        )
        candidate = detour_segment + suffix[1:]

        if path_collides(candidate, obstacles, clearance * 0.95):
            continue

        sample_step = max(1, len(candidate) // 30)
        min_clearance = min(nearest_obstacle_distance(p, obstacles) for p in candidate[::sample_step])
        score = path_length(candidate) - 0.2 * min_clearance

        if score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate
