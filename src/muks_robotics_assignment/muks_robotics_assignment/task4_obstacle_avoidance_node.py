#!/usr/bin/env python3
"""
Task 4 node: obstacle-aware trajectory tracking using /scan.

Single responsibility:
- Track a time-parameterized trajectory like Task 3
- Monitor LaserScan points in front of the robot
- Locally replan around detected obstacles and continue tracking
"""

import math
from bisect import bisect_right
from typing import List

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster
from trajectory_msgs.msg import MultiDOFJointTrajectory
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState  
from muks_robotics_assignment.common_types import Point2D, ReferenceState, TrajectoryPoint2D
from muks_robotics_assignment.geometry_utils import distance, normalize_angle, yaw_from_quaternion
from muks_robotics_assignment.obstacle_avoidance import (
    AvoidanceConfig,
    build_detour_path,
    find_blocked_index,
    scan_to_world_points,
)
from muks_robotics_assignment.path_smoothing import smooth_path_catmull_rom
from muks_robotics_assignment.ros_message_utils import (
    to_marker_point,
    to_odom_msg,
    to_path_msg,
    to_pose_msg,
    to_timed_trajectory_msg,
)
from muks_robotics_assignment.trajectory_generation import generate_time_parameterized_trajectory
from muks_robotics_assignment.trajectory_tracking import compute_tracking_control, sample_reference_at_time
from muks_robotics_assignment.waypoint_utils import parse_flat_waypoints


class Task4ObstacleAvoidanceNode(Node):
    """ROS 2 node for local obstacle avoidance with trajectory tracking."""

    def __init__(self) -> None:
        super().__init__('task4_obstacle_avoidance_node')

        self._declare_parameters()
        self._load_parameters()
        self._build_trajectory()
        self._init_state()
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_timers()
        self._log_summary()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_rate_hz', 8.0)
        self.declare_parameter('samples_per_segment', 25)
        self.declare_parameter('marker_scale', 0.10)

        self.declare_parameter('trajectory_sample_distance', 0.10)
        self.declare_parameter('velocity_profile', 'constant')
        self.declare_parameter('constant_speed', 0.6)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_accel', 0.8)
        self.declare_parameter('use_external_path', False)
        self.declare_parameter('external_path_topic', '/interactive_smoothed_path')
        self.declare_parameter('external_path_is_smoothed', True)
        self.declare_parameter('wait_for_external_path', True)

        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('controller_kx', 1.5)
        self.declare_parameter('controller_ky', 4.0)
        self.declare_parameter('controller_kth', 2.5)
        self.declare_parameter('max_linear_speed', 0.2)
        self.declare_parameter('max_angular_speed', 0.3)
        self.declare_parameter('goal_tolerance', 0.05)
        self.declare_parameter('goal_yaw_tolerance', 0.10)
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('start_offset_x', 0.0)
        self.declare_parameter('start_offset_y', 0.0)
        self.declare_parameter('start_offset_yaw', 0.0)
        self.declare_parameter('use_external_odometry', False)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('base_frame_id', 'base_link')

        self.declare_parameter('use_obstacle_avoidance', True)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('scan_downsample_step', 3)
        self.declare_parameter('scan_max_points', 500)
        self.declare_parameter('obstacle_detection_range', 3.5)
        self.declare_parameter('obstacle_inflation_radius', 0.35)
        self.declare_parameter('avoidance_margin', 0.20)
        self.declare_parameter('avoidance_lookahead_points', 60)
        self.declare_parameter('avoidance_forward_points', 45)
        self.declare_parameter('avoidance_replan_cooldown_sec', 0.8)
        self.declare_parameter('avoidance_lateral_offsets', [-1.0, -0.8, -0.6, 0.6, 0.8, 1.0])
        self.declare_parameter('publish_joint_states', True)
        self.declare_parameter(
            'wheel_joint_names',
            ['wheel_left_joint', 'wheel_right_joint', 'left_wheel_joint', 'right_wheel_joint'],
        )
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('wheel_separation', 0.16)

        self.declare_parameter(
            'waypoints',
            [
                0.0, 0.0,
                1.0, 0.8,
                2.0, 1.5,
                3.0, 1.2,
                4.0, 0.2,
                5.0, -0.3,
                6.0, 0.0,
            ],
        )

    def _load_parameters(self) -> None:
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.samples_per_segment = int(self.get_parameter('samples_per_segment').value)
        self.marker_scale = float(self.get_parameter('marker_scale').value)

        self.trajectory_sample_distance = float(self.get_parameter('trajectory_sample_distance').value)
        self.velocity_profile = str(self.get_parameter('velocity_profile').value).strip().lower()
        self.constant_speed = float(self.get_parameter('constant_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_accel = float(self.get_parameter('max_accel').value)
        self.use_external_path = bool(self.get_parameter('use_external_path').value)
        self.external_path_topic = str(self.get_parameter('external_path_topic').value)
        self.external_path_is_smoothed = bool(self.get_parameter('external_path_is_smoothed').value)
        self.wait_for_external_path = bool(self.get_parameter('wait_for_external_path').value)
        self.publish_joint_states = bool(self.get_parameter('publish_joint_states').value)
        wheel_joint_names = list(self.get_parameter('wheel_joint_names').value)
        if len(wheel_joint_names) < 2:
            wheel_joint_names = ['wheel_left_joint', 'wheel_right_joint']
        self.wheel_joint_names = [str(name) for name in wheel_joint_names]
        self.wheel_radius = max(1e-4, float(self.get_parameter('wheel_radius').value))
        self.wheel_separation = max(1e-4, float(self.get_parameter('wheel_separation').value))

        if self.trajectory_sample_distance <= 0.0:
            self.get_logger().warn('trajectory_sample_distance must be > 0. Using 0.10 m.')
            self.trajectory_sample_distance = 0.10
        if self.constant_speed <= 0.0:
            self.get_logger().warn('constant_speed must be > 0. Using 0.60 m/s.')
            self.constant_speed = 0.60
        if self.max_speed <= 0.0:
            self.get_logger().warn('max_speed must be > 0. Using 1.00 m/s.')
            self.max_speed = 1.00
        if self.max_accel <= 0.0:
            self.get_logger().warn('max_accel must be > 0. Using 0.80 m/s^2.')
            self.max_accel = 0.80

        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.controller_kx = float(self.get_parameter('controller_kx').value)
        self.controller_ky = float(self.get_parameter('controller_ky').value)
        self.controller_kth = float(self.get_parameter('controller_kth').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.goal_yaw_tolerance = float(self.get_parameter('goal_yaw_tolerance').value)
        self.allow_reverse = bool(self.get_parameter('allow_reverse').value)
        self.start_offset_x = float(self.get_parameter('start_offset_x').value)
        self.start_offset_y = float(self.get_parameter('start_offset_y').value)
        self.start_offset_yaw = float(self.get_parameter('start_offset_yaw').value)
        self.use_external_odometry = bool(self.get_parameter('use_external_odometry').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.base_frame_id = str(self.get_parameter('base_frame_id').value)

        if self.velocity_profile not in ('constant', 'trapezoidal'):
            self.get_logger().warn(
                f'Unknown velocity_profile="{self.velocity_profile}". Falling back to "constant".'
            )
            self.velocity_profile = 'constant'

        if self.velocity_profile == 'constant' and self.constant_speed > self.max_linear_speed:
            suggested = max(self.max_linear_speed * 0.8, 0.05)
            self.get_logger().warn(
                f'constant_speed ({self.constant_speed:.2f}) > max_linear_speed ({self.max_linear_speed:.2f}). '
                f'Clamping constant_speed to {suggested:.2f} for stable tracking.'
            )
            self.constant_speed = suggested

        if self.velocity_profile == 'trapezoidal' and self.max_speed > self.max_linear_speed:
            suggested = max(self.max_linear_speed * 0.9, 0.05)
            self.get_logger().warn(
                f'max_speed ({self.max_speed:.2f}) > max_linear_speed ({self.max_linear_speed:.2f}). '
                f'Clamping max_speed to {suggested:.2f} for stable tracking.'
            )
            self.max_speed = suggested

        self.use_obstacle_avoidance = bool(self.get_parameter('use_obstacle_avoidance').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.scan_downsample_step = max(1, int(self.get_parameter('scan_downsample_step').value))
        self.scan_max_points = max(50, int(self.get_parameter('scan_max_points').value))
        self.obstacle_detection_range = max(0.5, float(self.get_parameter('obstacle_detection_range').value))
        self.obstacle_inflation_radius = max(0.05, float(self.get_parameter('obstacle_inflation_radius').value))
        self.avoidance_margin = max(0.01, float(self.get_parameter('avoidance_margin').value))
        self.avoidance_lookahead_points = max(10, int(self.get_parameter('avoidance_lookahead_points').value))
        self.avoidance_forward_points = max(10, int(self.get_parameter('avoidance_forward_points').value))
        self.avoidance_replan_cooldown_sec = max(
            0.0,
            float(self.get_parameter('avoidance_replan_cooldown_sec').value),
        )

        raw_offsets = self.get_parameter('avoidance_lateral_offsets').value
        try:
            lateral_offsets = [float(v) for v in raw_offsets]
        except TypeError:
            lateral_offsets = [-1.0, -0.8, -0.6, 0.6, 0.8, 1.0]
        if len(lateral_offsets) < 2:
            lateral_offsets = [-1.0, -0.8, -0.6, 0.6, 0.8, 1.0]

        self.avoidance_cfg = AvoidanceConfig(
            obstacle_detection_range=self.obstacle_detection_range,
            obstacle_inflation_radius=self.obstacle_inflation_radius,
            avoidance_margin=self.avoidance_margin,
            avoidance_lookahead_points=self.avoidance_lookahead_points,
            avoidance_forward_points=self.avoidance_forward_points,
            avoidance_lateral_offsets=lateral_offsets,
            scan_downsample_step=self.scan_downsample_step,
            scan_max_points=self.scan_max_points,
        )

    def _build_trajectory(self) -> None:
        flat_waypoints = self.get_parameter('waypoints').value
        self.raw_waypoints: List[Point2D] = parse_flat_waypoints(flat_waypoints)

        self.smoothed_path: List[Point2D] = smooth_path_catmull_rom(
            self.raw_waypoints,
            samples_per_segment=self.samples_per_segment,
        )
        base_trajectory = self._generate_trajectory(self.smoothed_path)
        if len(base_trajectory) < 2:
            raise ValueError('Trajectory generation returned fewer than 2 samples.')

        self.base_trajectory: List[TrajectoryPoint2D] = list(base_trajectory)
        self.base_trajectory_xy: List[Point2D] = [(x, y) for x, y, _ in base_trajectory]
        self._set_active_trajectory(base_trajectory)

    def _generate_trajectory(self, smoothed_path: List[Point2D]) -> List[TrajectoryPoint2D]:
        return generate_time_parameterized_trajectory(
            smoothed_path,
            sample_distance=self.trajectory_sample_distance,
            velocity_profile=self.velocity_profile,
            constant_speed=self.constant_speed,
            max_speed=self.max_speed,
            max_accel=self.max_accel,
        )

    def _set_active_trajectory(self, trajectory: List[TrajectoryPoint2D]) -> None:
        self.trajectory = list(trajectory)
        self.trajectory_xy = [(x, y) for x, y, _ in self.trajectory]
        self.trajectory_times = [t for _, _, t in self.trajectory]
        self.trajectory_total_time = self.trajectory_times[-1] if self.trajectory_times else 0.0

    def _init_state(self) -> None:
        self.robot_x = self.raw_waypoints[0][0] + self.start_offset_x
        self.robot_y = self.raw_waypoints[0][1] + self.start_offset_y
        self.robot_yaw = self.start_offset_yaw

        self.external_path_received = not self.use_external_path
        self._last_external_points: List[Point2D] = []
        self._external_frame_warned = False

        self.odom_received = False
        self.scan_received = False
        self.controller_started = False
        self.tracking_finished = False
        self.start_time = None
        self.last_control_time = None
        self.last_replan_time_sec = -1e9

        self.last_reference = ReferenceState(
            x=self.trajectory[0][0],
            y=self.trajectory[0][1],
            yaw=0.0,
            v=0.0,
            w=0.0,
        )
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_pos_error = 0.0
        self.last_e_theta = 0.0

        self.tracked_path_xy: List[Point2D] = [(self.robot_x, self.robot_y)]
        self.obstacle_points_world: List[Point2D] = []
        self.avoidance_path_xy: List[Point2D] = []
        self.avoidance_replan_count = 0

        self.sum_tracking_error_sq = 0.0
        self.max_tracking_error = 0.0
        self.tracking_samples = 0
        self.left_wheel_pos = 0.0
        self.right_wheel_pos = 0.0

    def _setup_publishers(self) -> None:
        self.raw_path_pub = self.create_publisher(Path, '/raw_path', 10)
        self.smoothed_path_pub = self.create_publisher(Path, '/smoothed_path', 10)
        self.base_trajectory_path_pub = self.create_publisher(Path, '/base_trajectory_path', 10)
        self.trajectory_path_pub = self.create_publisher(Path, '/trajectory_path', 10)
        self.avoidance_path_pub = self.create_publisher(Path, '/avoidance_path', 10)
        self.robot_path_pub = self.create_publisher(Path, '/robot_tracked_path', 10)
        self.timed_traj_pub = self.create_publisher(MultiDOFJointTrajectory, '/timed_trajectory', 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/sim_odom', 10)
        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.reference_pose_pub = self.create_publisher(PoseStamped, '/reference_pose', 10)
        self.tracking_error_pub = self.create_publisher(Float64, '/tracking_error', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/avoidance_markers', 10)
        self.joint_state_pub = (
            self.create_publisher(JointState, '/joint_states', 10)
            if self.publish_joint_states else None
        )
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

    def _setup_subscribers(self) -> None:
        if self.use_external_odometry:
            self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 20)
        if self.use_external_path:
            self.create_subscription(Path, self.external_path_topic, self._external_path_callback, 20)

        if self.use_obstacle_avoidance:
            self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 20)

    def _setup_timers(self) -> None:
        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_visualizations)
        self.create_timer(1.0 / max(self.control_rate_hz, 1e-3), self.control_step)

    def _log_summary(self) -> None:
        self.get_logger().info(
            f'Task 4 ready: {len(self.trajectory)} active trajectory samples, duration '
            f'{self.trajectory_total_time:.2f}s. Obstacle avoidance={self.use_obstacle_avoidance}, '
            f'publish_tf={self.publish_tf} ({self.frame_id}->{self.base_frame_id}).'
        )
        if self.use_external_odometry and self.publish_tf:
            self.get_logger().warn(
                'use_external_odometry=True and publish_tf=True may conflict with other TF publishers. '
                'Disable publish_tf if another source already publishes odom->base_link.'
            )
        if self.use_obstacle_avoidance:
            self.get_logger().info(
                f'Scan topic "{self.scan_topic}", inflation={self.obstacle_inflation_radius:.2f}m, '
                f'margin={self.avoidance_margin:.2f}m, cooldown={self.avoidance_replan_cooldown_sec:.2f}s.'
            )
        if self.use_external_path:
            self.get_logger().info(
                f'External path mode enabled. Listening on "{self.external_path_topic}" '
                f'(already_smoothed={self.external_path_is_smoothed}, '
                f'wait_for_path={self.wait_for_external_path}).'
            )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _odom_callback(self, msg: Odometry) -> None:
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.robot_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.odom_received = True

    def _paths_are_same(self, a: List[Point2D], b: List[Point2D], eps: float = 1e-4) -> bool:
        if len(a) != len(b):
            return False
        for (ax, ay), (bx, by) in zip(a, b):
            if abs(ax - bx) > eps or abs(ay - by) > eps:
                return False
        return True

    def _reset_tracking_runtime_for_new_trajectory(self) -> None:
        if not self.use_external_odometry:
            self.robot_x = self.raw_waypoints[0][0] + self.start_offset_x
            self.robot_y = self.raw_waypoints[0][1] + self.start_offset_y
            self.robot_yaw = self.start_offset_yaw

        self.tracked_path_xy = [(self.robot_x, self.robot_y)]
        self.avoidance_path_xy = []
        self.avoidance_replan_count = 0
        self.sum_tracking_error_sq = 0.0
        self.max_tracking_error = 0.0
        self.tracking_samples = 0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_pos_error = 0.0
        self.last_e_theta = 0.0
        self.tracking_finished = False
        self.controller_started = False
        self.start_time = None
        self.last_control_time = None
        self.last_replan_time_sec = -1e9
        self.last_reference = ReferenceState(
            x=self.trajectory[0][0],
            y=self.trajectory[0][1],
            yaw=0.0,
            v=0.0,
            w=0.0,
        )
        self.left_wheel_pos = 0.0
        self.right_wheel_pos = 0.0

    def _external_path_callback(self, msg: Path) -> None:
        points = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
        if len(points) < 2:
            return
        if self._paths_are_same(points, self._last_external_points):
            return

        if msg.header.frame_id and msg.header.frame_id != self.frame_id and not self._external_frame_warned:
            self._external_frame_warned = True
            self.get_logger().warn(
                f'External path frame "{msg.header.frame_id}" differs from frame_id "{self.frame_id}". '
                'Using coordinates directly.'
            )

        raw = list(points)
        smoothed = list(points) if self.external_path_is_smoothed else smooth_path_catmull_rom(
            raw,
            samples_per_segment=self.samples_per_segment,
        )
        base_trajectory = self._generate_trajectory(smoothed)
        if len(base_trajectory) < 2:
            self.get_logger().warn('Ignored external path: generated trajectory has fewer than 2 samples.')
            return

        self.raw_waypoints = raw
        self.smoothed_path = smoothed
        self.base_trajectory = list(base_trajectory)
        self.base_trajectory_xy = [(x, y) for x, y, _ in base_trajectory]
        self._set_active_trajectory(base_trajectory)
        self.external_path_received = True
        self._last_external_points = list(points)
        self._reset_tracking_runtime_for_new_trajectory()

        self.get_logger().info(
            f'Loaded external path with {len(points)} points -> '
            f'{len(self.trajectory)} trajectory samples.'
        )

    def _publish_robot_tf(self, stamp) -> None:
        if self.tf_broadcaster is None:
            return

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.frame_id
        tf.child_frame_id = self.base_frame_id
        tf.transform.translation.x = float(self.robot_x)
        tf.transform.translation.y = float(self.robot_y)
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = math.sin(0.5 * self.robot_yaw)
        tf.transform.rotation.w = math.cos(0.5 * self.robot_yaw)
        self.tf_broadcaster.sendTransform(tf)

    def _scan_callback(self, msg: LaserScan) -> None:
        if self.use_external_odometry and not self.odom_received:
            return

        self.obstacle_points_world = scan_to_world_points(
            scan_msg=msg,
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            robot_yaw=self.robot_yaw,
            detection_range=self.obstacle_detection_range,
            downsample_step=self.scan_downsample_step,
            max_points=self.scan_max_points,
        )
        self.scan_received = True

    # ------------------------------------------------------------------
    # Planning and Control
    # ------------------------------------------------------------------
    def _maybe_replan_for_obstacles(self, now) -> None:
        if not self.use_obstacle_avoidance:
            return
        if not self.scan_received or not self.obstacle_points_world:
            return
        if len(self.trajectory_xy) < 5 or self.start_time is None:
            return

        now_sec = now.nanoseconds * 1e-9
        if (now_sec - self.last_replan_time_sec) < self.avoidance_replan_cooldown_sec:
            return

        elapsed = (now - self.start_time).nanoseconds * 1e-9
        current_idx = max(0, bisect_right(self.trajectory_times, elapsed) - 1)
        clearance = self.obstacle_inflation_radius + self.avoidance_margin

        blocked_idx = find_blocked_index(
            trajectory_xy=self.trajectory_xy,
            obstacles=self.obstacle_points_world,
            start_idx=current_idx + 2,
            lookahead_points=self.avoidance_lookahead_points,
            clearance=clearance,
        )
        if blocked_idx is None:
            return

        self.last_replan_time_sec = now_sec
        detour_path = build_detour_path(
            robot_xy=(self.robot_x, self.robot_y),
            trajectory_xy=self.trajectory_xy,
            current_idx=current_idx,
            blocked_idx=blocked_idx,
            obstacles=self.obstacle_points_world,
            cfg=self.avoidance_cfg,
            samples_per_segment=self.samples_per_segment,
        )
        if not detour_path:
            self.get_logger().warn('Obstacle detected, but no safe detour candidate found.')
            return

        new_trajectory = self._generate_trajectory(detour_path)
        if len(new_trajectory) < 2:
            self.get_logger().warn('Detour trajectory has too few points. Keeping current trajectory.')
            return

        self._set_active_trajectory(new_trajectory)
        self.avoidance_path_xy = detour_path
        self.avoidance_replan_count += 1
        self.start_time = now
        self.last_control_time = now
        self.tracking_finished = False

        self.get_logger().info(
            f'Replan #{self.avoidance_replan_count}: switched to {len(self.trajectory_xy)} active points.'
        )

    def control_step(self) -> None:
        now = self.get_clock().now()

        if self.use_external_path and self.wait_for_external_path and not self.external_path_received:
            return
        if self.use_external_odometry and not self.odom_received:
            return

        if not self.controller_started:
            self.controller_started = True
            self.last_control_time = now
            self.start_time = now
            return

        dt = (now - self.last_control_time).nanoseconds * 1e-9
        dt = max(min(dt, 0.1), 1e-4)
        self.last_control_time = now

        self._maybe_replan_for_obstacles(now)

        elapsed = (now - self.start_time).nanoseconds * 1e-9
        reference = sample_reference_at_time(self.trajectory, self.trajectory_times, elapsed)
        self.last_reference = reference

        if self.tracking_finished:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            v_cmd, w_cmd, _, _, _ = compute_tracking_control(
                current_x=self.robot_x,
                current_y=self.robot_y,
                current_yaw=self.robot_yaw,
                reference=reference,
                kx=self.controller_kx,
                ky=self.controller_ky,
                kth=self.controller_kth,
                max_linear_speed=self.max_linear_speed,
                max_angular_speed=self.max_angular_speed,
                allow_reverse=self.allow_reverse,
            )

            if not self.use_external_odometry:
                self.robot_x += v_cmd * math.cos(self.robot_yaw) * dt
                self.robot_y += v_cmd * math.sin(self.robot_yaw) * dt
                self.robot_yaw = normalize_angle(self.robot_yaw + w_cmd * dt)

            if distance(self.tracked_path_xy[-1], (self.robot_x, self.robot_y)) > 0.01:
                self.tracked_path_xy.append((self.robot_x, self.robot_y))

            pos_error = distance((reference.x, reference.y), (self.robot_x, self.robot_y))
            yaw_error = abs(normalize_angle(reference.yaw - self.robot_yaw))

            self.last_pos_error = pos_error
            self.last_e_theta = yaw_error
            self.sum_tracking_error_sq += pos_error * pos_error
            self.tracking_samples += 1
            self.max_tracking_error = max(self.max_tracking_error, pos_error)

            if (
                elapsed >= self.trajectory_total_time
                and pos_error <= self.goal_tolerance
                and yaw_error <= self.goal_yaw_tolerance
            ):
                self.tracking_finished = True
                v_cmd = 0.0
                w_cmd = 0.0
                rms = math.sqrt(self.sum_tracking_error_sq / max(self.tracking_samples, 1))
                self.get_logger().info(
                    f'Tracking finished. RMS error={rms:.3f} m, max error={self.max_tracking_error:.3f} m.'
                )

        self.last_cmd_v = v_cmd
        self.last_cmd_w = w_cmd

        stamp = now.to_msg()
        cmd_msg = Twist()
        cmd_msg.linear.x = float(v_cmd)
        cmd_msg.angular.z = float(w_cmd)
        self.cmd_vel_pub.publish(cmd_msg)

        self.odom_pub.publish(to_odom_msg(self.robot_x, self.robot_y, self.robot_yaw, v_cmd, w_cmd, self.frame_id, stamp))
        self.robot_pose_pub.publish(to_pose_msg(self.robot_x, self.robot_y, self.robot_yaw, self.frame_id, stamp))
        self.reference_pose_pub.publish(to_pose_msg(reference.x, reference.y, reference.yaw, self.frame_id, stamp))
        self._publish_robot_tf(stamp)
        self._publish_wheel_joint_states(stamp, v_cmd, w_cmd, dt)

        err_msg = Float64()
        err_msg.data = float(self.last_pos_error)
        self.tracking_error_pub.publish(err_msg)

        
    def _publish_wheel_joint_states(self, stamp, v_cmd: float, w_cmd: float, dt: float) -> None:
        """
        Calculates and publishes how much each wheel has rotated.
        
        We use the commanded linear and angular velocities to estimate wheel speeds 
        (using differential drive math). We do this just so the 3D robot model in RViz 
        has spinning wheels, making the simulation look realistic.
        """
        if self.joint_state_pub is None:
            return

        v_left = v_cmd - 0.5 * self.wheel_separation * w_cmd
        v_right = v_cmd + 0.5 * self.wheel_separation * w_cmd
        w_left = v_left / self.wheel_radius
        w_right = v_right / self.wheel_radius

        self.left_wheel_pos += w_left * dt
        self.right_wheel_pos += w_right * dt

        js = JointState()
        js.header.stamp = stamp
        js.name = list(self.wheel_joint_names)
        js.position = []
        js.velocity = []
        for idx, name in enumerate(self.wheel_joint_names):
            lname = name.lower()
            if 'left' in lname:
                js.position.append(self.left_wheel_pos)
                js.velocity.append(w_left)
            elif 'right' in lname:
                js.position.append(self.right_wheel_pos)
                js.velocity.append(w_right)
            elif idx % 2 == 0:
                js.position.append(self.left_wheel_pos)
                js.velocity.append(w_left)
            else:
                js.position.append(self.right_wheel_pos)
                js.velocity.append(w_right)
        self.joint_state_pub.publish(js)
    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _build_markers(self, stamp) -> MarkerArray:
        markers = MarkerArray()

        obstacle_points = Marker()
        obstacle_points.header.frame_id = self.frame_id
        obstacle_points.header.stamp = stamp
        obstacle_points.ns = 'task4_obstacles'
        obstacle_points.id = 0
        obstacle_points.type = Marker.POINTS
        obstacle_points.action = Marker.ADD
        obstacle_points.scale.x = max(self.marker_scale * 0.45, 0.03)
        obstacle_points.scale.y = max(self.marker_scale * 0.45, 0.03)
        obstacle_points.color.r = 1.0
        obstacle_points.color.g = 0.1
        obstacle_points.color.b = 0.1
        obstacle_points.color.a = 0.9
        obstacle_points.points = [to_marker_point(x, y) for x, y in self.obstacle_points_world]

        avoidance_line = Marker()
        avoidance_line.header.frame_id = self.frame_id
        avoidance_line.header.stamp = stamp
        avoidance_line.ns = 'task4_avoidance_path'
        avoidance_line.id = 1
        avoidance_line.type = Marker.LINE_STRIP
        avoidance_line.action = Marker.ADD
        avoidance_line.scale.x = max(self.marker_scale * 0.35, 0.02)
        avoidance_line.color.r = 0.0
        avoidance_line.color.g = 1.0
        avoidance_line.color.b = 1.0
        avoidance_line.color.a = 0.9
        avoidance_line.points = [to_marker_point(x, y) for x, y in self.avoidance_path_xy]

        markers.markers = [obstacle_points, avoidance_line]
        return markers

    def publish_visualizations(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.raw_path_pub.publish(to_path_msg(self.raw_waypoints, self.frame_id, stamp))
        self.smoothed_path_pub.publish(to_path_msg(self.smoothed_path, self.frame_id, stamp))
        self.base_trajectory_path_pub.publish(to_path_msg(self.base_trajectory_xy, self.frame_id, stamp))
        self.trajectory_path_pub.publish(to_path_msg(self.trajectory_xy, self.frame_id, stamp))
        self.avoidance_path_pub.publish(to_path_msg(self.avoidance_path_xy, self.frame_id, stamp))
        self.robot_path_pub.publish(to_path_msg(self.tracked_path_xy, self.frame_id, stamp))
        self.timed_traj_pub.publish(to_timed_trajectory_msg(self.trajectory, self.frame_id, stamp))
        self.marker_pub.publish(self._build_markers(stamp))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Task4ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
