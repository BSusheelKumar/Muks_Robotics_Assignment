#!/usr/bin/env python3
"""
ROS 2 node for assignment tasks with modular architecture.

Tasks covered:
1) Path smoothing from discrete waypoints
2) Time-parameterized trajectory generation
3) Trajectory tracking controller (differential drive)
4) Obstacle-aware local replanning using /scan (extra credit)

Architecture:
- Algorithm modules:
  - path_smoothing.py
  - trajectory_generation.py
  - trajectory_tracking.py
  - obstacle_avoidance.py
  - geometry_utils.py
- This file focuses on ROS orchestration only:
  parameter loading, subscriptions, publishers, control loop, and RViz output.
"""

from bisect import bisect_right
import math
from typing import Iterable, List, Sequence

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, Transform, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import ParameterType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from muks_robotics_assignment.common_types import Point2D, ReferenceState, TrajectoryPoint2D
from muks_robotics_assignment.geometry_utils import distance, normalize_angle, yaw_from_quaternion
from muks_robotics_assignment.obstacle_avoidance import (
    AvoidanceConfig,
    build_detour_path,
    find_blocked_index,
    scan_to_world_points,
)
from muks_robotics_assignment.path_smoothing import smooth_path_catmull_rom
from muks_robotics_assignment.trajectory_generation import generate_time_parameterized_trajectory
from muks_robotics_assignment.trajectory_tracking import compute_tracking_control, sample_reference_at_time


class PathSmoothingNode(Node):
    """Main orchestration node for planning, tracking, and obstacle avoidance."""

    def __init__(self) -> None:
        super().__init__('path_smoothing_node')

        self._declare_parameters()
        self._load_parameters()
        self._build_base_trajectory()
        self._init_runtime_state()
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_timers()
        self._log_configuration_summary()

    # ---------------------------------------------------------------------
    # Parameter + Configuration
    # ---------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_rate_hz', 8.0)
        self.declare_parameter('samples_per_segment', 25)
        self.declare_parameter('marker_scale', 0.10)

        # Task 2: trajectory generation
        self.declare_parameter('trajectory_sample_distance', 0.10)
        self.declare_parameter('velocity_profile', 'constant')
        self.declare_parameter('constant_speed', 0.6)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_accel', 0.8)

        # Task 3: tracking controller
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('controller_kx', 1.5)
        self.declare_parameter('controller_ky', 4.0)
        self.declare_parameter('controller_kth', 2.5)
        self.declare_parameter('max_linear_speed', 1.0)
        self.declare_parameter('max_angular_speed', 2.0)
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

        # Extra credit: obstacle avoidance
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

        lateral_param = self.get_parameter('avoidance_lateral_offsets').get_parameter_value()
        if lateral_param.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            lateral_offsets = [float(v) for v in lateral_param.double_array_value]
        elif lateral_param.type == ParameterType.PARAMETER_INTEGER_ARRAY:
            lateral_offsets = [float(v) for v in lateral_param.integer_array_value]
        else:
            lateral_offsets = [float(v) for v in list(self.get_parameter('avoidance_lateral_offsets').value)]
        if not lateral_offsets:
            lateral_offsets = [-1.0, -0.8, -0.6, 0.6, 0.8, 1.0]

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

    def _load_waypoints(self) -> List[Point2D]:
        """Read and validate flattened [x0,y0,x1,y1,...] waypoint parameter."""
        param = self.get_parameter('waypoints').get_parameter_value()

        values: Iterable[float]
        if param.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            values = list(param.double_array_value)
        elif param.type == ParameterType.PARAMETER_INTEGER_ARRAY:
            values = [float(v) for v in param.integer_array_value]
        else:
            values = list(self.get_parameter('waypoints').value)

        values = [float(v) for v in values]
        if len(values) < 4:
            raise ValueError('Parameter "waypoints" must have at least 2 points: [x0,y0,x1,y1,...]')

        if len(values) % 2 != 0:
            self.get_logger().warn('Odd number of waypoint values received; last value will be ignored.')
            values = values[:-1]

        return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

    def _build_base_trajectory(self) -> None:
        """Build smooth geometric path + initial timed trajectory."""
        self.raw_waypoints = self._load_waypoints()
        self.smooth_waypoints = smooth_path_catmull_rom(
            self.raw_waypoints,
            samples_per_segment=self.samples_per_segment,
        )

        self.base_trajectory = generate_time_parameterized_trajectory(
            self.smooth_waypoints,
            sample_distance=self.trajectory_sample_distance,
            velocity_profile=self.velocity_profile,
            constant_speed=self.constant_speed,
            max_speed=self.max_speed,
            max_accel=self.max_accel,
        )
        if not self.base_trajectory:
            raise ValueError('Generated trajectory is empty.')

        self.base_trajectory_xy = [(x, y) for x, y, _ in self.base_trajectory]
        self._set_active_trajectory(self.base_trajectory)

    def _init_runtime_state(self) -> None:
        """Initialize state estimates, tracking metrics, and replan bookkeeping."""
        start_ref = sample_reference_at_time(self.trajectory, self.trajectory_times, 0.0)

        self.robot_x = start_ref.x + self.start_offset_x
        self.robot_y = start_ref.y + self.start_offset_y
        self.robot_yaw = normalize_angle(start_ref.yaw + self.start_offset_yaw)

        self.last_reference = start_ref
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_pos_error = distance((start_ref.x, start_ref.y), (self.robot_x, self.robot_y))
        self.last_e_theta = normalize_angle(start_ref.yaw - self.robot_yaw)

        self.tracked_path_xy: List[Point2D] = [(self.robot_x, self.robot_y)]
        self.max_tracking_error = 0.0
        self.sum_tracking_error_sq = 0.0
        self.tracking_samples = 0
        self.tracking_finished = False

        self.odom_received = False
        self.controller_started = False
        self.odom_frame_warned = False

        self.scan_received = False
        self.obstacle_points_world: List[Point2D] = []
        self.avoidance_path_xy: List[Point2D] = []
        self.avoidance_replan_count = 0
        self.last_replan_time_sec = -1e9

        self.start_time = self.get_clock().now()
        self.last_control_time = None

    def _set_active_trajectory(self, trajectory: Sequence[TrajectoryPoint2D]) -> None:
        """Set currently tracked trajectory and derived indexing arrays."""
        self.trajectory = list(trajectory)
        self.trajectory_xy = [(x, y) for x, y, _ in self.trajectory]
        self.trajectory_times = [t for _, _, t in self.trajectory]
        self.trajectory_total_time = self.trajectory_times[-1] if self.trajectory_times else 0.0

    def _log_configuration_summary(self) -> None:
        self.get_logger().info(
            f'Initialized with {len(self.raw_waypoints)} raw waypoints -> '
            f'{len(self.smooth_waypoints)} smoothed points -> '
            f'{len(self.trajectory)} trajectory samples. '
            f'Controller running at {1.0 / self.control_period:.1f} Hz.'
        )
        self.get_logger().info(
            f'Tracking params: profile={self.velocity_profile}, '
            f'constant_speed={self.constant_speed:.2f}, max_speed={self.max_speed:.2f}, '
            f'max_linear_speed={self.max_linear_speed:.2f}, max_angular_speed={self.max_angular_speed:.2f}, '
            f'kx={self.controller_kx:.2f}, ky={self.controller_ky:.2f}, kth={self.controller_kth:.2f}.'
        )
        self.get_logger().info(
            f'publish_tf={self.publish_tf} ({self.frame_id}->{self.base_frame_id}).'
        )
        if self.use_external_odometry and self.publish_tf:
            self.get_logger().warn(
                'use_external_odometry=True and publish_tf=True may conflict with other TF publishers. '
                'Disable publish_tf if another source already publishes odom->base_link.'
            )
        if self.use_obstacle_avoidance:
            self.get_logger().info(
                f'Avoidance params: inflation={self.obstacle_inflation_radius:.2f} m, '
                f'margin={self.avoidance_margin:.2f} m, lookahead_points={self.avoidance_lookahead_points}, '
                f'forward_points={self.avoidance_forward_points}, cooldown={self.avoidance_replan_cooldown_sec:.2f}s.'
            )

    # ---------------------------------------------------------------------
    # ROS Wiring
    # ---------------------------------------------------------------------
    def _setup_publishers(self) -> None:
        self.raw_path_pub = self.create_publisher(Path, 'raw_path', 10)
        self.smooth_path_pub = self.create_publisher(Path, 'smoothed_path', 10)
        self.trajectory_path_pub = self.create_publisher(Path, 'trajectory_path', 10)
        self.base_trajectory_path_pub = self.create_publisher(Path, 'base_trajectory_path', 10)
        self.avoidance_path_pub = self.create_publisher(Path, 'avoidance_path', 10)
        self.timed_trajectory_pub = self.create_publisher(MultiDOFJointTrajectory, 'timed_trajectory', 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, 'sim_odom', 10)
        self.robot_pose_pub = self.create_publisher(PoseStamped, 'robot_pose', 10)
        self.reference_pose_pub = self.create_publisher(PoseStamped, 'reference_pose', 10)
        self.robot_path_pub = self.create_publisher(Path, 'robot_tracked_path', 10)
        self.tracking_error_pub = self.create_publisher(Float64, 'tracking_error', 10)

        self.marker_pub = self.create_publisher(MarkerArray, 'path_markers', 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

    def _setup_subscribers(self) -> None:
        if self.use_external_odometry:
            self.odom_sub = self.create_subscription(
                Odometry,
                self.odom_topic,
                self._odom_callback,
                10,
            )
            self.get_logger().info(
                f'Using external odometry from "{self.odom_topic}" for controller feedback.'
            )

        if self.use_obstacle_avoidance:
            self.scan_sub = self.create_subscription(
                LaserScan,
                self.scan_topic,
                self._scan_callback,
                10,
            )
            self.get_logger().info(
                f'Obstacle avoidance enabled using scan topic "{self.scan_topic}".'
            )

    def _setup_timers(self) -> None:
        self.viz_period = 1.0 / self.publish_rate_hz if self.publish_rate_hz > 0.0 else 0.2
        self.control_period = 1.0 / self.control_rate_hz if self.control_rate_hz > 0.0 else 0.02
        self.viz_timer = self.create_timer(self.viz_period, self.publish_visualizations)
        self.control_timer = self.create_timer(self.control_period, self.control_step)

    # ---------------------------------------------------------------------
    # Sensor Callbacks
    # ---------------------------------------------------------------------
    def _odom_callback(self, msg: Odometry) -> None:
        """Update robot state from external odometry when enabled."""
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        self.robot_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        if not self.odom_received:
            self.odom_received = True
            self.tracked_path_xy = [(self.robot_x, self.robot_y)]
            self.get_logger().info(
                f'External odom received. Starting controller from '
                f'({self.robot_x:.2f}, {self.robot_y:.2f}, yaw={self.robot_yaw:.2f}).'
            )

        if msg.header.frame_id and msg.header.frame_id != self.frame_id and not self.odom_frame_warned:
            self.odom_frame_warned = True
            self.get_logger().warn(
                f'Odometry frame is "{msg.header.frame_id}" but frame_id is "{self.frame_id}". '
                f'Use matching frames in RViz to avoid visual offsets.'
            )

    def _scan_callback(self, msg: LaserScan) -> None:
        """Convert /scan data to world-frame obstacle samples."""
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

    # ---------------------------------------------------------------------
    # Planning: Obstacle Replanning
    # ---------------------------------------------------------------------
    def _maybe_replan_for_obstacles(self, now) -> None:
        """Replan active trajectory if scan indicates blockage ahead."""
        if not self.use_obstacle_avoidance:
            return
        if not self.scan_received or not self.obstacle_points_world:
            return
        if len(self.trajectory_xy) < 5:
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
            self.get_logger().warn('Obstacle detected but no safe detour candidate found.')
            return

        new_trajectory = generate_time_parameterized_trajectory(
            detour_path,
            sample_distance=self.trajectory_sample_distance,
            velocity_profile=self.velocity_profile,
            constant_speed=self.constant_speed,
            max_speed=self.max_speed,
            max_accel=self.max_accel,
        )
        if len(new_trajectory) < 2:
            self.get_logger().warn('Detour generation produced too few points. Keeping current trajectory.')
            return

        self._set_active_trajectory(new_trajectory)
        self.avoidance_path_xy = detour_path
        self.avoidance_replan_count += 1

        self.start_time = now
        self.last_control_time = now
        self.tracking_finished = False

        self.get_logger().info(
            f'Obstacle replanning #{self.avoidance_replan_count}: '
            f'{len(self.trajectory_xy)} active trajectory points.'
        )

    # ---------------------------------------------------------------------
    # Message Builders
    # ---------------------------------------------------------------------
    def _set_pose_orientation_from_yaw(self, pose_msg, yaw: float) -> None:
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = math.sin(yaw * 0.5)
        pose_msg.orientation.w = math.cos(yaw * 0.5)

    def _to_path_msg(self, points: Sequence[Point2D], stamp) -> Path:
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        poses: List[PoseStamped] = []
        for x, y in points:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        msg.poses = poses
        return msg

    def _to_timed_trajectory_msg(self, stamp) -> MultiDOFJointTrajectory:
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        for x, y, t in self.trajectory:
            point = MultiDOFJointTrajectoryPoint()
            tf = Transform()
            tf.translation.x = float(x)
            tf.translation.y = float(y)
            tf.translation.z = 0.0
            tf.rotation.w = 1.0
            point.transforms.append(tf)

            sec = int(t)
            nanosec = int(round((t - sec) * 1e9))
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            msg.points.append(point)

        return msg

    def _to_pose_msg(self, x: float, y: float, yaw: float, stamp) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        self._set_pose_orientation_from_yaw(msg.pose, yaw)
        return msg

    def _to_odom_msg(self, x: float, y: float, yaw: float, v: float, w: float, stamp) -> Odometry:
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        self._set_pose_orientation_from_yaw(msg.pose.pose, yaw)
        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.angular.z = float(w)
        return msg

    def _to_marker_point(self, x: float, y: float, z: float = 0.0) -> Point:
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

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

    # ---------------------------------------------------------------------
    # Control Loop
    # ---------------------------------------------------------------------
    def control_step(self) -> None:
        """Track active trajectory and publish cmd_vel + state debug topics."""
        now = self.get_clock().now()

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
            v_cmd = 0.0
            w_cmd = 0.0
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

            if elapsed >= self.trajectory_total_time and pos_error <= self.goal_tolerance and yaw_error <= self.goal_yaw_tolerance:
                self.tracking_finished = True
                v_cmd = 0.0
                w_cmd = 0.0
                rms = math.sqrt(self.sum_tracking_error_sq / max(self.tracking_samples, 1))
                self.get_logger().info(
                    f'Tracking finished. RMS error={rms:.3f} m, max error={self.max_tracking_error:.3f} m, '
                    f'samples={self.tracking_samples}.'
                )

        self.last_cmd_v = v_cmd
        self.last_cmd_w = w_cmd

        stamp = now.to_msg()
        cmd_msg = Twist()
        cmd_msg.linear.x = float(v_cmd)
        cmd_msg.angular.z = float(w_cmd)
        self.cmd_vel_pub.publish(cmd_msg)

        self.odom_pub.publish(self._to_odom_msg(self.robot_x, self.robot_y, self.robot_yaw, v_cmd, w_cmd, stamp))
        self.robot_pose_pub.publish(self._to_pose_msg(self.robot_x, self.robot_y, self.robot_yaw, stamp))
        self.reference_pose_pub.publish(self._to_pose_msg(reference.x, reference.y, reference.yaw, stamp))
        self._publish_robot_tf(stamp)

        err_msg = Float64()
        err_msg.data = float(self.last_pos_error)
        self.tracking_error_pub.publish(err_msg)

    # ---------------------------------------------------------------------
    # RViz Markers
    # ---------------------------------------------------------------------
    def _build_markers(self, stamp) -> MarkerArray:
        markers = MarkerArray()

        raw_points = Marker()
        raw_points.header.frame_id = self.frame_id
        raw_points.header.stamp = stamp
        raw_points.ns = 'raw_waypoints'
        raw_points.id = 0
        raw_points.type = Marker.SPHERE_LIST
        raw_points.action = Marker.ADD
        raw_points.scale.x = self.marker_scale
        raw_points.scale.y = self.marker_scale
        raw_points.scale.z = self.marker_scale
        raw_points.color.r = 1.0
        raw_points.color.g = 0.2
        raw_points.color.b = 0.2
        raw_points.color.a = 1.0
        raw_points.points = [self._to_marker_point(x, y) for x, y in self.raw_waypoints]

        raw_line = Marker()
        raw_line.header.frame_id = self.frame_id
        raw_line.header.stamp = stamp
        raw_line.ns = 'raw_path'
        raw_line.id = 1
        raw_line.type = Marker.LINE_STRIP
        raw_line.action = Marker.ADD
        raw_line.scale.x = max(self.marker_scale * 0.35, 0.02)
        raw_line.color.r = 1.0
        raw_line.color.g = 0.6
        raw_line.color.b = 0.0
        raw_line.color.a = 1.0
        raw_line.points = [self._to_marker_point(x, y) for x, y in self.raw_waypoints]

        smooth_line = Marker()
        smooth_line.header.frame_id = self.frame_id
        smooth_line.header.stamp = stamp
        smooth_line.ns = 'smoothed_path'
        smooth_line.id = 2
        smooth_line.type = Marker.LINE_STRIP
        smooth_line.action = Marker.ADD
        smooth_line.scale.x = max(self.marker_scale * 0.45, 0.03)
        smooth_line.color.r = 0.1
        smooth_line.color.g = 0.9
        smooth_line.color.b = 0.2
        smooth_line.color.a = 1.0
        smooth_line.points = [self._to_marker_point(x, y) for x, y in self.smooth_waypoints]

        trajectory_points = Marker()
        trajectory_points.header.frame_id = self.frame_id
        trajectory_points.header.stamp = stamp
        trajectory_points.ns = 'trajectory_samples'
        trajectory_points.id = 3
        trajectory_points.type = Marker.SPHERE_LIST
        trajectory_points.action = Marker.ADD
        trajectory_points.scale.x = max(self.marker_scale * 0.75, 0.04)
        trajectory_points.scale.y = max(self.marker_scale * 0.75, 0.04)
        trajectory_points.scale.z = max(self.marker_scale * 0.75, 0.04)
        trajectory_points.color.r = 0.1
        trajectory_points.color.g = 0.5
        trajectory_points.color.b = 1.0
        trajectory_points.color.a = 1.0
        trajectory_points.points = [self._to_marker_point(x, y) for x, y in self.trajectory_xy]

        tracked_line = Marker()
        tracked_line.header.frame_id = self.frame_id
        tracked_line.header.stamp = stamp
        tracked_line.ns = 'robot_tracked_path'
        tracked_line.id = 4
        tracked_line.type = Marker.LINE_STRIP
        tracked_line.action = Marker.ADD
        tracked_line.scale.x = max(self.marker_scale * 0.32, 0.02)
        tracked_line.color.r = 0.9
        tracked_line.color.g = 0.1
        tracked_line.color.b = 1.0
        tracked_line.color.a = 1.0
        tracked_line.points = [self._to_marker_point(x, y) for x, y in self.tracked_path_xy]

        robot_marker = Marker()
        robot_marker.header.frame_id = self.frame_id
        robot_marker.header.stamp = stamp
        robot_marker.ns = 'robot_pose'
        robot_marker.id = 5
        robot_marker.type = Marker.SPHERE
        robot_marker.action = Marker.ADD
        robot_marker.scale.x = max(self.marker_scale * 1.2, 0.08)
        robot_marker.scale.y = max(self.marker_scale * 1.2, 0.08)
        robot_marker.scale.z = max(self.marker_scale * 1.2, 0.08)
        robot_marker.color.r = 0.2
        robot_marker.color.g = 0.2
        robot_marker.color.b = 1.0
        robot_marker.color.a = 1.0
        robot_marker.pose.position.x = float(self.robot_x)
        robot_marker.pose.position.y = float(self.robot_y)
        robot_marker.pose.position.z = 0.0
        self._set_pose_orientation_from_yaw(robot_marker.pose, self.robot_yaw)

        reference_marker = Marker()
        reference_marker.header.frame_id = self.frame_id
        reference_marker.header.stamp = stamp
        reference_marker.ns = 'reference_pose'
        reference_marker.id = 6
        reference_marker.type = Marker.SPHERE
        reference_marker.action = Marker.ADD
        reference_marker.scale.x = max(self.marker_scale, 0.06)
        reference_marker.scale.y = max(self.marker_scale, 0.06)
        reference_marker.scale.z = max(self.marker_scale, 0.06)
        reference_marker.color.r = 1.0
        reference_marker.color.g = 1.0
        reference_marker.color.b = 0.0
        reference_marker.color.a = 1.0
        reference_marker.pose.position.x = float(self.last_reference.x)
        reference_marker.pose.position.y = float(self.last_reference.y)
        reference_marker.pose.position.z = 0.0
        self._set_pose_orientation_from_yaw(reference_marker.pose, self.last_reference.yaw)

        error_line = Marker()
        error_line.header.frame_id = self.frame_id
        error_line.header.stamp = stamp
        error_line.ns = 'tracking_error_line'
        error_line.id = 7
        error_line.type = Marker.LINE_STRIP
        error_line.action = Marker.ADD
        error_line.scale.x = max(self.marker_scale * 0.20, 0.015)
        error_line.color.r = 1.0
        error_line.color.g = 1.0
        error_line.color.b = 1.0
        error_line.color.a = 0.9
        error_line.points = [
            self._to_marker_point(self.robot_x, self.robot_y),
            self._to_marker_point(self.last_reference.x, self.last_reference.y),
        ]

        obstacle_points = Marker()
        obstacle_points.header.frame_id = self.frame_id
        obstacle_points.header.stamp = stamp
        obstacle_points.ns = 'obstacles'
        obstacle_points.id = 8
        obstacle_points.type = Marker.POINTS
        obstacle_points.action = Marker.ADD
        obstacle_points.scale.x = max(self.marker_scale * 0.45, 0.03)
        obstacle_points.scale.y = max(self.marker_scale * 0.45, 0.03)
        obstacle_points.color.r = 1.0
        obstacle_points.color.g = 0.1
        obstacle_points.color.b = 0.1
        obstacle_points.color.a = 0.9
        obstacle_points.points = [self._to_marker_point(x, y) for x, y in self.obstacle_points_world]

        avoidance_line = Marker()
        avoidance_line.header.frame_id = self.frame_id
        avoidance_line.header.stamp = stamp
        avoidance_line.ns = 'avoidance_path'
        avoidance_line.id = 9
        avoidance_line.type = Marker.LINE_STRIP
        avoidance_line.action = Marker.ADD
        avoidance_line.scale.x = max(self.marker_scale * 0.35, 0.02)
        avoidance_line.color.r = 0.0
        avoidance_line.color.g = 1.0
        avoidance_line.color.b = 1.0
        avoidance_line.color.a = 0.9
        avoidance_line.points = [self._to_marker_point(x, y) for x, y in self.avoidance_path_xy]

        markers.markers = [
            raw_points,
            raw_line,
            smooth_line,
            trajectory_points,
            tracked_line,
            robot_marker,
            reference_marker,
            error_line,
            obstacle_points,
            avoidance_line,
        ]
        return markers

    # ---------------------------------------------------------------------
    # Visualization Loop
    # ---------------------------------------------------------------------
    def publish_visualizations(self) -> None:
        """Publish path-level outputs and marker diagnostics to RViz."""
        stamp = self.get_clock().now().to_msg()

        self.raw_path_pub.publish(self._to_path_msg(self.raw_waypoints, stamp))
        self.smooth_path_pub.publish(self._to_path_msg(self.smooth_waypoints, stamp))
        self.trajectory_path_pub.publish(self._to_path_msg(self.trajectory_xy, stamp))
        self.base_trajectory_path_pub.publish(self._to_path_msg(self.base_trajectory_xy, stamp))
        self.avoidance_path_pub.publish(self._to_path_msg(self.avoidance_path_xy, stamp))
        self.timed_trajectory_pub.publish(self._to_timed_trajectory_msg(stamp))
        self.robot_path_pub.publish(self._to_path_msg(self.tracked_path_xy, stamp))
        self.marker_pub.publish(self._build_markers(stamp))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathSmoothingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
