#!/usr/bin/env python3
"""
Task 2 node: trajectory generation.

Single responsibility:
- Build smooth geometric path from waypoints
- Time-parameterize that path into (x, y, t) samples
- Publish geometry + timed trajectory outputs for RViz2/debugging
"""

from typing import List

import rclpy
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

from muks_robotics_assignment.common_types import Point2D, TrajectoryPoint2D
from muks_robotics_assignment.path_smoothing import smooth_path_catmull_rom
from muks_robotics_assignment.ros_message_utils import to_marker_point, to_path_msg, to_timed_trajectory_msg
from muks_robotics_assignment.trajectory_generation import generate_time_parameterized_trajectory
from muks_robotics_assignment.waypoint_utils import parse_flat_waypoints


class Task2TrajectoryGenerationNode(Node):
    """ROS 2 node for Task 2 (trajectory generation)."""

    def __init__(self) -> None:
        super().__init__('task2_trajectory_generation_node')

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

        if self.trajectory_sample_distance <= 0.0:
            self.get_logger().warn('trajectory_sample_distance must be > 0. Using 0.10 m.')
            self.trajectory_sample_distance = 0.10
        if self.velocity_profile not in ('constant', 'trapezoidal'):
            self.get_logger().warn(
                f'Unknown velocity_profile="{self.velocity_profile}". Falling back to "constant".'
            )
            self.velocity_profile = 'constant'
        if self.constant_speed <= 0.0:
            self.get_logger().warn('constant_speed must be > 0. Using 0.60 m/s.')
            self.constant_speed = 0.60
        if self.max_speed <= 0.0:
            self.get_logger().warn('max_speed must be > 0. Using 1.00 m/s.')
            self.max_speed = 1.00
        if self.max_accel <= 0.0:
            self.get_logger().warn('max_accel must be > 0. Using 0.80 m/s^2.')
            self.max_accel = 0.80

        flat_waypoints = self.get_parameter('waypoints').value
        initial_waypoints: List[Point2D] = parse_flat_waypoints(flat_waypoints)
        self.raw_waypoints: List[Point2D] = []
        self.smoothed_path: List[Point2D] = []
        self.trajectory: List[TrajectoryPoint2D] = []
        self.trajectory_xy: List[Point2D] = []
        self._last_external_points: List[Point2D] = []
        self._external_frame_warned = False
        self._recompute_from_points(
            points=initial_waypoints,
            input_is_smoothed=False,
            source='waypoints parameter',
        )

        self.raw_path_pub = self.create_publisher(Path, '/raw_path', 10)
        self.smoothed_path_pub = self.create_publisher(Path, '/smoothed_path', 10)
        self.trajectory_path_pub = self.create_publisher(Path, '/trajectory_path', 10)
        self.timed_traj_pub = self.create_publisher(MultiDOFJointTrajectory, '/timed_trajectory', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/trajectory_markers', 10)

        if self.use_external_path:
            self.create_subscription(Path, self.external_path_topic, self._external_path_callback, 20)

        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_visualizations)

        duration = self.trajectory[-1][2] if self.trajectory else 0.0
        self.get_logger().info(
            f'Task 2 ready: {len(self.smoothed_path)} path points -> '
            f'{len(self.trajectory)} trajectory samples, duration {duration:.2f}s.'
        )
        if self.use_external_path:
            self.get_logger().info(
                f'External path mode enabled. Listening on "{self.external_path_topic}" '
                f'(already_smoothed={self.external_path_is_smoothed}).'
            )

    def _paths_are_same(self, a: List[Point2D], b: List[Point2D], eps: float = 1e-4) -> bool:
        if len(a) != len(b):
            return False
        for (ax, ay), (bx, by) in zip(a, b):
            if abs(ax - bx) > eps or abs(ay - by) > eps:
                return False
        return True

    def _recompute_from_points(self, points: List[Point2D], input_is_smoothed: bool, source: str) -> bool:
        if len(points) < 2:
            self.get_logger().warn(f'Ignored {source}: need at least 2 points.')
            return False

        raw = list(points)
        smoothed = list(points) if input_is_smoothed else smooth_path_catmull_rom(
            raw,
            samples_per_segment=self.samples_per_segment,
        )
        trajectory = generate_time_parameterized_trajectory(
            smoothed,
            sample_distance=self.trajectory_sample_distance,
            velocity_profile=self.velocity_profile,
            constant_speed=self.constant_speed,
            max_speed=self.max_speed,
            max_accel=self.max_accel,
        )
        if len(trajectory) < 2:
            self.get_logger().warn(f'Ignored {source}: generated trajectory has fewer than 2 samples.')
            return False

        self.raw_waypoints = raw
        self.smoothed_path = smoothed
        self.trajectory = trajectory
        self.trajectory_xy = [(x, y) for x, y, _ in trajectory]

        duration = self.trajectory[-1][2] if self.trajectory else 0.0
        self.get_logger().info(
            f'Updated trajectory from {source}: {len(self.smoothed_path)} path points -> '
            f'{len(self.trajectory)} samples, duration {duration:.2f}s.'
        )
        return True

    def _external_path_callback(self, msg: Path) -> None:
        points = [
            (float(p.pose.position.x), float(p.pose.position.y))
            for p in msg.poses
        ]
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

        if self._recompute_from_points(points, self.external_path_is_smoothed, self.external_path_topic):
            self._last_external_points = list(points)

    def _build_markers(self, stamp) -> MarkerArray:
        markers = MarkerArray()

        samples = Marker()
        samples.header.frame_id = self.frame_id
        samples.header.stamp = stamp
        samples.ns = 'task2_trajectory_samples'
        samples.id = 0
        samples.type = Marker.SPHERE_LIST
        samples.action = Marker.ADD
        samples.scale.x = self.marker_scale * 0.7
        samples.scale.y = self.marker_scale * 0.7
        samples.scale.z = self.marker_scale * 0.7
        samples.color.r = 0.3
        samples.color.g = 0.6
        samples.color.b = 1.0
        samples.color.a = 1.0
        samples.points = [to_marker_point(x, y) for x, y in self.trajectory_xy]
        markers.markers.append(samples)

        return markers

    def publish_visualizations(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.raw_path_pub.publish(to_path_msg(self.raw_waypoints, self.frame_id, stamp))
        self.smoothed_path_pub.publish(to_path_msg(self.smoothed_path, self.frame_id, stamp))
        self.trajectory_path_pub.publish(to_path_msg(self.trajectory_xy, self.frame_id, stamp))
        self.timed_traj_pub.publish(to_timed_trajectory_msg(self.trajectory, self.frame_id, stamp))
        self.marker_pub.publish(self._build_markers(stamp))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Task2TrajectoryGenerationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
