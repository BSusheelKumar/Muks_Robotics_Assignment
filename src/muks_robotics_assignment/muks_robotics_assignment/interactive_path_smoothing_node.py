#!/usr/bin/env python3
"""
Interactive path smoothing node for RViz demo.

Workflow:
1) User adds waypoints in RViz using "2D Goal Pose" (/goal_pose) or "Publish Point" (/clicked_point).
2) Node accumulates clicked points as raw waypoints.
3) Node computes smooth path using Catmull-Rom spline.
4) Node publishes both raw and smoothed paths + markers for live visualization.
"""

from typing import List

from geometry_msgs.msg import PointStamped, PoseStamped
import rclpy
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

from muks_robotics_assignment.common_types import Point2D
from muks_robotics_assignment.geometry_utils import distance
from muks_robotics_assignment.path_smoothing import smooth_path_catmull_rom
from muks_robotics_assignment.ros_message_utils import to_marker_point, to_path_msg


class InteractivePathSmoothingNode(Node):
    """Build and smooth a waypoint path interactively from RViz tool inputs."""

    def __init__(self) -> None:
        super().__init__('interactive_path_smoothing_node')

        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('clicked_point_topic', '/clicked_point')
        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter('publish_rate_hz', 8.0)
        self.declare_parameter('samples_per_segment', 25)
        self.declare_parameter('marker_scale', 0.10)
        self.declare_parameter('min_point_distance', 0.05)
        self.declare_parameter('max_waypoints', 500)
        self.declare_parameter('raw_path_topic', '/interactive_raw_path')
        self.declare_parameter('smoothed_path_topic', '/interactive_smoothed_path')
        self.declare_parameter('marker_topic', '/interactive_path_markers')
        self.declare_parameter('clear_service', '/interactive_path/clear')

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.clicked_point_topic = str(self.get_parameter('clicked_point_topic').value)
        self.goal_pose_topic = str(self.get_parameter('goal_pose_topic').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.samples_per_segment = max(2, int(self.get_parameter('samples_per_segment').value))
        self.marker_scale = float(self.get_parameter('marker_scale').value)
        self.min_point_distance = max(0.0, float(self.get_parameter('min_point_distance').value))
        self.max_waypoints = max(2, int(self.get_parameter('max_waypoints').value))
        self.raw_path_topic = str(self.get_parameter('raw_path_topic').value)
        self.smoothed_path_topic = str(self.get_parameter('smoothed_path_topic').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.clear_service_name = str(self.get_parameter('clear_service').value)

        self.raw_waypoints: List[Point2D] = []
        self.smoothed_path: List[Point2D] = []
        self.frame_warned_sources = set()

        self.raw_path_pub = self.create_publisher(Path, self.raw_path_topic, 10)
        self.smoothed_path_pub = self.create_publisher(Path, self.smoothed_path_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.create_subscription(PointStamped, self.clicked_point_topic, self._clicked_point_callback, 20)
        self.create_subscription(PoseStamped, self.goal_pose_topic, self._goal_pose_callback, 20)
        self.create_service(Empty, self.clear_service_name, self._clear_callback)
        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_visualizations)

        self.get_logger().info(
            f'Interactive node ready. Input topics: "{self.goal_pose_topic}" (2D Goal Pose), '
            f'"{self.clicked_point_topic}" (Publish Point). '
            f'Clear service: "{self.clear_service_name}".'
        )

    def _add_waypoint(self, point: Point2D, source_frame: str, source_name: str) -> None:
        if source_frame and source_frame != self.frame_id and source_name not in self.frame_warned_sources:
            self.frame_warned_sources.add(source_name)
            self.get_logger().warn(
                f'Input frame "{source_frame}" from "{source_name}" differs from frame_id "{self.frame_id}". '
                f'Using incoming coordinates directly.'
            )

        if self.raw_waypoints and distance(point, self.raw_waypoints[-1]) < self.min_point_distance:
            self.get_logger().info('Ignored point: too close to previous waypoint.')
            return

        if len(self.raw_waypoints) >= self.max_waypoints:
            self.get_logger().warn(
                f'Max waypoint limit ({self.max_waypoints}) reached. Clear path to continue.'
            )
            return

        self.raw_waypoints.append(point)
        if len(self.raw_waypoints) >= 2:
            self.smoothed_path = smooth_path_catmull_rom(
                self.raw_waypoints,
                samples_per_segment=self.samples_per_segment,
            )
        else:
            self.smoothed_path = list(self.raw_waypoints)

        self.get_logger().info(
            f'Added waypoint #{len(self.raw_waypoints)} from "{source_name}" '
            f'at ({point[0]:.2f}, {point[1]:.2f}). Smoothed points: {len(self.smoothed_path)}'
        )

    def _clicked_point_callback(self, msg: PointStamped) -> None:
        point = (float(msg.point.x), float(msg.point.y))
        self._add_waypoint(point, msg.header.frame_id, self.clicked_point_topic)

    def _goal_pose_callback(self, msg: PoseStamped) -> None:
        point = (float(msg.pose.position.x), float(msg.pose.position.y))
        self._add_waypoint(point, msg.header.frame_id, self.goal_pose_topic)

    def _clear_callback(self, request: Empty.Request, response: Empty.Response) -> Empty.Response:
        del request
        self.raw_waypoints.clear()
        self.smoothed_path.clear()
        self.get_logger().info('Cleared interactive waypoints and smoothed path.')
        return response

    def _build_markers(self, stamp) -> MarkerArray:
        markers = MarkerArray()

        raw_points = Marker()
        raw_points.header.frame_id = self.frame_id
        raw_points.header.stamp = stamp
        raw_points.ns = 'interactive_raw_waypoints'
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
        raw_points.points = [to_marker_point(x, y) for x, y in self.raw_waypoints]
        markers.markers.append(raw_points)

        raw_line = Marker()
        raw_line.header.frame_id = self.frame_id
        raw_line.header.stamp = stamp
        raw_line.ns = 'interactive_raw_path'
        raw_line.id = 1
        raw_line.type = Marker.LINE_STRIP
        raw_line.action = Marker.ADD
        raw_line.scale.x = max(0.015, self.marker_scale * 0.30)
        raw_line.color.r = 1.0
        raw_line.color.g = 0.6
        raw_line.color.b = 0.0
        raw_line.color.a = 1.0
        raw_line.points = [to_marker_point(x, y) for x, y in self.raw_waypoints]
        markers.markers.append(raw_line)

        smooth_line = Marker()
        smooth_line.header.frame_id = self.frame_id
        smooth_line.header.stamp = stamp
        smooth_line.ns = 'interactive_smoothed_path'
        smooth_line.id = 2
        smooth_line.type = Marker.LINE_STRIP
        smooth_line.action = Marker.ADD
        smooth_line.scale.x = max(0.02, self.marker_scale * 0.35)
        smooth_line.color.r = 0.2
        smooth_line.color.g = 1.0
        smooth_line.color.b = 0.3
        smooth_line.color.a = 1.0
        smooth_line.points = [to_marker_point(x, y) for x, y in self.smoothed_path]
        markers.markers.append(smooth_line)

        return markers

    def publish_visualizations(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.raw_path_pub.publish(to_path_msg(self.raw_waypoints, self.frame_id, stamp))
        self.smoothed_path_pub.publish(to_path_msg(self.smoothed_path, self.frame_id, stamp))
        self.marker_pub.publish(self._build_markers(stamp))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InteractivePathSmoothingNode()
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
