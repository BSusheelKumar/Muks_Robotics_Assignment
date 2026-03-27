#!/usr/bin/env python3
"""
Task 1 node: path smoothing only.

Single responsibility:
- Read discrete waypoints
- Generate smooth continuous path
- Publish both raw and smoothed paths for RViz2
"""

from typing import List

import rclpy
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from muks_robotics_assignment.common_types import Point2D
from muks_robotics_assignment.path_smoothing import smooth_path_catmull_rom
from muks_robotics_assignment.ros_message_utils import to_marker_point, to_path_msg
from muks_robotics_assignment.waypoint_utils import parse_flat_waypoints


class Task1PathSmoothingNode(Node):
    """ROS 2 node for Task 1 (path smoothing visualization)."""

    def __init__(self) -> None:
        super().__init__('task1_path_smoothing_node')

        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_rate_hz', 8.0)
        self.declare_parameter('samples_per_segment', 25)
        self.declare_parameter('marker_scale', 0.10)
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

        flat_waypoints = self.get_parameter('waypoints').value
        self.raw_waypoints: List[Point2D] = parse_flat_waypoints(flat_waypoints)
        self.smoothed_path: List[Point2D] = smooth_path_catmull_rom(
            self.raw_waypoints,
            samples_per_segment=self.samples_per_segment,
        )

        self.raw_path_pub = self.create_publisher(Path, '/raw_path', 10)
        self.smoothed_path_pub = self.create_publisher(Path, '/smoothed_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)

        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_visualizations)

        self.get_logger().info(
            f'Task 1 ready: {len(self.raw_waypoints)} raw waypoints -> '
            f'{len(self.smoothed_path)} smoothed points.'
        )

    def _build_markers(self, stamp) -> MarkerArray:
        markers = MarkerArray()

        raw_points = Marker()
        raw_points.header.frame_id = self.frame_id
        raw_points.header.stamp = stamp
        raw_points.ns = 'task1_raw_waypoints'
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

        smooth_line = Marker()
        smooth_line.header.frame_id = self.frame_id
        smooth_line.header.stamp = stamp
        smooth_line.ns = 'task1_smoothed_path'
        smooth_line.id = 1
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
    node = Task1PathSmoothingNode()
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
