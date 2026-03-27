"""Reusable ROS message builders for 2D planning/tracking nodes."""

import math
from typing import Sequence

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, Transform
from nav_msgs.msg import Odometry, Path
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from muks_robotics_assignment.common_types import Point2D, TrajectoryPoint2D


def set_pose_orientation_from_yaw(pose_msg, yaw: float) -> None:
    """Populate quaternion orientation fields from planar yaw."""
    pose_msg.orientation.x = 0.0
    pose_msg.orientation.y = 0.0
    pose_msg.orientation.z = math.sin(yaw * 0.5)
    pose_msg.orientation.w = math.cos(yaw * 0.5)


def to_path_msg(points: Sequence[Point2D], frame_id: str, stamp) -> Path:
    """Build nav_msgs/Path from 2D points."""
    msg = Path()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    poses = []
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


def to_timed_trajectory_msg(
    trajectory: Sequence[TrajectoryPoint2D],
    frame_id: str,
    stamp,
) -> MultiDOFJointTrajectory:
    """Build MultiDOFJointTrajectory from (x, y, t) samples."""
    msg = MultiDOFJointTrajectory()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    for x, y, t in trajectory:
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


def to_pose_msg(x: float, y: float, yaw: float, frame_id: str, stamp) -> PoseStamped:
    """Build PoseStamped in world frame."""
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = 0.0
    set_pose_orientation_from_yaw(msg.pose, yaw)
    return msg


def to_odom_msg(
    x: float,
    y: float,
    yaw: float,
    v: float,
    w: float,
    frame_id: str,
    stamp,
) -> Odometry:
    """Build Odometry message for simulated state feedback."""
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = 'base_link'
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = 0.0
    set_pose_orientation_from_yaw(msg.pose.pose, yaw)
    msg.twist.twist.linear.x = float(v)
    msg.twist.twist.angular.z = float(w)
    return msg


def to_marker_point(x: float, y: float, z: float = 0.0) -> Point:
    """Build geometry_msgs/Point helper for marker lists."""
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p
