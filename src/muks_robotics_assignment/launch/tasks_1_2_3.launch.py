#!/usr/bin/env python3
"""Launch Task 1, Task 2 and Task 3 nodes together."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='odom',
        description='Common frame_id passed to all three task nodes.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if available.',
    )

    frame_id = LaunchConfiguration('frame_id')
    use_sim_time = LaunchConfiguration('use_sim_time')

    task1 = Node(
        package='muks_robotics_assignment',
        executable='task1_path_smoothing_node',
        name='task1_path_smoothing_node',
        namespace='task1',
        output='screen',
        parameters=[{'frame_id': frame_id, 'use_sim_time': use_sim_time}],
    )

    task2 = Node(
        package='muks_robotics_assignment',
        executable='task2_trajectory_generation_node',
        name='task2_trajectory_generation_node',
        namespace='task2',
        output='screen',
        parameters=[{'frame_id': frame_id, 'use_sim_time': use_sim_time}],
    )

    task3 = Node(
        package='muks_robotics_assignment',
        executable='task3_trajectory_tracking_node',
        name='task3_trajectory_tracking_node',
        namespace='task3',
        output='screen',
        parameters=[{'frame_id': frame_id, 'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        frame_id_arg,
        use_sim_time_arg,
        task1,
        task2,
        task3,
    ])
