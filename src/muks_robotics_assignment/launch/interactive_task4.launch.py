#!/usr/bin/env python3
"""Launch interactive waypoint input + Task 4 obstacle avoidance tracking."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='odom',
        description='Common frame for interactive path and tracking outputs.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if available.',
    )

    frame_id = LaunchConfiguration('frame_id')
    use_sim_time = LaunchConfiguration('use_sim_time')

    interactive = Node(
        package='muks_robotics_assignment',
        executable='interactive_path_smoothing_node',
        name='interactive_path_smoothing_node',
        output='screen',
        parameters=[{'frame_id': frame_id, 'use_sim_time': use_sim_time}],
    )

    task4 = Node(
        package='muks_robotics_assignment',
        executable='task4_obstacle_avoidance_node',
        name='task4_obstacle_avoidance_node',
        output='screen',
        parameters=[
            {
                'frame_id': frame_id,
                'use_sim_time': use_sim_time,
                'use_external_path': True,
                'external_path_topic': '/interactive_smoothed_path',
                'external_path_is_smoothed': True,
                'wait_for_external_path': True,
            }
        ],
    )

    return LaunchDescription([
        frame_id_arg,
        use_sim_time_arg,
        interactive,
        task4,
    ])

