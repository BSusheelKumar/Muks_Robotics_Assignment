from glob import glob
from setuptools import find_packages, setup

package_name = 'muks_robotics_assignment'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'ARCHITECTURE.md', 'REPORT.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='susheel',
    maintainer_email='susheel@todo.todo',
    description='ROS 2 assignment package: path smoothing, trajectory generation, tracking, and obstacle avoidance.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_smoothing_node = muks_robotics_assignment.path_smoothing_node:main',
            'task1_path_smoothing_node = muks_robotics_assignment.task1_path_smoothing_node:main',
            'interactive_path_smoothing_node = muks_robotics_assignment.interactive_path_smoothing_node:main',
            'task2_trajectory_generation_node = muks_robotics_assignment.task2_trajectory_generation_node:main',
            'task3_trajectory_tracking_node = muks_robotics_assignment.task3_trajectory_tracking_node:main',
            'task4_obstacle_avoidance_node = muks_robotics_assignment.task4_obstacle_avoidance_node:main',
        ],
    },
)
