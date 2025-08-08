#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    f1tenth_pkg_dir = get_package_share_directory('f1tenth_gym_ros')
    lattice_pkg_dir = get_package_share_directory('lattice_planner_rb')
    
    return LaunchDescription([
        # F1TENTH Gym Simulator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(f1tenth_pkg_dir, 'launch', 'gym_bridge_launch.py')
            ]),
        ),
        
        # Waypoints to Path Converter
        Node(
            package='lattice_planner_rb',
            executable='waypoints_to_path_converter',
            name='waypoints_to_path_converter',
            output='screen'
        ),
        
        # Simple Path Follower (ego auto-drive)
        Node(
            package='lattice_planner_rb',
            executable='simple_path_follower',
            name='simple_path_follower',
            output='screen'
        ),
        
        # Lattice Planner (for visualization)
        Node(
            package='lattice_planner_rb',
            executable='lattice_planner_node',
            name='lattice_planner',
            output='screen',
            remappings=[
                ('/global_waypoints', '/global_waypoints'),
                ('/ego_racecar/odom', '/ego_racecar/odom'),
                ('/planned_trajectory', '/planned_trajectory'),
                ('/lattice_trajectories', '/lattice_trajectories'),
                ('/drive', '/lattice_drive')  # 다른 topic으로 변경 (충돌 방지)
            ]
        ),
        
        # RViz with lattice planner config
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz_lattice',
            arguments=['-d', os.path.join(lattice_pkg_dir, 'config', 'lattice_planner.rviz')],
            output='screen'
        )
    ])
