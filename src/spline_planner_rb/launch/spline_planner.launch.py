#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # Spline planner node
    spline_planner_node = Node(
        package='spline_planner_rb',
        executable='spline_planner',
        name='spline_planner',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'lookahead_distance': 10.0},
            {'evasion_distance': 0.65},
            {'obs_traj_threshold': 0.3},
            {'spline_resolution': 0.1}
        ],
        remappings=[
            ('/ego_racecar/odom', '/ego_racecar/odom'),
            ('/global_waypoints', '/global_waypoints'),
            ('/perception/obstacles', '/perception/obstacles'),
            ('/local_waypoints', '/local_waypoints')
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        spline_planner_node
    ])
