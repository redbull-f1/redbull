#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        
        # Waypoints to Path Converter Node
        Node(
            package='lattice_planner_rb',
            executable='waypoints_to_path_converter',
            name='waypoints_to_path_converter',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }]
        ),
        
        # Small delay to ensure waypoints are published first
        TimerAction(
            period=2.0,
            actions=[
                # Lattice Planner Node
                Node(
                    package='lattice_planner_rb',
                    executable='lattice_planner_node',
                    name='lattice_planner',
                    output='screen',
                    parameters=[{
                        'use_sim_time': LaunchConfiguration('use_sim_time')
                    }],
                    remappings=[
                        ('/global_waypoints', '/global_waypoints'),
                        ('/ego_racecar/odom', '/ego_racecar/odom'),
                        ('/planned_trajectory', '/planned_trajectory'),
                        ('/lattice_trajectories', '/lattice_trajectories'),
                        ('/drive', '/drive')
                    ]
                )
            ]
        )
    ])
