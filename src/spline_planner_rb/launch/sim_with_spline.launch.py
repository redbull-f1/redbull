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
    
    # Package directories
    f1tenth_gym_ros_share = FindPackageShare('f1tenth_gym_ros')
    lattice_planner_share = FindPackageShare('lattice_planner_rb')
    spline_planner_share = FindPackageShare('spline_planner_rb')
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # F1TENTH Gym simulator
    gym_bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='gym_bridge',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'map_path': '/home/jeong/sim_ws/src/f1tenth_gym_ros/maps'},
            {'map_img_ext': '.png'},
            {'csv_path': '/home/jeong/sim_ws/src/planning_ws/src/lattice_planner_rb/config/redbull_0.csv'}
        ]
    )
    
    # Waypoints converter
    waypoints_converter_node = Node(
        package='lattice_planner_rb',
        executable='waypoints_converter',
        name='waypoints_converter',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'csv_file_path': '/home/jeong/sim_ws/src/planning_ws/src/lattice_planner_rb/config/redbull_0.csv'}
        ]
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
        ]
    )
    
    # Simple path follower for autonomous driving
    path_follower_node = Node(
        package='lattice_planner_rb',
        executable='simple_path_follower',
        name='simple_path_follower',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('/local_waypoints_topic', '/local_waypoints')
        ]
    )
    
    # RViz visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([spline_planner_share, 'config', 'spline_planner.rviz'])],
        condition=IfCondition('true')
    )

    return LaunchDescription([
        use_sim_time_arg,
        gym_bridge_node,
        waypoints_converter_node,
        spline_planner_node,
        path_follower_node,
        rviz_node
    ])
