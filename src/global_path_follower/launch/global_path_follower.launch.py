#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch configuration variables
    waypoint_file = LaunchConfiguration('waypoint_file')
    
    # Declare launch arguments
    declare_waypoint_file_cmd = DeclareLaunchArgument(
        'waypoint_file',
        default_value='redbull_0.csv',
        description='Waypoint CSV file name')
    
    # Global path follower node
    global_path_follower_node = Node(
        package='global_path_follower',
        executable='global_path_follower_node',
        name='global_path_follower',
        output='screen',
        parameters=[{
            'waypoint_file': waypoint_file,
            'map_frame': 'map',
            'control_frequency': 20.0,
            'lookahead_distance': 3.0,
            'max_speed': 2.0,
            'min_speed': 0.5,
            'wheelbase': 0.33
        }]
    )
    
    return LaunchDescription([
        declare_waypoint_file_cmd,
        global_path_follower_node
    ])
