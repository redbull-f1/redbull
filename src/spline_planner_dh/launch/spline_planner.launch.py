from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the launch directory
    pkg_dir = get_package_share_directory('spline_planner_dh')
    
    # Declare launch arguments
    csv_file_arg = DeclareLaunchArgument(
        'csv_file',
        default_value='redbull_0.csv',
        description='CSV file containing waypoints'
    )
    
    # Global waypoints publisher node
    global_waypoints_publisher = Node(
        package='spline_planner_dh',
        executable='global_waypoints_publisher',
        name='global_waypoints_publisher',
        parameters=[{
            'csv_file': LaunchConfiguration('csv_file')
        }],
        output='screen'
    )
    
    # Spline planner node
    spline_planner_node = Node(
        package='spline_planner_dh',
        executable='spline_planner_node',
        name='spline_planner_node',
        output='screen',
        remappings=[
            ('/ego_racecar/odom', '/ego_racecar/odom'),
            ('/perception/obstacles', '/perception/obstacles'),
            ('/global_waypoints', '/global_waypoints'),
            ('/local_trajectory', '/local_trajectory')
        ]
    )
    
    return LaunchDescription([
        csv_file_arg,
        global_waypoints_publisher,
        spline_planner_node
    ])
