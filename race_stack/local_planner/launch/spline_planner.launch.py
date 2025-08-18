from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch arguments
    import os
    # launch 파일 위치 기준으로 trajectory 경로 생성
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    default_csv = os.path.abspath(os.path.join(launch_dir, '..', '..', 'maps', 'trajectory', 'redbull_1.csv'))

    csv_file     = LaunchConfiguration('csv_file')
    frame_id     = LaunchConfiguration('frame_id')
    publish_rate = LaunchConfiguration('publish_rate')

    return LaunchDescription([
        DeclareLaunchArgument(
            'csv_file',
            default_value=default_csv,
            description='CSV file containing waypoints'
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='map',
            description='Frame id for waypoint data & markers'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='2.0',
            description='Publish rate (Hz) for global waypoints/markers'
        ),

        # Global waypoints publisher (CSV→/global_waypoints + visualization)
        Node(
            package='spline_planner_dh_test',
            executable='global_waypoints_publisher',
            name='global_waypoints_publisher',
            output='screen',
            parameters=[{
                'csv_file': csv_file,
                'frame_id': frame_id,
                'publish_rate': publish_rate,
            }],
        ),

        # Spline planner node
        Node(
            package='spline_planner_dh_test',
            executable='spline_planner_node',
            name='spline_planner_node',
            output='screen',
            remappings=[
                ('/car_state/odom', '/car_state/odom'),
                ('/obstacles', '/obstacles'),
                ('/global_waypoints', '/global_waypoints'),
                ('/local_waypoints', '/local_waypoints'),
            ],
        ),
    ])
