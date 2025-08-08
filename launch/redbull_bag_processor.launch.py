from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    bag_file_arg = DeclareLaunchArgument(
        'bag_file',
        description='Path to the input bag file'
    )
    
    output_file_arg = DeclareLaunchArgument(
        'output_file',
        default_value='parsed_output.csv',
        description='Output CSV file name'
    )
    
    create_bag_arg = DeclareLaunchArgument(
        'create_bag',
        default_value='false',
        description='Create visualization bag file'
    )

    # Parse bag node
    parse_bag_node = Node(
        package='redbull',
        executable='parse_bag_csv',
        name='parse_bag_processor',
        arguments=[
            LaunchConfiguration('bag_file'),
            '--output', LaunchConfiguration('output_file')
        ],
        output='screen'
    )

    return LaunchDescription([
        bag_file_arg,
        output_file_arg,
        create_bag_arg,
        parse_bag_node
    ])
