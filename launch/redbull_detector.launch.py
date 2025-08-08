from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='train/trained_models/TinyCenterSpeed.pt',
        description='Path to the trained model file'
    )
    
    detection_threshold_arg = DeclareLaunchArgument(
        'detection_threshold',
        default_value='0.3',
        description='Detection confidence threshold'
    )
    
    num_opponents_arg = DeclareLaunchArgument(
        'num_opponents',
        default_value='5',
        description='Maximum number of objects to detect'
    )
    
    dense_arg = DeclareLaunchArgument(
        'dense',
        default_value='true',
        description='Use dense model architecture'
    )

    # Dynamic vehicle detector node
    detector_node = Node(
        package='redbull',
        executable='dynamic_vehicle_detector',
        name='dynamic_vehicle_detector',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'detection_threshold': LaunchConfiguration('detection_threshold'),
            'num_opponents': LaunchConfiguration('num_opponents'),
            'dense': LaunchConfiguration('dense'),
            'image_size': 64
        }],
        output='screen'
    )

    return LaunchDescription([
        model_path_arg,
        detection_threshold_arg,
        num_opponents_arg,
        dense_arg,
        detector_node
    ])
