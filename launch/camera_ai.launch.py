from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mindir_arg = DeclareLaunchArgument('mindir_path', default_value='yolov8x.mindir')
    visualize_arg = DeclareLaunchArgument('visualize', default_value='false')
    interval_arg = DeclareLaunchArgument('detection_interval', default_value='1.0')
    device_arg = DeclareLaunchArgument('device_target', default_value='Ascend')
    enabled_arg = DeclareLaunchArgument('enabled', default_value='true')

    node = Node(
        package='camera_ai_ros2',
        executable='camera_ai_node',
        name='camera_ai_node',
        output='screen',
        parameters=[{
            'mindir_path': LaunchConfiguration('mindir_path'),
            'visualize': LaunchConfiguration('visualize'),
            'detection_interval': LaunchConfiguration('detection_interval'),
            'device_target': LaunchConfiguration('device_target'),
            'enabled': LaunchConfiguration('enabled'),
        }]
    )

    return LaunchDescription([
        mindir_arg,
        visualize_arg,
        interval_arg,
        device_arg,
        enabled_arg,
        node,
    ])
