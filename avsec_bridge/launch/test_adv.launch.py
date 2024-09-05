from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config_launch_arg = DeclareLaunchArgument(
        "rviz_config", default_value="test_adv_rviz_config.rviz"
    )
    rviz_param = PathJoinSubstitution(
        [
            get_package_share_directory("security_bridge"),
            "config",
            LaunchConfiguration("rviz_config"),
        ]
    )

    adv_config = PathJoinSubstitution(
        [
            get_package_share_directory("security_bridge"),
            "config",
            "adversary.yaml",
        ]
    )

    adv_node = Node(
        package="security_bridge",
        executable="adversary",
        namespace="adversary0",
        name="adversary0",
        parameters=[adv_config],
        remappings=[("input", "detections_3d"), ("output", "/agent0/detections_3d")],
        arguments=["--ros-args", "--log-level", "INFO"],
    )

    agent_pub = Node(
        package="security_bridge",
        executable="agent_detection_sample",
        namespace="agent0",
        name="agent_detection_sample",
        remappings=[("detections_3d", "/adversary0/detections_3d")],
    )

    rviz_node = Node(
        package="rviz2",
        namespace="",
        executable="rviz2",
        name="rviz2",
        parameters=[{"use_sim_time": False}],
        arguments=["-d", rviz_param],
    )

    return LaunchDescription([rviz_config_launch_arg, adv_node, agent_pub, rviz_node])
