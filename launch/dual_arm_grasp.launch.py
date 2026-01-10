#!/usr/bin/env python3
"""
Launch file for Dual Arm Grasp Planning Pipeline

Usage:
    ros2 launch dual_arm_grasp dual_arm_grasp.launch.py
    
    # With custom parameters:
    ros2 launch dual_arm_grasp dual_arm_grasp.launch.py \
        left_pc_topic:=/left_cam/pointcloud \
        right_pc_topic:=/right_cam/pointcloud
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ==================== Launch Arguments ====================
    declare_left_pc_topic = DeclareLaunchArgument(
        'left_pc_topic',
        default_value='/left_camera/segmented_pointcloud',
        description='Left camera segmented point cloud topic'
    )
    
    declare_right_pc_topic = DeclareLaunchArgument(
        'right_pc_topic', 
        default_value='/right_camera/segmented_pointcloud',
        description='Right camera segmented point cloud topic'
    )
    
    declare_rm_dir = DeclareLaunchArgument(
        'reachability_map_dir',
        default_value='/home/kimjungtae/graspgen_ws/GraspGen/dual_arm_grasp/reachability_maps',
        description='Directory containing reachability maps'
    )
    
    declare_config_path = DeclareLaunchArgument(
        'graspgen_config',
        default_value='/models/checkpoints/graspgen_franka_panda.yml',
        description='GraspGen model config path'
    )
    
    declare_base_frame = DeclareLaunchArgument(
        'base_frame',
        default_value='base',
        description='Robot base frame'
    )
    
    declare_top_k = DeclareLaunchArgument(
        'top_k',
        default_value='20',
        description='Number of top grasp candidates to publish'
    )
    
    # ==================== Nodes ====================
    
    # 1. Point Cloud Preprocessing Node
    pointcloud_preprocessing_node = Node(
        package='dual_arm_grasp',
        executable='pointcloud_preprocessing_node',
        name='pointcloud_preprocessing_node',
        output='screen',
        parameters=[{
            'left_pc_topic': LaunchConfiguration('left_pc_topic'),
            'right_pc_topic': LaunchConfiguration('right_pc_topic'),
            'target_points': 2048,
            'enable_left': True,
            'enable_right': True,
        }],
    )
    
    # 2. Arm Selection Node
    arm_selection_node = Node(
        package='dual_arm_grasp',
        executable='arm_selection_node',
        name='arm_selection_node',
        output='screen',
        parameters=[{
            'reachability_map_dir': LaunchConfiguration('reachability_map_dir'),
            'left_camera_frame': 'left_camera_color_optical_frame',
            'right_camera_frame': 'right_camera_color_optical_frame',
            'torso_5_frame': 'link_torso_5',
            'base_frame': LaunchConfiguration('base_frame'),
            'score_threshold': 0.1,
        }],
    )
    
    # 3. GraspGen Inference Node
    graspgen_inference_node = Node(
        package='dual_arm_grasp',
        executable='graspgen_inference_node',
        name='graspgen_inference_node',
        output='screen',
        parameters=[{
            'config_path': LaunchConfiguration('graspgen_config'),
            'robot_base_frame': LaunchConfiguration('base_frame'),
            'top_k': LaunchConfiguration('top_k'),
            'yaw_adjust_mode': 'local',
            'z_offset': 0.0928,
            'filter_z_down': False,
        }],
    )
    
    return LaunchDescription([
        # Launch arguments
        declare_left_pc_topic,
        declare_right_pc_topic,
        declare_rm_dir,
        declare_config_path,
        declare_base_frame,
        declare_top_k,
        
        # Log info
        LogInfo(msg="Starting Dual Arm Grasp Planning Pipeline..."),
        
        # Nodes
        pointcloud_preprocessing_node,
        arm_selection_node,
        graspgen_inference_node,
    ])
