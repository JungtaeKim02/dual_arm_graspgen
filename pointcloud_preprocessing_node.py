#!/usr/bin/env python3
"""
Point Cloud Preprocessing Node for RBY1 Dual Arm Grasping

기능:
- GroundedSAM2 노드에서 퍼블리시되는 /detection_node/segmented_pointcloud 토픽 수신
- 왼손/오른손 카메라에서 각각 포인트 클라우드 수신
- [2048, 3] 크기로 다운샘플링
- 전처리된 포인트 클라우드 퍼블리시

입력 토픽:
- /left_camera/segmented_pointcloud (sensor_msgs/PointCloud2)
- /right_camera/segmented_pointcloud (sensor_msgs/PointCloud2)

출력 토픽:
- /grasp_planning/left_pointcloud (sensor_msgs/PointCloud2)
- /grasp_planning/right_pointcloud (sensor_msgs/PointCloud2)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Bool
import struct


class PointCloudPreprocessingNode(Node):
    """
    Point Cloud 전처리 노드
    
    양쪽 카메라의 세그먼트된 포인트 클라우드를 수신하여
    2048개 포인트로 다운샘플링 후 퍼블리시
    """
    
    TARGET_NUM_POINTS = 2048
    
    def __init__(self):
        super().__init__('pointcloud_preprocessing_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('left_pc_topic', '/left_camera/segmented_pointcloud')
        self.declare_parameter('right_pc_topic', '/right_camera/segmented_pointcloud')
        self.declare_parameter('target_points', 2048)
        self.declare_parameter('enable_left', True)
        self.declare_parameter('enable_right', True)
        
        self.left_pc_topic = self.get_parameter('left_pc_topic').get_parameter_value().string_value
        self.right_pc_topic = self.get_parameter('right_pc_topic').get_parameter_value().string_value
        self.target_points = self.get_parameter('target_points').get_parameter_value().integer_value
        self.enable_left = self.get_parameter('enable_left').get_parameter_value().bool_value
        self.enable_right = self.get_parameter('enable_right').get_parameter_value().bool_value
        
        # ==================== State ====================
        self.left_pc_received = False
        self.right_pc_received = False
        self.left_pc_data = None
        self.right_pc_data = None
        self.left_header = None
        self.right_header = None
        
        # ==================== QoS ====================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Subscribers ====================
        if self.enable_left:
            self.left_pc_sub = self.create_subscription(
                PointCloud2,
                self.left_pc_topic,
                self.left_pc_callback,
                qos
            )
            self.get_logger().info(f"Subscribed to: {self.left_pc_topic}")
        
        if self.enable_right:
            self.right_pc_sub = self.create_subscription(
                PointCloud2,
                self.right_pc_topic,
                self.right_pc_callback,
                qos
            )
            self.get_logger().info(f"Subscribed to: {self.right_pc_topic}")
        
        # ==================== Publishers ====================
        self.left_pc_pub = self.create_publisher(
            PointCloud2,
            '/grasp_planning/left_pointcloud',
            10
        )
        
        self.right_pc_pub = self.create_publisher(
            PointCloud2,
            '/grasp_planning/right_pointcloud',
            10
        )
        
        # Ready signal publisher (양쪽 모두 수신 완료 시)
        self.ready_pub = self.create_publisher(
            Bool,
            '/grasp_planning/pointclouds_ready',
            10
        )
        
        self._print_init_info()
    
    def _print_init_info(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("Point Cloud Preprocessing Node Initialized")
        self.get_logger().info(f"  Target points: {self.target_points}")
        self.get_logger().info(f"  Left camera: {self.enable_left}")
        self.get_logger().info(f"  Right camera: {self.enable_right}")
        self.get_logger().info("Output topics:")
        self.get_logger().info(f"  /grasp_planning/left_pointcloud")
        self.get_logger().info(f"  /grasp_planning/right_pointcloud")
        self.get_logger().info(f"  /grasp_planning/pointclouds_ready")
        self.get_logger().info("=" * 60)
    
    def _pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        """
        PointCloud2 메시지를 numpy 배열로 변환
        
        Returns:
            points: (N, 3) float32 array
        """
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        
        if len(points) == 0:
            return np.array([], dtype=np.float32).reshape(0, 3)
        
        return np.array(points, dtype=np.float32)
    
    def _numpy_to_pointcloud2(
        self, 
        points: np.ndarray, 
        header: Header,
        frame_id: str = None
    ) -> PointCloud2:
        """
        Numpy 배열을 PointCloud2 메시지로 변환
        
        Args:
            points: (N, 3) float32 array
            header: 원본 헤더
            frame_id: 프레임 ID (None이면 원본 사용)
        """
        msg = PointCloud2()
        msg.header = header
        if frame_id is not None:
            msg.header.frame_id = frame_id
        
        msg.height = 1
        msg.width = points.shape[0]
        
        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * 4 bytes
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        
        # Pack points to bytes
        msg.data = points.astype(np.float32).tobytes()
        
        return msg
    
    def _downsample_points(
        self, 
        points: np.ndarray, 
        target_num: int
    ) -> np.ndarray:
        """
        포인트 클라우드 다운샘플링
        
        - target_num보다 많으면: 랜덤 서브샘플링
        - target_num보다 적으면: 복제하여 패딩
        
        Args:
            points: (N, 3) array
            target_num: 목표 포인트 수
            
        Returns:
            downsampled: (target_num, 3) array
        """
        n_points = points.shape[0]
        
        if n_points == 0:
            self.get_logger().warn("Empty point cloud received, returning zeros")
            return np.zeros((target_num, 3), dtype=np.float32)
        
        if n_points >= target_num:
            # 랜덤 서브샘플링
            indices = np.random.choice(n_points, target_num, replace=False)
            return points[indices]
        else:
            # 포인트가 부족한 경우: 복제하여 패딩
            repeat_times = (target_num // n_points) + 1
            padded = np.tile(points, (repeat_times, 1))
            indices = np.random.choice(padded.shape[0], target_num, replace=False)
            return padded[indices]
    
    def _preprocess_and_publish(
        self, 
        points: np.ndarray, 
        header: Header,
        side: str  # 'left' or 'right'
    ):
        """포인트 클라우드 전처리 및 퍼블리시"""
        # 다운샘플링
        downsampled = self._downsample_points(points, self.target_points)
        
        # PointCloud2 메시지 생성
        pc_msg = self._numpy_to_pointcloud2(downsampled, header)
        
        # 퍼블리시
        if side == 'left':
            self.left_pc_pub.publish(pc_msg)
            self.left_pc_data = downsampled
            self.left_header = header
            self.left_pc_received = True
            self.get_logger().info(
                f"Left PC: {points.shape[0]} -> {downsampled.shape[0]} points"
            )
        else:
            self.right_pc_pub.publish(pc_msg)
            self.right_pc_data = downsampled
            self.right_header = header
            self.right_pc_received = True
            self.get_logger().info(
                f"Right PC: {points.shape[0]} -> {downsampled.shape[0]} points"
            )
        
        # 양쪽 모두 수신 완료 확인
        self._check_and_publish_ready()
    
    def _check_and_publish_ready(self):
        """양쪽 포인트 클라우드 수신 완료 여부 확인 및 알림"""
        # 단일 카메라 모드
        if not self.enable_left and self.enable_right and self.right_pc_received:
            self._publish_ready(True)
        elif self.enable_left and not self.enable_right and self.left_pc_received:
            self._publish_ready(True)
        # 양쪽 모두 활성화된 경우
        elif self.enable_left and self.enable_right:
            if self.left_pc_received and self.right_pc_received:
                self._publish_ready(True)
    
    def _publish_ready(self, ready: bool):
        """Ready 신호 퍼블리시"""
        msg = Bool()
        msg.data = ready
        self.ready_pub.publish(msg)
        if ready:
            self.get_logger().info("✅ Point clouds ready for arm selection")
    
    def left_pc_callback(self, msg: PointCloud2):
        """왼쪽 카메라 포인트 클라우드 콜백"""
        points = self._pointcloud2_to_numpy(msg)
        if points.shape[0] == 0:
            self.get_logger().warn("Empty left point cloud received")
            return
        
        self._preprocess_and_publish(points, msg.header, 'left')
    
    def right_pc_callback(self, msg: PointCloud2):
        """오른쪽 카메라 포인트 클라우드 콜백"""
        points = self._pointcloud2_to_numpy(msg)
        if points.shape[0] == 0:
            self.get_logger().warn("Empty right point cloud received")
            return
        
        self._preprocess_and_publish(points, msg.header, 'right')
    
    def reset_state(self):
        """상태 초기화 (새로운 탐지 시작 시)"""
        self.left_pc_received = False
        self.right_pc_received = False
        self.left_pc_data = None
        self.right_pc_data = None


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPreprocessingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
