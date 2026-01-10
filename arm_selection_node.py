#!/usr/bin/env python3
"""
Arm Selection Node (Model Preprocessing) for RBY1 Dual Arm Grasping

기능:
- 왼손/오른손 포인트 클라우드 (R.PC, L.PC) 수신
- Reachability Map (R.RM, L.RM) 로드 및 비교
- 도달 가능성 점수 계산: Score = (RM에 포함된 점 개수) / 2048
- 더 높은 점수를 받은 팔을 target planning group으로 선정
- 선정된 팔의 카메라-베이스 TF와 포인트 클라우드 퍼블리시

입력 토픽:
- /grasp_planning/left_pointcloud (sensor_msgs/PointCloud2)
- /grasp_planning/right_pointcloud (sensor_msgs/PointCloud2)
- /grasp_planning/pointclouds_ready (std_msgs/Bool)
- /tf (TF2)

출력 토픽:
- /grasp_planning/selected_arm (std_msgs/String)
- /grasp_planning/selected_pointcloud (sensor_msgs/PointCloud2)
- /grasp_planning/selected_transform (geometry_msgs/TransformStamped)
- /grasp_planning/arm_scores (std_msgs/String)  # JSON format
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import json
import os

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Bool
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from tf2_ros import Buffer, TransformListener


class ArmSelectionNode(Node):
    """
    팔 선택 노드
    
    Reachability Map을 기반으로 왼팔/오른팔 중 
    목표 객체에 더 잘 도달할 수 있는 팔을 선택
    """
    
    def __init__(self):
        super().__init__('arm_selection_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('reachability_map_dir', 
            '/home/kimjungtae/graspgen_ws/GraspGen/dual_arm_grasp/reachability_maps')
        self.declare_parameter('left_camera_frame', 'left_camera_color_optical_frame')
        self.declare_parameter('right_camera_frame', 'right_camera_color_optical_frame')
        self.declare_parameter('torso_5_frame', 'link_torso_5')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('score_threshold', 0.1)  # 최소 도달 가능 점수
        
        self.rm_dir = self.get_parameter('reachability_map_dir').get_parameter_value().string_value
        self.left_cam_frame = self.get_parameter('left_camera_frame').get_parameter_value().string_value
        self.right_cam_frame = self.get_parameter('right_camera_frame').get_parameter_value().string_value
        self.torso_5_frame = self.get_parameter('torso_5_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.score_threshold = self.get_parameter('score_threshold').get_parameter_value().double_value
        
        # ==================== Load Reachability Maps ====================
        self.left_rm = None
        self.right_rm = None
        self._load_reachability_maps()
        
        # ==================== State ====================
        self.left_pc_data = None
        self.right_pc_data = None
        self.left_header = None
        self.right_header = None
        self.has_processed = False
        
        # ==================== TF2 ====================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ==================== QoS ====================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Subscribers ====================
        self.left_pc_sub = self.create_subscription(
            PointCloud2,
            '/grasp_planning/left_pointcloud',
            self.left_pc_callback,
            qos
        )
        
        self.right_pc_sub = self.create_subscription(
            PointCloud2,
            '/grasp_planning/right_pointcloud',
            self.right_pc_callback,
            qos
        )
        
        self.ready_sub = self.create_subscription(
            Bool,
            '/grasp_planning/pointclouds_ready',
            self.ready_callback,
            10
        )
        
        # ==================== Publishers ====================
        self.selected_arm_pub = self.create_publisher(
            String,
            '/grasp_planning/selected_arm',
            10
        )
        
        self.selected_pc_pub = self.create_publisher(
            PointCloud2,
            '/grasp_planning/selected_pointcloud',
            10
        )
        
        self.selected_tf_pub = self.create_publisher(
            TransformStamped,
            '/grasp_planning/selected_transform',
            10
        )
        
        self.scores_pub = self.create_publisher(
            String,
            '/grasp_planning/arm_scores',
            10
        )
        
        self._print_init_info()
    
    def _print_init_info(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("Arm Selection Node Initialized")
        self.get_logger().info(f"  Reachability map dir: {self.rm_dir}")
        self.get_logger().info(f"  Left camera frame: {self.left_cam_frame}")
        self.get_logger().info(f"  Right camera frame: {self.right_cam_frame}")
        self.get_logger().info(f"  Torso 5 frame: {self.torso_5_frame}")
        self.get_logger().info(f"  Score threshold: {self.score_threshold}")
        self.get_logger().info(f"  Left RM loaded: {self.left_rm is not None}")
        self.get_logger().info(f"  Right RM loaded: {self.right_rm is not None}")
        self.get_logger().info("Output topics:")
        self.get_logger().info(f"  /grasp_planning/selected_arm")
        self.get_logger().info(f"  /grasp_planning/selected_pointcloud")
        self.get_logger().info(f"  /grasp_planning/selected_transform")
        self.get_logger().info("=" * 60)
    
    def _load_reachability_maps(self):
        """Reachability Map 로드"""
        left_path = os.path.join(self.rm_dir, 'left_arm_reachability.npz')
        right_path = os.path.join(self.rm_dir, 'right_arm_reachability.npz')
        
        if os.path.exists(left_path):
            try:
                data = np.load(left_path, allow_pickle=True)
                self.left_rm = {
                    'occupancy': data['occupancy'],
                    'resolution': float(data['resolution']),
                    'workspace_bounds': tuple(map(tuple, data['workspace_bounds'])),
                    'grid_shape': tuple(data['grid_shape']),
                }
                self.get_logger().info(f"✅ Loaded left arm reachability map")
            except Exception as e:
                self.get_logger().error(f"Failed to load left RM: {e}")
        else:
            self.get_logger().warn(f"Left RM not found: {left_path}")
        
        if os.path.exists(right_path):
            try:
                data = np.load(right_path, allow_pickle=True)
                self.right_rm = {
                    'occupancy': data['occupancy'],
                    'resolution': float(data['resolution']),
                    'workspace_bounds': tuple(map(tuple, data['workspace_bounds'])),
                    'grid_shape': tuple(data['grid_shape']),
                }
                self.get_logger().info(f"✅ Loaded right arm reachability map")
            except Exception as e:
                self.get_logger().error(f"Failed to load right RM: {e}")
        else:
            self.get_logger().warn(f"Right RM not found: {right_path}")
    
    def _pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        """PointCloud2 메시지를 numpy 배열로 변환"""
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        
        if len(points) == 0:
            return np.array([], dtype=np.float32).reshape(0, 3)
        
        return np.array(points, dtype=np.float32)
    
    def _transform_points_to_torso5(
        self, 
        points: np.ndarray, 
        source_frame: str
    ) -> np.ndarray:
        """
        포인트 클라우드를 torso_5 프레임으로 변환
        
        Args:
            points: (N, 3) array in source_frame
            source_frame: 원본 프레임 ID
            
        Returns:
            transformed_points: (N, 3) array in torso_5 frame
        """
        try:
            # TF 조회
            transform = self.tf_buffer.lookup_transform(
                self.torso_5_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Transform 적용
            t = transform.transform.translation
            q = transform.transform.rotation
            
            # Quaternion to rotation matrix
            import trimesh.transformations as tra
            rot_matrix = tra.quaternion_matrix([q.w, q.x, q.y, q.z])[:3, :3]
            translation = np.array([t.x, t.y, t.z])
            
            # Apply transformation
            transformed = (rot_matrix @ points.T).T + translation
            
            return transformed
            
        except Exception as e:
            self.get_logger().error(f"TF lookup failed ({source_frame} -> {self.torso_5_frame}): {e}")
            return points  # 변환 실패 시 원본 반환
    
    def _calculate_reachability_score(
        self, 
        points: np.ndarray, 
        rm: dict
    ) -> float:
        """
        Reachability Map을 기반으로 도달 가능성 점수 계산 (NumPy 벡터화)
        
        Score = (RM에 포함된 점 개수) / total_points
        
        Args:
            points: (N, 3) array in torso_5 frame
            rm: Reachability map dict
            
        Returns:
            score: 0.0 ~ 1.0
        """
        if rm is None or points.shape[0] == 0:
            return 0.0
        
        occupancy = rm['occupancy']
        resolution = rm['resolution']
        bounds = np.array(rm['workspace_bounds'])  # (3, 2) array
        grid_shape = np.array(occupancy.shape)
        
        # bounds 추출: (3,) arrays
        mins = bounds[:, 0]  # [x_min, y_min, z_min]
        maxs = bounds[:, 1]  # [x_max, y_max, z_max]
        
        # 1. 경계 내부 점 필터링 (벡터화)
        in_bounds_mask = np.all(
            (points >= mins) & (points <= maxs), 
            axis=1
        )  # (N,) bool array
        
        # 경계 내 점이 없으면 0점
        if not np.any(in_bounds_mask):
            return 0.0
        
        # 2. 경계 내부 점만 추출
        valid_points = points[in_bounds_mask]  # (M, 3)
        
        # 3. 그리드 인덱스 계산 (벡터화)
        indices = ((valid_points - mins) / resolution).astype(np.int32)
        
        # 4. 인덱스 클리핑 (경계 처리)
        indices = np.clip(indices, 0, grid_shape - 1)
        
        # 5. Occupancy 조회 (벡터화 - advanced indexing)
        ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]
        occupancy_values = occupancy[ix, iy, iz]
        
        # 6. 도달 가능 점 카운트
        reachable_count = np.sum(occupancy_values > 0)
        
        score = reachable_count / points.shape[0]
        return float(score)
    
    def _get_camera_to_base_transform(self, camera_frame: str) -> TransformStamped:
        """카메라 프레임에서 베이스 프레임으로의 변환 조회"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return transform
        except Exception as e:
            self.get_logger().error(f"TF lookup failed ({camera_frame} -> {self.base_frame}): {e}")
            return None
    
    def left_pc_callback(self, msg: PointCloud2):
        """왼쪽 포인트 클라우드 콜백"""
        self.left_pc_data = self._pointcloud2_to_numpy(msg)
        self.left_header = msg.header
        self.get_logger().debug(f"Received left PC: {self.left_pc_data.shape}")
    
    def right_pc_callback(self, msg: PointCloud2):
        """오른쪽 포인트 클라우드 콜백"""
        self.right_pc_data = self._pointcloud2_to_numpy(msg)
        self.right_header = msg.header
        self.get_logger().debug(f"Received right PC: {self.right_pc_data.shape}")
    
    def ready_callback(self, msg: Bool):
        """포인트 클라우드 준비 완료 콜백"""
        if not msg.data:
            return
        
        if self.has_processed:
            self.get_logger().info("Already processed, skipping...")
            return
        
        self.get_logger().info("Point clouds ready, starting arm selection...")
        self._select_arm()
    
    def _select_arm(self):
        """팔 선택 로직 실행"""
        scores = {'left': 0.0, 'right': 0.0}
        
        # 왼팔 점수 계산
        if self.left_pc_data is not None and self.left_rm is not None:
            # torso_5 프레임으로 변환
            left_points_torso5 = self._transform_points_to_torso5(
                self.left_pc_data, 
                self.left_header.frame_id if self.left_header else self.left_cam_frame
            )
            scores['left'] = self._calculate_reachability_score(left_points_torso5, self.left_rm)
            self.get_logger().info(f"Left arm score: {scores['left']:.4f}")
        
        # 오른팔 점수 계산
        if self.right_pc_data is not None and self.right_rm is not None:
            right_points_torso5 = self._transform_points_to_torso5(
                self.right_pc_data,
                self.right_header.frame_id if self.right_header else self.right_cam_frame
            )
            scores['right'] = self._calculate_reachability_score(right_points_torso5, self.right_rm)
            self.get_logger().info(f"Right arm score: {scores['right']:.4f}")
        
        # 점수 퍼블리시
        scores_msg = String()
        scores_msg.data = json.dumps(scores)
        self.scores_pub.publish(scores_msg)
        
        # 팔 선택
        if scores['left'] < self.score_threshold and scores['right'] < self.score_threshold:
            self.get_logger().warn(
                f"Both arms below threshold ({self.score_threshold}). "
                f"Left: {scores['left']:.4f}, Right: {scores['right']:.4f}"
            )
            # 그래도 더 높은 쪽 선택
        
        if scores['left'] >= scores['right']:
            selected_arm = 'left'
            selected_pc = self.left_pc_data
            selected_header = self.left_header
            selected_cam_frame = self.left_cam_frame
        else:
            selected_arm = 'right'
            selected_pc = self.right_pc_data
            selected_header = self.right_header
            selected_cam_frame = self.right_cam_frame
        
        self.get_logger().info(f"✅ Selected arm: {selected_arm.upper()} (score: {scores[selected_arm]:.4f})")
        
        # 선택된 팔 퍼블리시
        arm_msg = String()
        arm_msg.data = selected_arm
        self.selected_arm_pub.publish(arm_msg)
        
        # 선택된 포인트 클라우드 퍼블리시
        if selected_pc is not None and selected_header is not None:
            pc_msg = self._numpy_to_pointcloud2(selected_pc, selected_header)
            self.selected_pc_pub.publish(pc_msg)
        
        # 선택된 카메라의 TF 퍼블리시
        transform = self._get_camera_to_base_transform(selected_cam_frame)
        if transform is not None:
            self.selected_tf_pub.publish(transform)
        
        self.has_processed = True
    
    def _numpy_to_pointcloud2(self, points: np.ndarray, header) -> PointCloud2:
        """Numpy 배열을 PointCloud2 메시지로 변환"""
        from sensor_msgs.msg import PointField
        
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        
        return msg
    
    def reset_state(self):
        """상태 초기화"""
        self.left_pc_data = None
        self.right_pc_data = None
        self.has_processed = False


def main(args=None):
    rclpy.init(args=args)
    node = ArmSelectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
