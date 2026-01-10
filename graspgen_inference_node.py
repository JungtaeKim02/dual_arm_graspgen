#!/usr/bin/env python3
"""
GraspGen Inference Node for RBY1 Dual Arm Grasping

기능:
- Arm Selection Node에서 선정된 포인트 클라우드와 TF 수신
- GraspGen 모델 로드 및 grasp pose 생성
- 베이스 좌표계로 변환하여 결과 퍼블리시

입력 토픽:
- /grasp_planning/selected_pointcloud (sensor_msgs/PointCloud2)
- /grasp_planning/selected_transform (geometry_msgs/TransformStamped)
- /grasp_planning/selected_arm (std_msgs/String)

출력 토픽:
- /grasp_candidates (geometry_msgs/PoseArray)
- /filtered_grasp_poses_viz (geometry_msgs/PoseArray)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import torch
import trimesh.transformations as tra
import sensor_msgs_py.point_cloud2 as pc2
import os
import yaml

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_pose

# GraspGen
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from easydict import EasyDict as edict


class GraspGenInferenceNode(Node):
    """
    GraspGen 추론 노드
    
    선정된 팔의 포인트 클라우드를 받아 grasp pose를 생성
    """
    
    def __init__(self):
        super().__init__('graspgen_inference_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('config_path', '/models/checkpoints/graspgen_franka_panda.yml')
        self.declare_parameter('robot_base_frame', 'base')
        self.declare_parameter('top_k', 20)
        self.declare_parameter('yaw_adjust_mode', 'local')  # 'local' or 'world'
        self.declare_parameter('z_offset', 0.0928)  # 그리퍼 오프셋
        self.declare_parameter('filter_z_down', False)  # Z축 아래 방향 필터링
        
        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.robot_base_frame = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        self.top_k = self.get_parameter('top_k').get_parameter_value().integer_value
        self.yaw_adjust_mode = self.get_parameter('yaw_adjust_mode').get_parameter_value().string_value
        self.z_offset = self.get_parameter('z_offset').get_parameter_value().double_value
        self.filter_z_down = self.get_parameter('filter_z_down').get_parameter_value().bool_value
        
        # ==================== State ====================
        self.transform = None
        self.selected_arm = None
        self.has_published = False
        
        # Yaw flip quaternion (180도 회전)
        self.dq_flip_z_wxyz = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        
        # ==================== Device & Model ====================
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.get_logger().info(f"Using device: {self.device}")
        
        self.sampler = None
        self._load_model()
        
        # ==================== QoS ====================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Subscribers ====================
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/grasp_planning/selected_pointcloud',
            self.pointcloud_callback,
            qos
        )
        
        self.tf_sub = self.create_subscription(
            TransformStamped,
            '/grasp_planning/selected_transform',
            self.transform_callback,
            10
        )
        
        self.arm_sub = self.create_subscription(
            String,
            '/grasp_planning/selected_arm',
            self.arm_callback,
            10
        )
        
        # ==================== Publishers ====================
        self.candidates_pub = self.create_publisher(
            PoseArray,
            '/grasp_candidates',
            10
        )
        
        self.viz_pub = self.create_publisher(
            PoseArray,
            '/filtered_grasp_poses_viz',
            10
        )
        
        self._print_init_info()
    
    def _print_init_info(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("GraspGen Inference Node Initialized")
        self.get_logger().info(f"  Config: {self.config_path}")
        self.get_logger().info(f"  Device: {self.device}")
        self.get_logger().info(f"  Robot base frame: {self.robot_base_frame}")
        self.get_logger().info(f"  Top K: {self.top_k}")
        self.get_logger().info(f"  Yaw adjust mode: {self.yaw_adjust_mode}")
        self.get_logger().info(f"  Z offset: {self.z_offset}")
        self.get_logger().info(f"  Model loaded: {self.sampler is not None}")
        self.get_logger().info("Input topics:")
        self.get_logger().info(f"  /grasp_planning/selected_pointcloud")
        self.get_logger().info(f"  /grasp_planning/selected_transform")
        self.get_logger().info(f"  /grasp_planning/selected_arm")
        self.get_logger().info("Output topics:")
        self.get_logger().info(f"  /grasp_candidates")
        self.get_logger().info(f"  /filtered_grasp_poses_viz")
        self.get_logger().info("=" * 60)
    
    def _load_model(self):
        """GraspGen 모델 로드"""
        self.get_logger().info(f"Loading GraspGen config from: {self.config_path}")
        
        if not os.path.exists(self.config_path):
            self.get_logger().error(f"Config file not found: {self.config_path}")
            return
        
        # Config 로드
        cfg = None
        try:
            cfg = load_grasp_cfg(self.config_path)
            self.get_logger().info('Loaded config via load_grasp_cfg().')
        except Exception as e:
            self.get_logger().warn(f'load_grasp_cfg() failed: {e}. Falling back to yaml + EasyDict.')
            try:
                with open(self.config_path, 'r') as f:
                    cfg_dict = yaml.safe_load(f)
                cfg = edict(cfg_dict)
                self.get_logger().info('Loaded config via yaml.safe_load + EasyDict fallback.')
            except Exception as ee:
                self.get_logger().error(f'Config load fallback also failed: {ee}')
                return
        
        # Model 로드
        try:
            self.sampler = GraspGenSampler(cfg)
            self.get_logger().info('✅ GraspGen model loaded successfully.')
        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize GraspGen model: {e}")
    
    def transform_callback(self, msg: TransformStamped):
        """카메라->베이스 TF 저장"""
        if self.transform is None:
            self.get_logger().info('Received transform. Ready to process point clouds.')
        self.transform = msg
    
    def arm_callback(self, msg: String):
        """선택된 팔 정보 저장"""
        self.selected_arm = msg.data
        self.get_logger().info(f'Selected arm: {self.selected_arm}')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """포인트 클라우드 콜백 - GraspGen 추론 실행"""
        # 이미 퍼블리시했으면 스킵
        if self.has_published:
            return
        
        # TF가 필요
        if self.transform is None:
            self.get_logger().warn("Transform not yet received, skipping...", throttle_duration_sec=2)
            return
        
        # Model 체크    
        if self.sampler is None:
            self.get_logger().error("GraspGen model not loaded!")
            return
        
        # PointCloud2 -> numpy (벡터화 방식)
        # Step 1에서 이미 [N, 3] float32 dense array로 만들었으므로
        # 직접 버퍼에서 reshape하면 됨 (for 루프 불필요)
        try:
            # 포인트 수 계산
            n_points = msg.width * msg.height
            
            if n_points == 0:
                self.get_logger().warn("Empty point cloud received.")
                return
            
            # 직접 버퍼에서 numpy array로 변환 (매우 빠름)
            # msg.data는 이미 [x1,y1,z1, x2,y2,z2, ...] 형태의 float32 bytes
            pc_numpy = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, 3).copy()
            
        except Exception as e:
            self.get_logger().error(f"PointCloud2 to numpy failed: {e}")
            return
        
        self.get_logger().info(f"Processing point cloud: {pc_numpy.shape}")
        
        # Torch 텐서 변환
        try:
            pc_torch = torch.from_numpy(pc_numpy).float().to(self.device)
            if pc_torch.shape[0] > 2048:
                sel = np.random.choice(pc_torch.shape[0], 2048, replace=False)
                pc_torch = pc_torch[sel]
        except Exception as e:
            self.get_logger().error(f"Failed to move point cloud to {self.device}: {e}")
            return
        
        # GraspGen 추론
        try:
            res = self.sampler.sample(pc_torch)
            
            # 결과 파싱
            widths_t = None
            if isinstance(res, dict):
                grasps_t = res.get("grasps") or res.get("poses") or res.get("Ts") or res.get("T")
                scores_t = res.get("scores") or res.get("conf") or res.get("confidences") or res.get("quality")
                widths_t = res.get("widths") or res.get("openings")
            elif isinstance(res, (list, tuple)):
                if len(res) < 2:
                    raise ValueError(f"Unexpected tuple/list length from sampler.sample: len={len(res)}")
                grasps_t = res[0]
                scores_t = res[1]
                widths_t = res[2] if len(res) > 2 else None
            else:
                raise TypeError(f"Unexpected type from sampler.sample: {type(res)}")
            
            # numpy 변환
            grasps_np = (grasps_t.detach().cpu().numpy()
                         if torch.is_tensor(grasps_t) else np.asarray(grasps_t))
            scores_np = (scores_t.detach().cpu().numpy()
                         if torch.is_tensor(scores_t) else np.asarray(scores_t))
            scores_np = scores_np.reshape(-1)
            
            # 정렬 및 상위 K 선택
            order = np.argsort(-scores_np)
            if self.top_k is not None:
                order = order[:self.top_k]
            
            self.get_logger().info(
                f"Sampled {scores_np.shape[0]} grasps "
                f"(min={float(scores_np.min()):.5f}, max={float(scores_np.max()):.5f})."
            )
            
        except Exception as e:
            self.get_logger().error(f"GraspGen sampling failed: {e}")
            return
        
        # PoseArray 생성 및 변환
        pose_array_msg = self._create_pose_array(msg, grasps_np, order)
        
        if len(pose_array_msg.poses) == 0:
            self.get_logger().warn("No valid transformed poses to publish.")
            return
        
        # 퍼블리시
        self.candidates_pub.publish(pose_array_msg)
        self.viz_pub.publish(pose_array_msg)
        
        self.get_logger().info(
            f"✅ Published {len(pose_array_msg.poses)} grasp candidates "
            f"for {self.selected_arm or 'unknown'} arm"
        )
        self.has_published = True
    
    def _create_pose_array(
        self, 
        pc_msg: PointCloud2, 
        grasps_np: np.ndarray, 
        order: np.ndarray
    ) -> PoseArray:
        """Grasp 행렬들을 PoseArray로 변환"""
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = self.robot_base_frame
        
        for i in order:
            pose_matrix = grasps_np[i]
            
            try:
                # 4x4 -> (position, quaternion)
                p = tra.translation_from_matrix(pose_matrix)
                qw, qx, qy, qz = tra.quaternion_from_matrix(pose_matrix)
            except Exception as e:
                self.get_logger().warn(f"Invalid pose matrix skipped: {e}")
                continue
            
            # PoseStamped 생성 (카메라 프레임)
            ps = PoseStamped()
            ps.header = pc_msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.position.z = float(p[2])
            ps.pose.orientation.x = float(qx)
            ps.pose.orientation.y = float(qy)
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            
            # 카메라 -> 베이스 변환
            try:
                transformed_pose = do_transform_pose(ps.pose, self.transform)
            except Exception as e:
                self.get_logger().warn(f"Pose TF transform failed, skipped: {e}")
                continue
            
            # Yaw 보정
            q_wxyz = np.array([
                transformed_pose.orientation.w,
                transformed_pose.orientation.x,
                transformed_pose.orientation.y,
                transformed_pose.orientation.z
            ], dtype=np.float64)
            
            if self.yaw_adjust_mode == 'world':
                q_new = tra.quaternion_multiply(self.dq_flip_z_wxyz, q_wxyz)
            else:
                q_new = tra.quaternion_multiply(q_wxyz, self.dq_flip_z_wxyz)
            
            # 정규화
            nrm = np.linalg.norm(q_new)
            if nrm < 1e-12:
                self.get_logger().warn("Zero-norm quaternion after yaw adjust; skipping pose.")
                continue
            q_new /= nrm
            
            # Z축 방향 필터링 (옵션)
            rot_matrix = tra.quaternion_matrix(q_new)
            local_z_vector = rot_matrix[:3, 2]
            
            if self.filter_z_down:
                z_component = local_z_vector[2]
                if z_component >= -0.900:
                    continue
            
            # Z축 오프셋 적용
            p_offset = local_z_vector * self.z_offset
            new_x = transformed_pose.position.x + p_offset[0]
            new_y = transformed_pose.position.y + p_offset[1]
            new_z = transformed_pose.position.z + p_offset[2]
            
            # 최종 pose 설정
            transformed_pose.orientation.x = float(q_new[1])
            transformed_pose.orientation.y = float(q_new[2])
            transformed_pose.orientation.z = float(q_new[3])
            transformed_pose.orientation.w = float(q_new[0])
            transformed_pose.position.x = float(new_x)
            transformed_pose.position.y = float(new_y)
            transformed_pose.position.z = float(new_z)
            
            pose_array_msg.poses.append(transformed_pose)
        
        return pose_array_msg
    
    def reset_state(self):
        """상태 초기화"""
        self.transform = None
        self.selected_arm = None
        self.has_published = False


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
