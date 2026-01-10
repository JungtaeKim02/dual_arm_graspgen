#!/usr/bin/env python3
"""
Reachability Map Generator for RBY1 Dual Arm (URDF-based FK)

URDF 파일을 직접 로드하여 정확한 Forward Kinematics로 
Reachability Map을 생성합니다.

Features:
- URDF에서 joint axis, limits, link lengths 자동 추출
- 정확한 FK 계산 (DH 파라미터 또는 transformation chain)
- torso_5 프레임 기준 도달 가능 영역 생성

Usage:
    python reachability_map_generator_urdf.py \
        --urdf /home/kimjungtae/rby1_ws/rby1_ros2_ws/src/rby1_description/urdf/rby1_inha.urdf \
        --output_dir ./reachability_maps
        
Output:
    - left_arm_reachability.npz
    - right_arm_reachability.npz
"""

import numpy as np
import argparse
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
import xml.etree.ElementTree as ET


@dataclass
class JointInfo:
    """Joint 정보"""
    name: str
    type: str  # 'revolute', 'prismatic', 'fixed', 'continuous'
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    limits: Tuple[float, float]  # (lower, upper) in radians


@dataclass
class LinkInfo:
    """Link 정보"""
    name: str
    

class URDFParser:
    """URDF 파일 파서"""
    
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()
        
        self.joints: Dict[str, JointInfo] = {}
        self.links: Dict[str, LinkInfo] = {}
        
        self._parse()
    
    def _parse_origin(self, origin_elem) -> Tuple[np.ndarray, np.ndarray]:
        """Parse <origin xyz="..." rpy="..."/>"""
        xyz = np.array([0.0, 0.0, 0.0])
        rpy = np.array([0.0, 0.0, 0.0])
        
        if origin_elem is not None:
            xyz_str = origin_elem.get('xyz', '0 0 0')
            rpy_str = origin_elem.get('rpy', '0 0 0')
            xyz = np.array([float(x) for x in xyz_str.split()])
            rpy = np.array([float(x) for x in rpy_str.split()])
        
        return xyz, rpy
    
    def _parse_axis(self, axis_elem) -> np.ndarray:
        """Parse <axis xyz="..."/>"""
        if axis_elem is not None:
            axis_str = axis_elem.get('xyz', '0 0 1')
            return np.array([float(x) for x in axis_str.split()])
        return np.array([0.0, 0.0, 1.0])  # default Z axis
    
    def _parse_limits(self, limit_elem) -> Tuple[float, float]:
        """Parse <limit lower="..." upper="..."/>"""
        if limit_elem is not None:
            lower = float(limit_elem.get('lower', '-3.14159'))
            upper = float(limit_elem.get('upper', '3.14159'))
            return (lower, upper)
        return (-np.pi, np.pi)
    
    def _parse(self):
        """Parse URDF file"""
        # Parse links
        for link_elem in self.root.findall('link'):
            name = link_elem.get('name')
            self.links[name] = LinkInfo(name=name)
        
        # Parse joints
        for joint_elem in self.root.findall('joint'):
            name = joint_elem.get('name')
            joint_type = joint_elem.get('type')
            
            parent = joint_elem.find('parent').get('link')
            child = joint_elem.find('child').get('link')
            
            origin_xyz, origin_rpy = self._parse_origin(joint_elem.find('origin'))
            axis = self._parse_axis(joint_elem.find('axis'))
            limits = self._parse_limits(joint_elem.find('limit'))
            
            self.joints[name] = JointInfo(
                name=name,
                type=joint_type,
                parent_link=parent,
                child_link=child,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
                limits=limits
            )
    
    def get_chain(self, start_link: str, end_link: str) -> List[JointInfo]:
        """Get kinematic chain from start_link to end_link"""
        # Build parent-child map
        child_to_joint = {j.child_link: j for j in self.joints.values()}
        
        chain = []
        current = end_link
        
        while current != start_link:
            if current not in child_to_joint:
                raise ValueError(f"Cannot find path from {start_link} to {end_link}")
            
            joint = child_to_joint[current]
            chain.append(joint)
            current = joint.parent_link
        
        chain.reverse()
        return chain


class TransformationUtils:
    """Transformation 유틸리티"""
    
    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """X축 회전 행렬"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        """Y축 회전 행렬"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """Z축 회전 행렬"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """RPY to 4x4 rotation matrix (URDF convention: Rz * Ry * Rx)"""
        Rx = TransformationUtils.rotation_matrix_x(roll)
        Ry = TransformationUtils.rotation_matrix_y(pitch)
        Rz = TransformationUtils.rotation_matrix_z(yaw)
        return Rz @ Ry @ Rx
    
    @staticmethod
    def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
        """Translation 4x4 matrix"""
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        """Axis-angle to 4x4 rotation matrix (Rodrigues' formula)"""
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        T = np.eye(4)
        T[:3, :3] = R
        return T


class ReachabilityMapGeneratorURDF:
    """
    URDF 기반 Reachability Map 생성기
    
    정확한 FK를 사용하여 도달 가능 영역을 계산
    """
    
    def __init__(
        self,
        urdf_path: str,
        resolution: float = 0.02,
        workspace_bounds: Tuple[Tuple[float, float], ...] = (
            (-0.8, 0.8),   # x
            (-0.8, 0.8),   # y
            (-0.8, 0.8),   # z
        ),
        n_samples: int = 500000,
        base_link: str = 'link_torso_5',
    ):
        self.resolution = resolution
        self.workspace_bounds = workspace_bounds
        self.n_samples = n_samples
        self.base_link = base_link
        
        # Grid dimensions
        self.grid_shape = tuple(
            int((bounds[1] - bounds[0]) / resolution) + 1
            for bounds in workspace_bounds
        )
        
        # Parse URDF
        print(f"Loading URDF from: {urdf_path}")
        self.urdf = URDFParser(urdf_path)
        
        # Arm configurations
        self.arm_configs = {
            'left_arm': {
                'end_link': 'ee_left',
                'joints': ['left_arm_0', 'left_arm_1', 'left_arm_2', 
                          'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6']
            },
            'right_arm': {
                'end_link': 'ee_right', 
                'joints': ['right_arm_0', 'right_arm_1', 'right_arm_2',
                          'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6']
            }
        }
        
        # Cache kinematic chains
        self._chains = {}
        for arm_name, config in self.arm_configs.items():
            try:
                self._chains[arm_name] = self.urdf.get_chain(
                    self.base_link, 
                    config['end_link']
                )
                print(f"  {arm_name} chain: {len(self._chains[arm_name])} joints")
            except Exception as e:
                print(f"  Warning: Could not build chain for {arm_name}: {e}")
    
    def forward_kinematics(
        self, 
        arm_name: str, 
        joint_values: Dict[str, float]
    ) -> np.ndarray:
        """
        URDF 기반 정확한 Forward Kinematics
        
        Args:
            arm_name: 'left_arm' or 'right_arm'
            joint_values: {joint_name: angle} dictionary
            
        Returns:
            4x4 transformation matrix (base_link to end_effector)
        """
        chain = self._chains.get(arm_name)
        if chain is None:
            raise ValueError(f"No chain found for {arm_name}")
        
        T = np.eye(4)
        
        for joint in chain:
            # Joint origin transformation
            T_origin = TransformationUtils.translation_matrix(*joint.origin_xyz)
            T_rpy = TransformationUtils.rpy_to_matrix(*joint.origin_rpy)
            T = T @ T_origin @ T_rpy
            
            # Joint rotation (if not fixed)
            if joint.type in ['revolute', 'continuous']:
                angle = joint_values.get(joint.name, 0.0)
                T_joint = TransformationUtils.axis_angle_matrix(joint.axis, angle)
                T = T @ T_joint
            elif joint.type == 'prismatic':
                displacement = joint_values.get(joint.name, 0.0)
                T_joint = TransformationUtils.translation_matrix(
                    *(joint.axis * displacement)
                )
                T = T @ T_joint
        
        return T
    
    def get_ee_position(
        self, 
        arm_name: str, 
        joint_values: Dict[str, float]
    ) -> np.ndarray:
        """Get end-effector position from FK"""
        T = self.forward_kinematics(arm_name, joint_values)
        return T[:3, 3]
    
    def random_joint_config(self, arm_name: str) -> Dict[str, float]:
        """Generate random joint configuration within limits"""
        config = self.arm_configs[arm_name]
        joint_values = {}
        
        for joint_name in config['joints']:
            joint = self.urdf.joints.get(joint_name)
            if joint and joint.type in ['revolute', 'continuous']:
                lower, upper = joint.limits
                joint_values[joint_name] = np.random.uniform(lower, upper)
        
        return joint_values
    
    def position_to_grid_index(
        self, 
        position: np.ndarray
    ) -> Optional[Tuple[int, int, int]]:
        """Convert 3D position to grid index"""
        indices = []
        for i, (pos, bounds) in enumerate(zip(position, self.workspace_bounds)):
            if pos < bounds[0] or pos > bounds[1]:
                return None
            idx = int((pos - bounds[0]) / self.resolution)
            idx = min(idx, self.grid_shape[i] - 1)
            indices.append(idx)
        return tuple(indices)
    
    def generate_reachability_map(self, arm_name: str) -> np.ndarray:
        """
        Generate reachability map for specified arm
        
        Returns:
            occupancy_grid: (nx, ny, nz) uint8 array
        """
        print(f"\nGenerating reachability map for {arm_name}...")
        print(f"  Grid shape: {self.grid_shape}")
        print(f"  Resolution: {self.resolution}m")
        print(f"  Samples: {self.n_samples}")
        print(f"  Base link: {self.base_link}")
        
        if arm_name not in self._chains:
            print(f"  ERROR: No kinematic chain for {arm_name}")
            return np.zeros(self.grid_shape, dtype=np.uint8)
        
        occupancy = np.zeros(self.grid_shape, dtype=np.uint8)
        valid_samples = 0
        
        for i in range(self.n_samples):
            if i % 50000 == 0:
                print(f"  Progress: {i}/{self.n_samples} (valid: {valid_samples})")
            
            # Random joint configuration
            joint_values = self.random_joint_config(arm_name)
            
            # Forward kinematics
            try:
                ee_pos = self.get_ee_position(arm_name, joint_values)
                
                # Record in grid
                grid_idx = self.position_to_grid_index(ee_pos)
                if grid_idx is not None:
                    occupancy[grid_idx] = 1
                    valid_samples += 1
                    
            except Exception as e:
                continue
        
        # Statistics
        reachable_cells = np.sum(occupancy)
        total_cells = np.prod(self.grid_shape)
        coverage = reachable_cells / total_cells * 100
        
        print(f"  Valid samples: {valid_samples}/{self.n_samples}")
        print(f"  Reachable cells: {reachable_cells}/{total_cells} ({coverage:.2f}%)")
        
        return occupancy
    
    def save_reachability_map(
        self, 
        occupancy: np.ndarray, 
        arm_name: str,
        output_path: str
    ):
        """Save reachability map to file"""
        data = {
            'occupancy': occupancy,
            'resolution': self.resolution,
            'workspace_bounds': self.workspace_bounds,
            'grid_shape': self.grid_shape,
            'arm_name': arm_name,
            'base_link': self.base_link,
            'n_samples': self.n_samples,
        }
        
        np.savez_compressed(output_path, **data)
        print(f"Saved reachability map to: {output_path}")
    
    def generate_and_save_all(self, output_dir: str):
        """Generate and save reachability maps for both arms"""
        os.makedirs(output_dir, exist_ok=True)
        
        for arm_name in self.arm_configs.keys():
            occupancy = self.generate_reachability_map(arm_name)
            self.save_reachability_map(
                occupancy,
                arm_name,
                os.path.join(output_dir, f'{arm_name}_reachability.npz')
            )
        
        print("\n✅ All reachability maps generated successfully!")
    
    def visualize_reachability(self, arm_name: str, occupancy: np.ndarray):
        """Simple visualization of reachability map (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Get occupied voxel centers
            indices = np.argwhere(occupancy > 0)
            if len(indices) == 0:
                print("No reachable points to visualize")
                return
            
            # Convert to world coordinates
            points = np.zeros((len(indices), 3))
            for i, (ix, iy, iz) in enumerate(indices):
                points[i, 0] = self.workspace_bounds[0][0] + ix * self.resolution
                points[i, 1] = self.workspace_bounds[1][0] + iy * self.resolution
                points[i, 2] = self.workspace_bounds[2][0] + iz * self.resolution
            
            # Subsample for visualization
            if len(points) > 5000:
                idx = np.random.choice(len(points), 5000, replace=False)
                points = points[idx]
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 2], cmap='viridis', s=1, alpha=0.5)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Reachability Map - {arm_name}')
            plt.savefig(f'{arm_name}_reachability.png', dpi=150)
            print(f"Saved visualization to {arm_name}_reachability.png")
            plt.close()
            
        except ImportError:
            print("matplotlib not available for visualization")


def load_reachability_map(filepath: str) -> dict:
    """Load saved reachability map"""
    data = np.load(filepath, allow_pickle=True)
    return {
        'occupancy': data['occupancy'],
        'resolution': float(data['resolution']),
        'workspace_bounds': tuple(map(tuple, data['workspace_bounds'])),
        'grid_shape': tuple(data['grid_shape']),
        'arm_name': str(data['arm_name']),
        'base_link': str(data['base_link']) if 'base_link' in data else 'link_torso_5',
        'n_samples': int(data['n_samples']) if 'n_samples' in data else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate Reachability Maps using URDF-based FK'
    )
    parser.add_argument(
        '--urdf', type=str,
        default='/home/kimjungtae/rby1_ws/rby1_ros2_ws/src/rby1_description/urdf/rby1_inha.urdf',
        help='Path to URDF file'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/home/kimjungtae/graspgen_ws/GraspGen/dual_arm_grasp/reachability_maps',
        help='Output directory for reachability maps'
    )
    parser.add_argument(
        '--resolution', type=float, default=0.02,
        help='Grid resolution in meters'
    )
    parser.add_argument(
        '--n_samples', type=int, default=500000,
        help='Number of random FK samples'
    )
    parser.add_argument(
        '--base_link', type=str, default='link_torso_5',
        help='Base link for FK (reference frame)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization images'
    )
    
    args = parser.parse_args()
    
    # Check URDF exists
    if not os.path.exists(args.urdf):
        print(f"ERROR: URDF file not found: {args.urdf}")
        return
    
    # Create generator
    generator = ReachabilityMapGeneratorURDF(
        urdf_path=args.urdf,
        resolution=args.resolution,
        n_samples=args.n_samples,
        base_link=args.base_link,
    )
    
    # Generate maps
    generator.generate_and_save_all(args.output_dir)
    
    # Optional visualization
    if args.visualize:
        for arm_name in generator.arm_configs.keys():
            rm_path = os.path.join(args.output_dir, f'{arm_name}_reachability.npz')
            if os.path.exists(rm_path):
                data = load_reachability_map(rm_path)
                generator.visualize_reachability(arm_name, data['occupancy'])


if __name__ == '__main__':
    main()
