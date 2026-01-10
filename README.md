# Dual Arm Grasp Planning Pipeline for RBY1

RBY1 로봇의 양팔 파지 작업을 위한 모듈화된 grasp planning 파이프라인입니다.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Dual Arm Grasp Planning Pipeline                      │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  Reachability    │
                    │  Map Generator   │  (사전 실행 - 오프라인)
                    │  (Step 0)        │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   L.RM / R.RM    │  (저장된 .npz 파일)
                    └────────┬─────────┘
                             │
┌────────────────────────────┼────────────────────────────────┐
│                            │                                 │
│  ┌─────────────────────────▼─────────────────────────────┐  │
│  │                   Runtime Pipeline                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐       ┌──────────────┐                    │
│  │ Left Camera  │       │ Right Camera │                    │
│  │ /left_cam/   │       │ /right_cam/  │                    │
│  │ segmented_   │       │ segmented_   │                    │
│  │ pointcloud   │       │ pointcloud   │                    │
│  └──────┬───────┘       └──────┬───────┘                    │
│         │                      │                             │
│         └──────────┬───────────┘                             │
│                    │                                         │
│         ┌──────────▼──────────┐                             │
│         │  Point Cloud        │                             │
│         │  Preprocessing Node │  (Node 1)                   │
│         │                     │                             │
│         │  - Downsample to    │                             │
│         │    [2048, 3]        │                             │
│         └──────────┬──────────┘                             │
│                    │                                         │
│         ┌──────────▼──────────┐                             │
│         │  /grasp_planning/   │                             │
│         │  left_pointcloud    │  L.PC                       │
│         │  right_pointcloud   │  R.PC                       │
│         └──────────┬──────────┘                             │
│                    │                                         │
│         ┌──────────▼──────────┐                             │
│         │  Arm Selection      │                             │
│         │  Node (Node 2)      │                             │
│         │                     │                             │
│         │  - Compare L.PC     │                             │
│         │    with L.RM        │                             │
│         │  - Compare R.PC     │                             │
│         │    with R.RM        │                             │
│         │  - Score =          │                             │
│         │    (included pts)   │                             │
│         │    / 2048           │                             │
│         │  - Select higher    │                             │
│         │    score arm        │                             │
│         └──────────┬──────────┘                             │
│                    │                                         │
│         ┌──────────▼──────────┐                             │
│         │  /grasp_planning/   │                             │
│         │  selected_arm       │  "left" or "right"          │
│         │  selected_pointcloud│  선택된 P.C                  │
│         │  selected_transform │  cam→base TF                │
│         └──────────┬──────────┘                             │
│                    │                                         │
│         ┌──────────▼──────────┐                             │
│         │  GraspGen Inference │                             │
│         │  Node (Node 3)      │                             │
│         │                     │                             │
│         │  - Load GraspGen    │                             │
│         │    model            │                             │
│         │  - Generate grasp   │                             │
│         │    poses            │                             │
│         │  - Transform to     │                             │
│         │    base frame       │                             │
│         │  - Post-processing  │                             │
│         └──────────┬──────────┘                             │
│                    │                                         │
│         ┌──────────▼──────────┐                             │
│         │  /grasp_candidates  │  PoseArray                  │
│         │  (base frame)       │                             │
│         └─────────────────────┘                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 노드 설명

### 0. Reachability Map Generator (오프라인)

사전에 양팔의 도달 가능 영역을 계산하여 저장합니다.

```bash
# Reachability Map 생성
python reachability_map_generator.py --output_dir ./reachability_maps --n_samples 100000

# 출력:
# - reachability_maps/left_arm_reachability.npz
# - reachability_maps/right_arm_reachability.npz
```

### 1. Point Cloud Preprocessing Node

GroundedSAM2에서 세그먼트된 포인트 클라우드를 수신하여 2048개 포인트로 다운샘플링합니다.

**입력 토픽:**
- `/left_camera/segmented_pointcloud` (sensor_msgs/PointCloud2)
- `/right_camera/segmented_pointcloud` (sensor_msgs/PointCloud2)

**출력 토픽:**
- `/grasp_planning/left_pointcloud` (sensor_msgs/PointCloud2)
- `/grasp_planning/right_pointcloud` (sensor_msgs/PointCloud2)
- `/grasp_planning/pointclouds_ready` (std_msgs/Bool)

### 2. Arm Selection Node

Reachability Map을 기반으로 더 적합한 팔을 선택합니다.

**점수 계산:**
```
Score_L = (L.RM에 포함된 L.PC 점 개수) / 2048
Score_R = (R.RM에 포함된 R.PC 점 개수) / 2048
```

더 높은 점수의 팔이 target planning group으로 선정됩니다.

**입력 토픽:**
- `/grasp_planning/left_pointcloud`
- `/grasp_planning/right_pointcloud`
- `/grasp_planning/pointclouds_ready`
- TF2 (카메라 프레임 → torso_5/base)

**출력 토픽:**
- `/grasp_planning/selected_arm` (std_msgs/String)
- `/grasp_planning/selected_pointcloud` (sensor_msgs/PointCloud2)
- `/grasp_planning/selected_transform` (geometry_msgs/TransformStamped)
- `/grasp_planning/arm_scores` (std_msgs/String, JSON format)

### 3. GraspGen Inference Node

선택된 팔의 포인트 클라우드로 GraspGen 모델을 실행하여 grasp pose를 생성합니다.

**입력 토픽:**
- `/grasp_planning/selected_pointcloud`
- `/grasp_planning/selected_transform`
- `/grasp_planning/selected_arm`

**출력 토픽:**
- `/grasp_candidates` (geometry_msgs/PoseArray)
- `/filtered_grasp_poses_viz` (geometry_msgs/PoseArray)

## 사용법

### 1. Reachability Map 사전 생성

```bash
cd /home/kimjungtae/graspgen_ws/GraspGen/dual_arm_grasp
python reachability_map_generator.py
```

### 2. ROS2 패키지 빌드

```bash
cd /home/kimjungtae/graspgen_ws
colcon build --packages-select dual_arm_grasp
source install/setup.bash
```

### 3. 파이프라인 실행

```bash
# Launch 파일로 전체 파이프라인 실행
ros2 launch dual_arm_grasp dual_arm_grasp.launch.py

# 또는 개별 노드 실행
ros2 run dual_arm_grasp pointcloud_preprocessing_node
ros2 run dual_arm_grasp arm_selection_node
ros2 run dual_arm_grasp graspgen_inference_node
```

### 4. 커스텀 파라미터

```bash
ros2 launch dual_arm_grasp dual_arm_grasp.launch.py \
    left_pc_topic:=/detection_node/left_segmented_pointcloud \
    right_pc_topic:=/detection_node/right_segmented_pointcloud \
    base_frame:=base \
    top_k:=30
```

## 파라미터

### Point Cloud Preprocessing Node
| Parameter | Default | Description |
|-----------|---------|-------------|
| `left_pc_topic` | `/left_camera/segmented_pointcloud` | 왼쪽 카메라 포인트 클라우드 토픽 |
| `right_pc_topic` | `/right_camera/segmented_pointcloud` | 오른쪽 카메라 포인트 클라우드 토픽 |
| `target_points` | `2048` | 다운샘플링 목표 포인트 수 |

### Arm Selection Node
| Parameter | Default | Description |
|-----------|---------|-------------|
| `reachability_map_dir` | `./reachability_maps` | RM 파일 디렉토리 |
| `left_camera_frame` | `left_camera_color_optical_frame` | 왼쪽 카메라 프레임 |
| `right_camera_frame` | `right_camera_color_optical_frame` | 오른쪽 카메라 프레임 |
| `torso_5_frame` | `link_torso_5` | Torso 5 프레임 |
| `score_threshold` | `0.1` | 최소 도달 가능 점수 |

### GraspGen Inference Node
| Parameter | Default | Description |
|-----------|---------|-------------|
| `config_path` | `/models/checkpoints/graspgen_franka_panda.yml` | GraspGen 설정 파일 |
| `robot_base_frame` | `base` | 로봇 베이스 프레임 |
| `top_k` | `20` | 상위 K개 grasp pose 선택 |
| `yaw_adjust_mode` | `local` | Yaw 보정 모드 (`local`/`world`) |
| `z_offset` | `0.0928` | 그리퍼 Z축 오프셋 |



## 파일 구조

```
dual_arm_grasp/
├── __init__.py
├── CMakeLists.txt
├── package.xml
├── README.md
├── reachability_map_generator.py    # Step 0: RM 생성
├── pointcloud_preprocessing_node.py # Node 1: PC 전처리
├── arm_selection_node.py            # Node 2: 팔 선택
├── graspgen_inference_node.py       # Node 3: 추론
├── launch/
│   └── dual_arm_grasp.launch.py
└── reachability_maps/               # 생성된 RM 저장
    ├── left_arm_reachability.npz
    └── right_arm_reachability.npz
```

## 의존성

- ROS2 Humble
- Python 3.10+
- PyTorch
- NumPy
- trimesh
- sensor_msgs_py크
- tf2_ros
- tf2_geometry_msgs
- GraspGen (graspgen 프레임워)
