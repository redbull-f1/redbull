# 🚗 RedBull - 동적 차량 감지

TinyCenterSpeed 신경망과 LiDAR 데이터 처리를 이용한 ROS2 동적 차량 감지 패키지입니다.

## 🎯 주요 기능

- **동적 차량 감지**: TinyCenterSpeed 신경망을 이용한 실시간 감지
- **LiDAR 처리**: 고급 LiDAR 스캔 전처리 및 좌표 변환  
- **ROS2 Bag 처리**: 상대 차량 데이터를 포함한 bag 파일을 CSV로 변환
- **RViz2 시각화**: 감지된 객체의 실시간 시각화
- **다중 객체 추적**: 여러 차량을 동시에 감지

## 🚀 빠른 시작

### 1. 클론 및 설치
```bash
cd ~/ros2_ws/src
git clone https://github.com/redbull-f1/redbull.git
cd redbull

# 설치 스크립트 실행
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### 2. 패키지 빌드
```bash
cd ~/ros2_ws
colcon build --packages-select redbull
source install/setup.bash
```

### 3. 감지기 실행
```bash
# 방법 1: 런치 파일 사용 (권장)
ros2 launch redbull redbull_detector.launch.py

# 방법 2: 직접 노드 실행
ros2 run redbull dynamic_vehicle_detector

# 커스텀 파라미터로 런치 파일 사용
ros2 launch redbull redbull_detector.launch.py \
    model_path:=./my_model.pt \
    detection_threshold:=0.5 \
    num_opponents:=3

# 다른 터미널에서 RViz2로 시각화
rviz2
```

## 📋 주요 구성 요소

### 🔧 핵심 스크립트

- **`dynamic_vehicle_detector_ros2.py`**: TinyCenterSpeed 신경망을 이용한 실시간 차량 감지 노드
- **`parse_bag_ros2.py`**: 시간 동기화 및 전역 좌표에서 ego LiDAR 좌표계로 변환하는 bag 파일 변환기
- **`merge_bags.py`**: ego 토픽들(`/scan`, `/car_state/odom`, `/tf`, `/tf_static`)과 상대방 토픽들(`/opp/car_state/odom`)을 시간 정렬하여 병합
- **`car_tracker.py`**: LiDAR 전처리 및 좌표계 유틸리티

### 🧠 훈련 모듈 (`train/`)

- **`train_CenterSpeed_dense.ipynb`**: 검증을 포함한 완전한 훈련 파이프라인
- **`models/`**: 신경망 아키텍처
- **`trained_models/`**: 저장된 모델 파일

## 📊 토픽

### 📥 구독하는 토픽
- `/scan` (sensor_msgs/LaserScan): LiDAR 스캔 데이터

### 📤 발행하는 토픽
- `/objects` (visualization_msgs/MarkerArray): RViz2용 감지된 객체
- `/objects_data` (std_msgs/Float32MultiArray): 객체 데이터 [num_objects, x1, y1, vx1, vy1, yaw1, size1, ...]

## 🗂️ Bag 파일 처리

```bash
# 상대 차량 데이터를 포함한 bag 파일을 CSV로 변환
ros2 run redbull parse_bag_csv input.db3 --output result.csv

# MarkerArray가 포함된 시각화 bag 파일 생성 (시간 동기화됨)
ros2 run redbull parse_bag_ros2 input.db3 --create-bag --visualize

# ego와 상대방 bag 파일을 시간 정렬하여 병합
python3 merge_bags.py ego_bag.db3 opp_bag.db3 --output merged.db3
```

## ⚙️ 파라미터

감지기 노드는 다음 파라미터를 지원합니다:

```bash
ros2 run redbull dynamic_vehicle_detector --ros-args \
    -p model_path:="./my_model.pt" \
    -p detection_threshold:=0.5 \
    -p num_opponents:=3
```

## 📊 객체 데이터 형식

`/objects_data` 토픽은 다음 형식으로 데이터를 발행합니다:
```
[num_objects, x1, y1, vx1, vy1, yaw1, size1, x2, y2, vx2, vy2, yaw2, size2, ...]
```

각 객체는 6개의 값을 가집니다:
- `x, y`: LiDAR 좌표계에서의 위치 (미터)
- `vx, vy`: 속도 성분 (m/s)
- `yaw`: 방향각 (라디안)
- `size`: 시각화를 위한 객체 크기 (미터)

## 🔧 요구사항

### 시스템 요구사항
- **OS**: Ubuntu 22.04 (권장)
- **ROS2**: Humble Hawksbill
- **Python**: 3.8+

### Python 의존성 (스크립트로 자동 설치)
- NumPy >= 1.21.0
- PyTorch >= 1.12.0
- TorchVision >= 0.13.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- WandB >= 0.12.0 (선택사항, 훈련용)

지금 종명 노트북 상황
numpy 1.26.4
torch 2.7.1+cu126
Name: rclpy
Version: 3.3.17

### ROS2 의존성 (자동 설치)
- rclpy, sensor_msgs, geometry_msgs, visualization_msgs, std_msgs
- tf2_ros, rosbag2_py, nav_msgs, builtin_interfaces

## 🐛 문제 해결

### 모델을 찾을 수 없는 오류
```bash
# 훈련된 모델이 존재하는지 확인
ls ~/ros2_ws/src/redbull/train/trained_models/

# 노트북을 사용하여 새 모델 훈련
cd ~/ros2_ws/src/redbull/train
jupyter notebook train_CenterSpeed_dense.ipynb
```

### Import 오류
```bash
# 심볼릭 링크로 다시 빌드
cd ~/ros2_ws
colcon build --packages-select redbull --symlink-install
source install/setup.bash
```

### PyTorch CUDA 문제
```bash
# CUDA 사용 가능 여부 확인
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 설치 스크립트가 CUDA/CPU 선택을 자동으로 처리합니다
```

## 📁 패키지 구조

```
redbull/
├── dynamic_vehicle_detector_ros2.py  # 메인 감지 노드
├── parse_bag_ros2.py                 # 시간 동기화 bag 처리
├── merge_bags.py                     # ego/상대방 bag 병합
├── car_tracker.py                    # LiDAR 유틸리티
├── train/                           # 훈련 파이프라인
│   ├── models/                      # 신경망 아키텍처
│   ├── trained_models/              # 저장된 모델
│   └── train_CenterSpeed_dense.ipynb
├── launch/                          # 런치 파일
│   ├── redbull_detector.launch.py   # 감지기 런치
│   └── redbull_bag_processor.launch.py # bag 처리기 런치
├── requirements.txt                 # Python 의존성
├── install_dependencies.sh          # 자동 설치 스크립트
├── setup.py                        # 패키지 설정
├── package.xml                      # ROS2 패키지 매니페스트
└── README.md                        # 이 파일
```

## 🎓 사용법 예시

### 실시간 감지 실행
```bash
# 런치 파일로 모든 설정과 함께 실행
ros2 launch redbull redbull_detector.launch.py

# 특정 모델과 파라미터로 실행
ros2 launch redbull redbull_detector.launch.py \
    model_path:=train/trained_models/my_custom_model.pt \
    detection_threshold:=0.4 \
    num_opponents:=10
```

### Bag 파일 처리
```bash
# Bag 파일을 CSV로 변환
ros2 launch redbull redbull_bag_processor.launch.py \
    bag_file:=data/test_data.db3 \
    output_file:=results/parsed_data.csv
```

### 모델 훈련
```bash
# 훈련 디렉토리로 이동
cd ~/ros2_ws/src/redbull/train

# Jupyter 노트북 실행
jupyter notebook train_CenterSpeed_dense.ipynb

# 또는 직접 실행 (백그라운드)
nohup jupyter notebook --no-browser --port=8888 > jupyter.log 2>&1 &
```

## 🔧 고급 설정

### 모델 경로 설정
패키지는 다음 순서로 모델을 찾습니다:
1. 절대 경로로 지정된 모델
2. `train/trained_models/` 디렉토리의 모델
3. 패키지 설치 디렉토리의 모델

### 성능 최적화
- GPU 가속을 위해 CUDA 버전의 PyTorch 설치 권장
- 실시간 처리를 위해 `detection_threshold` 조정
- 메모리 사용량 최적화를 위해 `num_opponents` 제한

## 🤝 기여하기

1. 저장소를 포크합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/new-feature`)
3. 변경사항을 커밋합니다 (`git commit -am 'Add new feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/new-feature`)
5. Pull Request를 생성합니다

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 있습니다 - 자세한 내용은 LICENSE 파일을 참조하세요.

## 📬 연락처

- **작성자**: Harry
- **이메일**: your_email@example.com
- **프로젝트**: [GitHub Repository URL]

## 🙏 감사의 말

- TinyCenterSpeed 신경망 아키텍처
- 뛰어난 문서화를 제공한 ROS2 커뮤니티
- 딥러닝 프레임워크를 제공한 PyTorch 팀
