# redbull_msgs

이 패키지는 redbull 프로젝트에서 사용하는 커스텀 메시지(ObstacleArray, ObstacleWpnt)를 포함합니다.

- msg/ObstacleArray.msg
- msg/ObstacleWpnt.msg

빌드 후 Python에서 다음과 같이 import할 수 있습니다:

```python
from redbull_msgs.msg import ObstacleArray, ObstacleWpnt
```

---

## 빌드 방법

```bash
cd ~/ros2_ws
colcon build --packages-select redbull_msgs
source install/setup.bash
```
