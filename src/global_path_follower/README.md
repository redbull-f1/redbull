# Global Path Follower

A simple ROS2 package for following pre-defined global waypoints in F1TENTH simulation.

## Features

- Loads waypoints from CSV files
- Pure pursuit controller for path following
- Visualization of global path and vehicle position
- Simple and clean code structure

## Usage

### Building the package

```bash
cd /home/jeong/sim_ws
colcon build --packages-select global_path_follower
source install/setup.bash
```

### Running the package

1. Start F1TENTH simulation:
```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

2. Launch global path follower:
```bash
ros2 launch global_path_follower global_path_follower.launch.py
```

### Parameters

- `waypoint_file`: CSV file name in waypoints/ directory (default: "redbull_0.csv")
- `map_frame`: Map coordinate frame (default: "map")
- `control_frequency`: Control loop frequency in Hz (default: 20.0)
- `lookahead_distance`: Pure pursuit lookahead distance in meters (default: 3.0)
- `max_speed`: Maximum vehicle speed in m/s (default: 2.0)
- `min_speed`: Minimum vehicle speed in m/s (default: 0.5)
- `wheelbase`: Vehicle wheelbase in meters (default: 0.33)

### Topics

#### Subscribed Topics
- `/ego_racecar/odom` (nav_msgs/Odometry): Vehicle odometry

#### Published Topics
- `/cmd_vel` (geometry_msgs/Twist): Vehicle control commands
- `/global_path` (nav_msgs/Path): Global path for visualization
- `/global_path_vis` (visualization_msgs/MarkerArray): Visualization markers

## CSV Format

The waypoint CSV file should have the following columns:
- s_m: Path length coordinate
- x_m: X position in global frame
- y_m: Y position in global frame
- psi_rad: Heading angle in radians
- kappa_radpm: Curvature in rad/m
- vx_mps: Velocity in m/s
- ax_mps2: Acceleration in m/s^2
- d_right: Distance to right boundary
- d_left: Distance to left boundary
