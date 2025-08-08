#ifndef LATTICE_PLANNER_HPP
#define LATTICE_PLANNER_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "lattice_planner_rb/msg/wpnt_array.hpp"
#include "lattice_planner_rb/msg/ot_wpnt_array.hpp"
#include "lattice_planner_rb/frenet_converter.hpp"
#include "lattice_planner_rb/cubic_spline.hpp"

#include <vector>
#include <cmath>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace lattice_planner {

struct LatticeTrajectory {
public:
    std::vector<CartesianPoint> points;
    double cost;
    double progress_cost;
    double smoothness_cost;
    double lateral_cost;
    double collision_cost;
    int path_index;  // 0=center, -2,-1,+1,+2 for lateral offsets
    
    LatticeTrajectory() : cost(std::numeric_limits<double>::max()), path_index(0) {}
};

class LatticePlanner : public rclcpp::Node {
public:
    LatticePlanner();
    
private:
    // ROS2 subscribers and publishers
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_waypoints_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr planned_trajectory_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    
    // TF2
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Data members
    std::vector<Waypoint> global_path_;
    nav_msgs::msg::Odometry current_odom_;
    FrenetConverter frenet_converter_;
    
    bool global_path_received_;
    bool odom_received_;
    
    // Planning parameters
    double planning_horizon_;
    double time_step_;
    int num_lateral_samples_;
    double lateral_offset_max_;
    double longitudinal_step_;
    double lattice_planning_distance_;  // 격자 계획 거리 (파란색 구간)
    double spline_smoothing_distance_;  // 스플라인 보간 거리 (빨간색 구간)
    
    // Spline paths (5 paths: center + 4 lateral offsets)
    std::vector<CubicSplinePath> spline_paths_;
    std::vector<double> lateral_offsets_;  // [-2d, -d, 0, +d, +2d]
    
    // Cost weights
    double w_progress_;
    double w_smoothness_;
    double w_lateral_;
    double w_collision_;
    double w_curvature_;  // Enhanced: new curvature weight
    
    // Callbacks
    void globalWaypointsCallback(const nav_msgs::msg::Path::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void planningTimerCallback();
    
    // Core functions
    void loadGlobalPath();
    void generateSplinePaths();
    std::vector<LatticeTrajectory> generateTrajectories(const FrenetPoint& current_state);
    LatticeTrajectory selectBestTrajectory(const std::vector<LatticeTrajectory>& trajectories);
    double calculateCost(const LatticeTrajectory& trajectory);
    
    // Spline interpolation functions
    CubicSplinePath createOffsetPath(const std::vector<Waypoint>& reference_path, double lateral_offset);
    CubicSplinePath createFrenetOffsetPath(const std::vector<Waypoint>& reference_path, double d_offset);
    
    // Path planning functions
    std::vector<CartesianPoint> generateSmoothTrajectory(int path_index, const FrenetPoint& current_state);
    int findClosestWaypointIndex(double x, double y);
    
    // Utilities
    nav_msgs::msg::Path trajectoryToPath(const LatticeTrajectory& trajectory);
    visualization_msgs::msg::MarkerArray createTrajectoryMarkers(const std::vector<LatticeTrajectory>& trajectories);
    
    void publishTrajectory(const LatticeTrajectory& trajectory);
    void publishVisualization(const std::vector<LatticeTrajectory>& trajectories);
};

} // namespace lattice_planner

#endif // LATTICE_PLANNER_HPP
