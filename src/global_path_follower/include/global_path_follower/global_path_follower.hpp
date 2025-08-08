#ifndef GLOBAL_PATH_FOLLOWER_HPP_
#define GLOBAL_PATH_FOLLOWER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "global_path_follower/msg/wpnt.hpp"
#include "global_path_follower/msg/wpnt_array.hpp"
#include "global_path_follower/waypoint_loader.hpp"

namespace global_path_follower {

struct VehicleState {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double speed = 0.0;
};

class GlobalPathFollower : public rclcpp::Node {
public:
    GlobalPathFollower(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~GlobalPathFollower() = default;

private:
    // Callback functions
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void timerCallback();
    
    // Core functions
    void loadParameters();
    void followGlobalPath();
    void publishVisualization();
    void publishGlobalWaypoints();
    
    // Pure pursuit controller
    ackermann_msgs::msg::AckermannDriveStamped calculatePurePursuitCommand();
    
    // ROS2 components
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub_;
    rclcpp::Publisher<global_path_follower::msg::WpntArray>::SharedPtr global_waypoints_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Core components
    std::unique_ptr<WaypointLoader> waypoint_loader_;
    VehicleState vehicle_state_;
    
    // Configuration
    std::string waypoint_file_;
    std::string map_frame_;
    double control_frequency_;
    double lookahead_distance_;
    double max_speed_;
    double min_speed_;
    double wheelbase_;
    
    // State
    bool odom_received_;
    nav_msgs::msg::Path global_path_;
};

} // namespace global_path_follower

#endif // GLOBAL_PATH_FOLLOWER_HPP_
