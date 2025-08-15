#ifndef SPLINE_PLANNER_HPP
#define SPLINE_PLANNER_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "pl_msg/msg/wpnt_array.hpp"
#include "pl_msg/msg/obstacle_array.hpp"
#include "pl_msg/msg/ot_wpnt_array.hpp"
#include "pl_msg/msg/frenet_debug.hpp"
#include "pl_msg/msg/frenet_debug_array.hpp"
#include "spline_planner_dh_test/frenet_converter.hpp"
#include "spline_planner_dh_test/spline_interpolator.hpp"

#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <functional>

struct ObstacleFrenet {
    int id;
    double s_center, d_center;
    double vs, vd;      // velocity in s direction (along track)
    double yaw;     // orientation
    double size;    // obstacle size (radius or half-width)
    double priority = 0.0;  // 우선순위 (높을수록 우선)
};


class SplinePlanner : public rclcpp::Node {
public:
    SplinePlanner();

private:
    // Subscribers
    rclcpp::Subscription<pl_msg::msg::ObstacleArray>::SharedPtr obstacles_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ego_odom_sub_;
    rclcpp::Subscription<pl_msg::msg::WpntArray>::SharedPtr global_waypoints_sub_;
    
    // Publishers
    rclcpp::Publisher<pl_msg::msg::OTWpntArray>::SharedPtr local_trajectory_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr closest_obs_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr all_obstacles_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr reference_path_pub_;
    rclcpp::Publisher<pl_msg::msg::FrenetDebugArray>::SharedPtr frenet_debug_pub_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Data members
    pl_msg::msg::ObstacleArray::SharedPtr obstacles_;
    nav_msgs::msg::Odometry::SharedPtr ego_odom_;
    pl_msg::msg::WpntArray::SharedPtr global_waypoints_;

    std::unique_ptr<FrenetConverter> frenet_converter_;
    
    // Current state in Frenet coordinates
    double cur_s_, cur_d_, cur_vs_, cur_vd_;
    double ego_yaw_;
    
    // Parameters
    double lookahead_;
    double evasion_dist_;
    double obs_traj_thresh_;
    double spline_bound_mindist_;
    double gb_max_s_;
    int gb_max_idx_;
    double gb_vmax_;
    
    // Spline parameters
    double pre_apex_0_, pre_apex_1_, pre_apex_2_;
    double post_apex_0_, post_apex_1_, post_apex_2_;
    
    // State tracking
    std::string last_ot_side_;
    rclcpp::Time last_switch_time_;
    
    // Callbacks
    void obstacles_callback(const pl_msg::msg::ObstacleArray::SharedPtr msg);
    void ego_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void global_waypoints_callback(const pl_msg::msg::WpntArray::SharedPtr msg);

    // Main processing
    void timer_callback();
    void process_spline_planning();
    
    // Helper functions
    void initialize_converter();
    std::vector<ObstacleFrenet> convert_obstacles_to_frenet(const pl_msg::msg::ObstacleArray& obstacles);
    std::vector<ObstacleFrenet> filter_close_obstacles(const std::vector<ObstacleFrenet>& obstacles);
    std::pair<std::string, double> calculate_more_space(const ObstacleFrenet& obstacle);
    pl_msg::msg::OTWpntArray generate_evasion_trajectory(const ObstacleFrenet& closest_obstacle);
    pl_msg::msg::OTWpntArray generate_priority_based_evasion_trajectory(const std::vector<ObstacleFrenet>& obstacles);
    ObstacleFrenet select_primary_obstacle(const std::vector<ObstacleFrenet>& obstacles);
    pl_msg::msg::OTWpntArray adjust_trajectory_for_influence_zone(
        const pl_msg::msg::OTWpntArray& base_trajectory,
        const std::vector<ObstacleFrenet>& all_obstacles,
        const ObstacleFrenet& primary_obstacle);

    // Visualization
    visualization_msgs::msg::MarkerArray create_trajectory_markers(const pl_msg::msg::OTWpntArray& trajectory);
    visualization_msgs::msg::Marker create_obstacle_marker(const ObstacleFrenet& obstacle);
    visualization_msgs::msg::MarkerArray create_all_obstacles_markers(const std::vector<ObstacleFrenet>& obstacles);
    void publish_reference_path();
    void publish_all_obstacles(const std::vector<ObstacleFrenet>& obstacles);
    
    // Frenet debugging
    void publish_frenet_debug_info(const std::vector<ObstacleFrenet>& obstacles);
    
    // Utility functions
    bool wait_for_messages();
    double quaternion_to_yaw(double x, double y, double z, double w);
};

#endif // SPLINE_PLANNER_HPP
