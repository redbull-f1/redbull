#ifndef SPLINE_PLANNER_HPP
#define SPLINE_PLANNER_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "spline_planner_dh/msg/wpnt_array.hpp"
#include "spline_planner_dh/msg/obstacle_array.hpp"
#include "spline_planner_dh/msg/ot_wpnt_array.hpp"

#include "spline_planner_dh/frenet_converter.hpp"
#include "spline_planner_dh/spline_interpolator.hpp"

#include <memory>
#include <vector>
#include <string>

struct ObstacleFrenet {
    int id;
    double s_center, d_center;
    double s_left, d_left;
    double s_right, d_right;
    double s_start, d_start;
    double s_end, d_end;
};

class SplinePlanner : public rclcpp::Node {
public:
    SplinePlanner();

private:
    // Subscribers
    rclcpp::Subscription<spline_planner_dh::msg::ObstacleArray>::SharedPtr obstacles_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ego_odom_sub_;
    rclcpp::Subscription<spline_planner_dh::msg::WpntArray>::SharedPtr global_waypoints_sub_;
    
    // Publishers
    rclcpp::Publisher<spline_planner_dh::msg::OTWpntArray>::SharedPtr local_trajectory_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr closest_obs_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr reference_path_pub_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Data members
    spline_planner_dh::msg::ObstacleArray::SharedPtr obstacles_;
    nav_msgs::msg::Odometry::SharedPtr ego_odom_;
    spline_planner_dh::msg::WpntArray::SharedPtr global_waypoints_;
    
    std::unique_ptr<FrenetConverter> frenet_converter_;
    
    // Current state in Frenet coordinates
    double cur_s_, cur_d_, cur_vs_;
    
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
    void obstacles_callback(const spline_planner_dh::msg::ObstacleArray::SharedPtr msg);
    void ego_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void global_waypoints_callback(const spline_planner_dh::msg::WpntArray::SharedPtr msg);
    
    // Main processing
    void timer_callback();
    void process_spline_planning();
    
    // Helper functions
    void initialize_converter();
    std::vector<ObstacleFrenet> convert_obstacles_to_frenet(const spline_planner_dh::msg::ObstacleArray& obstacles);
    std::vector<ObstacleFrenet> filter_close_obstacles(const std::vector<ObstacleFrenet>& obstacles);
    std::pair<std::string, double> calculate_more_space(const ObstacleFrenet& obstacle);
    
    spline_planner_dh::msg::OTWpntArray generate_evasion_trajectory(const ObstacleFrenet& closest_obstacle);
    
    // Visualization
    visualization_msgs::msg::MarkerArray create_trajectory_markers(const spline_planner_dh::msg::OTWpntArray& trajectory);
    visualization_msgs::msg::Marker create_obstacle_marker(const ObstacleFrenet& obstacle);
    void publish_reference_path();
    
    // Utility functions
    bool wait_for_messages();
};

#endif // SPLINE_PLANNER_HPP
