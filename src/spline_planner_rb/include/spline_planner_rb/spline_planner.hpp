#ifndef SPLINE_PLANNER_RB__SPLINE_PLANNER_HPP_
#define SPLINE_PLANNER_RB__SPLINE_PLANNER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <lattice_planner_rb/msg/wpnt.hpp>
#include <lattice_planner_rb/msg/wpnt_array.hpp>
#include <lattice_planner_rb/msg/obstacle.hpp>
#include <lattice_planner_rb/msg/obstacle_array.hpp>
#include <lattice_planner_rb/msg/otwpnt_array.hpp>

#include "spline_planner_rb/frenet_converter.hpp"
#include "spline_planner_rb/spline_interpolator.hpp"

#include <vector>
#include <memory>
#include <string>
#include <cmath>

namespace spline_planner_rb
{

struct EvasionPoint
{
  double s;  // longitudinal coordinate
  double d;  // lateral coordinate
};

struct EgoState
{
  double s;     // current s position
  double d;     // current d position  
  double vs;    // current longitudinal velocity
  double x;     // current x position
  double y;     // current y position
  double yaw;   // current yaw angle
};

class SplinePlanner : public rclcpp::Node
{
public:
  SplinePlanner();
  ~SplinePlanner() = default;

private:
  // Callback functions
  void egoStateCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void globalWaypointsCallback(const lattice_planner_rb::msg::WpntArray::SharedPtr msg);
  void obstaclesCallback(const lattice_planner_rb::msg::ObstacleArray::SharedPtr msg);
  
  // Main planning function
  void plannerLoop();
  
  // Core spline planning functions
  std::pair<lattice_planner_rb::msg::OTWpntArray, visualization_msgs::msg::MarkerArray> 
  generateSplinePath(const lattice_planner_rb::msg::ObstacleArray& obstacles);
  
  // Obstacle filtering and processing
  std::vector<lattice_planner_rb::msg::Obstacle> filterObstacles(
    const lattice_planner_rb::msg::ObstacleArray& obstacles);
  
  // Evasion planning
  std::pair<std::string, double> determineEvasionDirection(
    const lattice_planner_rb::msg::Obstacle& obstacle);
  
  std::vector<EvasionPoint> generateEvasionPoints(
    double s_apex, double d_apex, double speed_scaling);
  
  // Utility functions
  lattice_planner_rb::msg::Wpnt createWaypoint(
    double x, double y, double s, double d, double v, int id);
  
  visualization_msgs::msg::Marker createVisualizationMarker(
    double x, double y, double v, int id);
  
  void publishGlobalPath();
  
  // Member variables
  rclcpp::TimerBase::SharedPtr planning_timer_;
  
  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ego_state_sub_;
  rclcpp::Subscription<lattice_planner_rb::msg::WpntArray>::SharedPtr global_waypoints_sub_;
  rclcpp::Subscription<lattice_planner_rb::msg::ObstacleArray>::SharedPtr obstacles_sub_;
  
  // Publishers
  rclcpp::Publisher<lattice_planner_rb::msg::OTWpntArray>::SharedPtr local_waypoints_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr closest_obs_pub_;
  
  // Core components
  std::unique_ptr<FrenetConverter> frenet_converter_;
  std::unique_ptr<SplineInterpolator> spline_interpolator_;
  
  // State variables
  EgoState ego_state_;
  std::vector<lattice_planner_rb::msg::Wpnt> global_waypoints_;
  lattice_planner_rb::msg::ObstacleArray current_obstacles_;
  
  // Planning parameters
  double lookahead_distance_;
  double evasion_distance_;
  double obs_traj_threshold_;
  double spline_resolution_;
  double max_track_s_;
  double max_velocity_;
  
  // Spline control parameters
  std::vector<double> spline_params_;  // [pre_apex_0, pre_apex_1, pre_apex_2, 0, post_apex_0, post_apex_1, post_apex_2]
  
  // Flags
  bool global_waypoints_received_;
  bool ego_state_received_;
  std::string last_evasion_side_;
};

}  // namespace spline_planner_rb

#endif  // SPLINE_PLANNER_RB__SPLINE_PLANNER_HPP_
