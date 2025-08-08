#include "spline_planner_rb/spline_planner.hpp"
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace spline_planner_rb
{

SplinePlanner::SplinePlanner() : Node("spline_planner")
{
  // Initialize parameters
  this->declare_parameter("lookahead_distance", 10.0);
  this->declare_parameter("evasion_distance", 0.65);
  this->declare_parameter("obs_traj_threshold", 0.3);
  this->declare_parameter("spline_resolution", 0.1);
  
  lookahead_distance_ = this->get_parameter("lookahead_distance").as_double();
  evasion_distance_ = this->get_parameter("evasion_distance").as_double();
  obs_traj_threshold_ = this->get_parameter("obs_traj_threshold").as_double();
  spline_resolution_ = this->get_parameter("spline_resolution").as_double();
  
  // Initialize spline control parameters (from Python code)
  spline_params_ = {-4.0, -3.0, -1.5, 0.0, 2.0, 3.0, 4.0};
  
  // Initialize components
  frenet_converter_ = std::make_unique<FrenetConverter>();
  spline_interpolator_ = std::make_unique<SplineInterpolator>();
  
  // Initialize flags
  global_waypoints_received_ = false;
  ego_state_received_ = false;
  last_evasion_side_ = "";
  
  // Create subscribers
  ego_state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/ego_racecar/odom", 10,
    std::bind(&SplinePlanner::egoStateCallback, this, std::placeholders::_1));
  
  global_waypoints_sub_ = this->create_subscription<lattice_planner_rb::msg::WpntArray>(
    "/global_waypoints", 10,
    std::bind(&SplinePlanner::globalWaypointsCallback, this, std::placeholders::_1));
  
  obstacles_sub_ = this->create_subscription<lattice_planner_rb::msg::ObstacleArray>(
    "/perception/obstacles", 10,
    std::bind(&SplinePlanner::obstaclesCallback, this, std::placeholders::_1));
  
  // Create publishers
  local_waypoints_pub_ = this->create_publisher<lattice_planner_rb::msg::OTWpntArray>(
    "/local_waypoints", 10);
  
  markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/spline_planner/markers", 10);
  
  closest_obs_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
    "/spline_planner/closest_obstacle", 10);
  
  // Create timer for main planning loop
  planning_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(50),  // 20Hz
    std::bind(&SplinePlanner::plannerLoop, this));
  
  RCLCPP_INFO(this->get_logger(), "Spline planner initialized");
}

void SplinePlanner::egoStateCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  // Extract position and orientation
  ego_state_.x = msg->pose.pose.position.x;
  ego_state_.y = msg->pose.pose.position.y;
  ego_state_.yaw = tf2::getYaw(msg->pose.pose.orientation);
  ego_state_.vs = std::sqrt(
    msg->twist.twist.linear.x * msg->twist.twist.linear.x +
    msg->twist.twist.linear.y * msg->twist.twist.linear.y);
  
  // Convert to Frenet coordinates if converter is ready
  if (frenet_converter_ && global_waypoints_received_) {
    auto frenet_pos = frenet_converter_->cartesianToFrenet(ego_state_.x, ego_state_.y);
    ego_state_.s = frenet_pos.first;
    ego_state_.d = frenet_pos.second;
  }
  
  ego_state_received_ = true;
}

void SplinePlanner::globalWaypointsCallback(const lattice_planner_rb::msg::WpntArray::SharedPtr msg)
{
  global_waypoints_ = msg->wpnts;
  
  if (!global_waypoints_.empty()) {
    // Extract coordinates for Frenet converter
    std::vector<double> x_coords, y_coords, yaw_coords;
    
    for (const auto& waypoint : global_waypoints_) {
      x_coords.push_back(waypoint.x_m);
      y_coords.push_back(waypoint.y_m);
      yaw_coords.push_back(waypoint.psi_rad);
    }
    
    // Initialize Frenet converter
    frenet_converter_->initialize(x_coords, y_coords, yaw_coords);
    
    // Store max values
    max_track_s_ = global_waypoints_.back().s_m;
    max_velocity_ = 0.0;
    for (const auto& wp : global_waypoints_) {
      max_velocity_ = std::max(max_velocity_, wp.vx_mps);
    }
    
    global_waypoints_received_ = true;
    RCLCPP_INFO(this->get_logger(), "Global waypoints received: %zu points", global_waypoints_.size());
  }
}

void SplinePlanner::obstaclesCallback(const lattice_planner_rb::msg::ObstacleArray::SharedPtr msg)
{
  current_obstacles_ = *msg;
}

void SplinePlanner::plannerLoop()
{
  if (!global_waypoints_received_ || !ego_state_received_) {
    return;
  }
  
  // Check if there are obstacles to avoid
  if (current_obstacles_.obstacles.empty()) {
    // No obstacles - publish global path as local waypoints
    publishGlobalPath();
    
    // Clear visualization markers
    visualization_msgs::msg::MarkerArray clear_markers;
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = this->now();
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_markers.markers.push_back(delete_marker);
    markers_pub_->publish(clear_markers);
    
    return;
  }
  
  // Generate spline path around obstacles
  auto [local_waypoints, markers] = generateSplinePath(current_obstacles_);
  
  // Publish results
  local_waypoints_pub_->publish(local_waypoints);
  markers_pub_->publish(markers);
}

std::pair<lattice_planner_rb::msg::OTWpntArray, visualization_msgs::msg::MarkerArray> 
SplinePlanner::generateSplinePath(const lattice_planner_rb::msg::ObstacleArray& obstacles)
{
  lattice_planner_rb::msg::OTWpntArray local_waypoints;
  visualization_msgs::msg::MarkerArray markers;
  
  // Filter obstacles
  auto filtered_obstacles = filterObstacles(obstacles);
  
  if (filtered_obstacles.empty()) {
    return {local_waypoints, markers};
  }
  
  // Find closest obstacle
  auto closest_obstacle = *std::min_element(
    filtered_obstacles.begin(), filtered_obstacles.end(),
    [this](const auto& a, const auto& b) {
      double dist_a = std::fmod(a.s_center - ego_state_.s + max_track_s_, max_track_s_);
      double dist_b = std::fmod(b.s_center - ego_state_.s + max_track_s_, max_track_s_);
      return dist_a < dist_b;
    });
  
  // Calculate obstacle center point
  double s_apex;
  if (closest_obstacle.s_end < closest_obstacle.s_start) {
    // Handle wraparound
    s_apex = std::fmod((closest_obstacle.s_end + max_track_s_ + closest_obstacle.s_start) / 2.0, max_track_s_);
  } else {
    s_apex = (closest_obstacle.s_end + closest_obstacle.s_start) / 2.0;
  }
  
  // Determine evasion direction and apex distance
  auto [evasion_side, d_apex] = determineEvasionDirection(closest_obstacle);
  
  // Generate evasion points
  double speed_scaling = std::clamp(1.0 + ego_state_.vs / max_velocity_, 1.0, 1.5);
  auto evasion_points = generateEvasionPoints(s_apex, d_apex, speed_scaling);
  
  // Prepare data for spline interpolation
  std::vector<double> s_coords, d_coords;
  for (const auto& point : evasion_points) {
    s_coords.push_back(point.s);
    d_coords.push_back(point.d);
  }
  
  // Initialize spline
  spline_interpolator_->initializeSpline(s_coords, d_coords);
  
  // Generate interpolated points
  std::vector<double> interpolated_s;
  double s_start = evasion_points.front().s;
  double s_end = evasion_points.back().s;
  
  for (double s = s_start; s < s_end; s += spline_resolution_) {
    interpolated_s.push_back(s);
  }
  
  auto interpolated_d = spline_interpolator_->evaluateBatch(interpolated_s);
  
  // Clip d values to prevent excessive deviation
  for (auto& d : interpolated_d) {
    if (d_apex < 0) {
      d = std::clamp(d, d_apex, 0.0);
    } else {
      d = std::clamp(d, 0.0, d_apex);
    }
  }
  
  // Handle s coordinate wraparound
  for (auto& s : interpolated_s) {
    s = std::fmod(s, max_track_s_);
    if (s < 0) s += max_track_s_;
  }
  
  // Convert to Cartesian coordinates
  auto cartesian_points = frenet_converter_->frenetToCartesianBatch(interpolated_s, interpolated_d);
  
  // Create waypoint messages
  local_waypoints.header.frame_id = "map";
  local_waypoints.header.stamp = this->now();
  local_waypoints.ot_side = evasion_side;
  local_waypoints.side_switch = (last_evasion_side_ != evasion_side);
  last_evasion_side_ = evasion_side;
  
  for (size_t i = 0; i < cartesian_points.size(); ++i) {
    // Find closest global waypoint for velocity reference
    double closest_dist = std::numeric_limits<double>::max();
    double reference_velocity = 10.0;  // default
    
    for (const auto& gw : global_waypoints_) {
      double dist = std::abs(std::fmod(gw.s_m - interpolated_s[i] + max_track_s_, max_track_s_));
      if (dist < closest_dist) {
        closest_dist = dist;
        reference_velocity = gw.vx_mps;
      }
    }
    
    // Create waypoint
    auto waypoint = createWaypoint(
      cartesian_points[i].first, cartesian_points[i].second,
      interpolated_s[i], interpolated_d[i], reference_velocity, i);
    
    local_waypoints.wpnts.push_back(waypoint);
    
    // Create visualization marker
    auto marker = createVisualizationMarker(
      cartesian_points[i].first, cartesian_points[i].second, reference_velocity, i);
    
    markers.markers.push_back(marker);
  }
  
  return {local_waypoints, markers};
}

std::vector<lattice_planner_rb::msg::Obstacle> SplinePlanner::filterObstacles(
  const lattice_planner_rb::msg::ObstacleArray& obstacles)
{
  std::vector<lattice_planner_rb::msg::Obstacle> filtered;
  
  for (const auto& obstacle : obstacles.obstacles) {
    // Filter by trajectory threshold
    if (std::abs(obstacle.d_center) > obs_traj_threshold_) {
      continue;
    }
    
    // Filter by lookahead distance
    double dist_in_front = std::fmod(obstacle.s_center - ego_state_.s + max_track_s_, max_track_s_);
    if (dist_in_front < lookahead_distance_) {
      filtered.push_back(obstacle);
    }
  }
  
  return filtered;
}

std::pair<std::string, double> SplinePlanner::determineEvasionDirection(
  const lattice_planner_rb::msg::Obstacle& obstacle)
{
  // Find closest global waypoint to obstacle
  double closest_dist = std::numeric_limits<double>::max();
  const lattice_planner_rb::msg::Wpnt* closest_wp = nullptr;
  
  for (const auto& wp : global_waypoints_) {
    double dist = std::abs(wp.s_m - obstacle.s_center);
    if (dist < closest_dist) {
      closest_dist = dist;
      closest_wp = &wp;
    }
  }
  
  if (!closest_wp) {
    return {"left", -evasion_distance_};
  }
  
  // Calculate available space on each side
  double left_gap = std::abs(closest_wp->d_left - obstacle.d_left);
  double right_gap = std::abs(closest_wp->d_right - obstacle.d_right);
  double min_space = evasion_distance_ + 0.2;  // safety margin
  
  if (right_gap > min_space && left_gap < min_space) {
    // Right side has more space
    double d_apex_right = obstacle.d_right - evasion_distance_;
    if (d_apex_right > 0) d_apex_right = 0;
    return {"right", d_apex_right};
  } else if (left_gap > min_space && right_gap < min_space) {
    // Left side has more space
    double d_apex_left = obstacle.d_left + evasion_distance_;
    if (d_apex_left < 0) d_apex_left = 0;
    return {"left", d_apex_left};
  } else {
    // Both sides have space, choose closer to raceline
    double candidate_d_apex_left = obstacle.d_left + evasion_distance_;
    double candidate_d_apex_right = obstacle.d_right - evasion_distance_;
    
    if (std::abs(candidate_d_apex_left) <= std::abs(candidate_d_apex_right)) {
      if (candidate_d_apex_left < 0) candidate_d_apex_left = 0;
      return {"left", candidate_d_apex_left};
    } else {
      if (candidate_d_apex_right > 0) candidate_d_apex_right = 0;
      return {"right", candidate_d_apex_right};
    }
  }
}

std::vector<EvasionPoint> SplinePlanner::generateEvasionPoints(
  double s_apex, double d_apex, double speed_scaling)
{
  std::vector<EvasionPoint> points;
  
  for (double spline_param : spline_params_) {
    EvasionPoint point;
    point.s = s_apex + spline_param * speed_scaling;
    point.d = (spline_param == 0.0) ? d_apex : 0.0;
    points.push_back(point);
  }
  
  return points;
}

lattice_planner_rb::msg::Wpnt SplinePlanner::createWaypoint(
  double x, double y, double s, double d, double v, int id)
{
  lattice_planner_rb::msg::Wpnt waypoint;
  waypoint.id = id;
  waypoint.x_m = x;
  waypoint.y_m = y;
  waypoint.s_m = s;
  waypoint.d_m = d;
  waypoint.vx_mps = v;
  return waypoint;
}

visualization_msgs::msg::Marker SplinePlanner::createVisualizationMarker(
  double x, double y, double v, int id)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = this->now();
  marker.type = visualization_msgs::msg::Marker::CYLINDER;
  marker.id = id;
  
  marker.pose.position.x = x;
  marker.pose.position.y = y;
  marker.pose.position.z = v / (max_velocity_ / 2.0);
  marker.pose.orientation.w = 1.0;
  
  marker.scale.x = 0.1;
  marker.scale.y = 0.1;
  marker.scale.z = v / max_velocity_;
  
  marker.color.a = 1.0;
  marker.color.r = 0.75;
  marker.color.g = 0.0;
  marker.color.b = 0.75;
  
  return marker;
}

void SplinePlanner::publishGlobalPath()
{
  lattice_planner_rb::msg::OTWpntArray local_waypoints;
  local_waypoints.header.frame_id = "map";
  local_waypoints.header.stamp = this->now();
  local_waypoints.ot_side = "";
  local_waypoints.side_switch = false;
  
  // Find waypoints near ego position
  double start_s = ego_state_.s;
  double end_s = start_s + lookahead_distance_;
  
  for (const auto& gw : global_waypoints_) {
    double s_dist = std::fmod(gw.s_m - start_s + max_track_s_, max_track_s_);
    if (s_dist <= lookahead_distance_) {
      lattice_planner_rb::msg::Wpnt wp;
      wp.id = gw.id;
      wp.x_m = gw.x_m;
      wp.y_m = gw.y_m;
      wp.s_m = gw.s_m;
      wp.d_m = gw.d_m;
      wp.vx_mps = gw.vx_mps;
      local_waypoints.wpnts.push_back(wp);
    }
  }
  
  local_waypoints_pub_->publish(local_waypoints);
}

}  // namespace spline_planner_rb

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<spline_planner_rb::SplinePlanner>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
