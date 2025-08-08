#include "spline_planner_rb/frenet_converter.hpp"

namespace spline_planner_rb
{

void FrenetConverter::initialize(
  const std::vector<double>& x_coords,
  const std::vector<double>& y_coords,
  const std::vector<double>& yaw_angles)
{
  x_waypoints_ = x_coords;
  y_waypoints_ = y_coords;
  yaw_waypoints_ = yaw_angles;
  
  // Calculate cumulative arc length
  s_waypoints_.clear();
  s_waypoints_.push_back(0.0);
  
  for (size_t i = 1; i < x_coords.size(); ++i) {
    double dx = x_coords[i] - x_coords[i-1];
    double dy = y_coords[i] - y_coords[i-1];
    double ds = std::sqrt(dx*dx + dy*dy);
    s_waypoints_.push_back(s_waypoints_.back() + ds);
  }
  
  initialized_ = true;
}

std::pair<double, double> FrenetConverter::cartesianToFrenet(double x, double y)
{
  if (!initialized_) {
    return {0.0, 0.0};
  }
  
  int closest_idx = findClosestWaypoint(x, y);
  
  // Get closest waypoint data
  double x_wp = x_waypoints_[closest_idx];
  double y_wp = y_waypoints_[closest_idx];
  double yaw_wp = yaw_waypoints_[closest_idx];
  double s_wp = s_waypoints_[closest_idx];
  
  // Vector from waypoint to query point
  double dx = x - x_wp;
  double dy = y - y_wp;
  
  // Calculate s coordinate (projection along track)
  double s_offset = dx * std::cos(yaw_wp) + dy * std::sin(yaw_wp);
  double s = s_wp + s_offset;
  
  // Calculate d coordinate (lateral offset)
  double d = -dx * std::sin(yaw_wp) + dy * std::cos(yaw_wp);
  
  return {s, d};
}

std::pair<double, double> FrenetConverter::frenetToCartesian(double s, double d)
{
  if (!initialized_) {
    return {0.0, 0.0};
  }
  
  // Handle wraparound
  while (s < 0) s += s_waypoints_.back();
  while (s >= s_waypoints_.back()) s -= s_waypoints_.back();
  
  // Find segment containing s
  auto it = std::upper_bound(s_waypoints_.begin(), s_waypoints_.end(), s);
  int idx = std::distance(s_waypoints_.begin(), it) - 1;
  idx = std::max(0, std::min(idx, static_cast<int>(s_waypoints_.size()) - 2));
  
  // Interpolate position and yaw
  double ratio = 0.0;
  if (s_waypoints_[idx+1] > s_waypoints_[idx]) {
    ratio = (s - s_waypoints_[idx]) / (s_waypoints_[idx+1] - s_waypoints_[idx]);
  }
  
  double x_ref = x_waypoints_[idx] + ratio * (x_waypoints_[idx+1] - x_waypoints_[idx]);
  double y_ref = y_waypoints_[idx] + ratio * (y_waypoints_[idx+1] - y_waypoints_[idx]);
  double yaw_ref = yaw_waypoints_[idx] + ratio * (yaw_waypoints_[idx+1] - yaw_waypoints_[idx]);
  
  // Apply lateral offset
  double x = x_ref - d * std::sin(yaw_ref);
  double y = y_ref + d * std::cos(yaw_ref);
  
  return {x, y};
}

std::vector<std::pair<double, double>> FrenetConverter::frenetToCartesianBatch(
  const std::vector<double>& s_coords,
  const std::vector<double>& d_coords)
{
  std::vector<std::pair<double, double>> result;
  result.reserve(s_coords.size());
  
  for (size_t i = 0; i < s_coords.size(); ++i) {
    result.push_back(frenetToCartesian(s_coords[i], d_coords[i]));
  }
  
  return result;
}

int FrenetConverter::findClosestWaypoint(double x, double y)
{
  double min_dist = std::numeric_limits<double>::max();
  int closest_idx = 0;
  
  for (size_t i = 0; i < x_waypoints_.size(); ++i) {
    double dx = x - x_waypoints_[i];
    double dy = y - y_waypoints_[i];
    double dist = dx*dx + dy*dy;
    
    if (dist < min_dist) {
      min_dist = dist;
      closest_idx = i;
    }
  }
  
  return closest_idx;
}

}  // namespace spline_planner_rb
