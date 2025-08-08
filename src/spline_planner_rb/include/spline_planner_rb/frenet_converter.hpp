#ifndef SPLINE_PLANNER_RB__FRENET_CONVERTER_HPP_
#define SPLINE_PLANNER_RB__FRENET_CONVERTER_HPP_

#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

namespace spline_planner_rb
{

class FrenetConverter
{
public:
  FrenetConverter() = default;
  ~FrenetConverter() = default;
  
  // Initialize with global waypoints
  void initialize(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    const std::vector<double>& yaw_angles);
  
  // Convert Cartesian to Frenet coordinates
  std::pair<double, double> cartesianToFrenet(double x, double y);
  
  // Convert Frenet to Cartesian coordinates
  std::pair<double, double> frenetToCartesian(double s, double d);
  
  // Batch conversion for multiple points
  std::vector<std::pair<double, double>> frenetToCartesianBatch(
    const std::vector<double>& s_coords,
    const std::vector<double>& d_coords);

private:
  std::vector<double> x_waypoints_;
  std::vector<double> y_waypoints_;
  std::vector<double> yaw_waypoints_;
  std::vector<double> s_waypoints_;  // cumulative arc length
  
  // Find closest waypoint index
  int findClosestWaypoint(double x, double y);
  
  // Interpolation helpers
  double interpolateYaw(double s);
  std::pair<double, double> interpolatePosition(double s);
  
  bool initialized_;
};

}  // namespace spline_planner_rb

#endif  // SPLINE_PLANNER_RB__FRENET_CONVERTER_HPP_
