#ifndef SPLINE_PLANNER_RB__SPLINE_INTERPOLATOR_HPP_
#define SPLINE_PLANNER_RB__SPLINE_INTERPOLATOR_HPP_

#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

namespace spline_planner_rb
{

class SplineInterpolator
{
public:
  SplineInterpolator() = default;
  ~SplineInterpolator() = default;
  
  // Cubic spline interpolation
  void initializeSpline(const std::vector<double>& x, const std::vector<double>& y);
  
  // Evaluate spline at given x value
  double evaluate(double x);
  
  // Evaluate spline at multiple x values
  std::vector<double> evaluateBatch(const std::vector<double>& x_values);
  
  // Simple linear interpolation for fallback
  static double linearInterpolate(
    double x, 
    const std::vector<double>& x_data, 
    const std::vector<double>& y_data);

private:
  std::vector<double> x_data_;
  std::vector<double> y_data_;
  std::vector<double> a_, b_, c_, d_;  // spline coefficients
  
  void calculateCoefficients();
  bool initialized_;
};

}  // namespace spline_planner_rb

#endif  // SPLINE_PLANNER_RB__SPLINE_INTERPOLATOR_HPP_
