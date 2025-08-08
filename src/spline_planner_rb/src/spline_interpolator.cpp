#include "spline_planner_rb/spline_interpolator.hpp"

namespace spline_planner_rb
{

void SplineInterpolator::initializeSpline(const std::vector<double>& x, const std::vector<double>& y)
{
  if (x.size() != y.size() || x.size() < 2) {
    initialized_ = false;
    return;
  }
  
  x_data_ = x;
  y_data_ = y;
  
  calculateCoefficients();
  initialized_ = true;
}

double SplineInterpolator::evaluate(double x)
{
  if (!initialized_) {
    return 0.0;
  }
  
  // Find the segment
  auto it = std::upper_bound(x_data_.begin(), x_data_.end(), x);
  int idx = std::distance(x_data_.begin(), it) - 1;
  idx = std::max(0, std::min(idx, static_cast<int>(x_data_.size()) - 2));
  
  // Evaluate cubic polynomial
  double dx = x - x_data_[idx];
  return a_[idx] + b_[idx] * dx + c_[idx] * dx * dx + d_[idx] * dx * dx * dx;
}

std::vector<double> SplineInterpolator::evaluateBatch(const std::vector<double>& x_values)
{
  std::vector<double> result;
  result.reserve(x_values.size());
  
  for (double x : x_values) {
    result.push_back(evaluate(x));
  }
  
  return result;
}

void SplineInterpolator::calculateCoefficients()
{
  int n = x_data_.size();
  
  // Initialize coefficient vectors
  a_.resize(n);
  b_.resize(n);
  c_.resize(n);
  d_.resize(n);
  
  // Simple cubic spline implementation (natural spline)
  std::vector<double> h(n-1);
  for (int i = 0; i < n-1; ++i) {
    h[i] = x_data_[i+1] - x_data_[i];
  }
  
  // Set up tridiagonal system for second derivatives
  std::vector<double> alpha(n-1);
  for (int i = 1; i < n-1; ++i) {
    alpha[i] = 3.0 * ((y_data_[i+1] - y_data_[i]) / h[i] - (y_data_[i] - y_data_[i-1]) / h[i-1]);
  }
  
  std::vector<double> l(n), mu(n-1), z(n);
  l[0] = 1.0;
  mu[0] = 0.0;
  z[0] = 0.0;
  
  for (int i = 1; i < n-1; ++i) {
    l[i] = 2.0 * (x_data_[i+1] - x_data_[i-1]) - h[i-1] * mu[i-1];
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
  }
  
  l[n-1] = 1.0;
  z[n-1] = 0.0;
  c_[n-1] = 0.0;
  
  for (int j = n-2; j >= 0; --j) {
    c_[j] = z[j] - mu[j] * c_[j+1];
    b_[j] = (y_data_[j+1] - y_data_[j]) / h[j] - h[j] * (c_[j+1] + 2.0 * c_[j]) / 3.0;
    d_[j] = (c_[j+1] - c_[j]) / (3.0 * h[j]);
    a_[j] = y_data_[j];
  }
}

double SplineInterpolator::linearInterpolate(
  double x, 
  const std::vector<double>& x_data, 
  const std::vector<double>& y_data)
{
  if (x_data.size() != y_data.size() || x_data.size() < 2) {
    return 0.0;
  }
  
  // Find the segment
  auto it = std::upper_bound(x_data.begin(), x_data.end(), x);
  int idx = std::distance(x_data.begin(), it) - 1;
  idx = std::max(0, std::min(idx, static_cast<int>(x_data.size()) - 2));
  
  // Linear interpolation
  double ratio = (x - x_data[idx]) / (x_data[idx+1] - x_data[idx]);
  return y_data[idx] + ratio * (y_data[idx+1] - y_data[idx]);
}

}  // namespace spline_planner_rb
