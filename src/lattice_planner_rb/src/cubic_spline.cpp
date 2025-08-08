#include "lattice_planner_rb/cubic_spline.hpp"
#include <cmath>
#include <iostream>

namespace lattice_planner {

//==============================================================================
// CubicSpline Implementation
//==============================================================================

CubicSpline::CubicSpline(const std::vector<double>& x, const std::vector<double>& y) 
    : initialized_(false) {
    setPoints(x, y);
}

void CubicSpline::setPoints(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        throw std::invalid_argument("Invalid input: x and y must have same size and at least 2 points");
    }
    
    // Check if x is in ascending order
    for (size_t i = 1; i < x.size(); i++) {
        if (x[i] <= x[i-1]) {
            throw std::invalid_argument("x coordinates must be in strictly ascending order");
        }
    }
    
    x_ = x;
    y_ = y;
    
    computeCoefficients();
    initialized_ = true;
}

void CubicSpline::computeCoefficients() {
    int n = x_.size() - 1;  // Number of intervals
    
    // Clear previous coefficients
    a_.clear(); b_.clear(); c_.clear(); d_.clear(); h_.clear();
    
    // Resize vectors
    a_.resize(n + 1);
    b_.resize(n);
    c_.resize(n + 1);
    d_.resize(n);
    h_.resize(n);
    
    // Calculate step sizes
    for (int i = 0; i < n; i++) {
        h_[i] = x_[i + 1] - x_[i];
        if (h_[i] <= 0) {
            throw std::runtime_error("Invalid step size in cubic spline");
        }
    }
    
    // Set a coefficients (y values)
    for (int i = 0; i <= n; i++) {
        a_[i] = y_[i];
    }
    
    // Calculate second derivatives using natural spline conditions
    if (n == 1) {
        // Linear case
        c_[0] = c_[1] = 0.0;
    } else {
        // Set up tridiagonal system for second derivatives
        std::vector<double> alpha(n - 1);
        
        for (int i = 1; i < n; i++) {
            alpha[i - 1] = 3.0 / h_[i] * (a_[i + 1] - a_[i]) - 3.0 / h_[i - 1] * (a_[i] - a_[i - 1]);
        }
        
        // Solve tridiagonal system
        std::vector<double> l(n + 1), mu(n), z(n + 1);
        
        // Natural spline boundary conditions
        l[0] = 1.0;
        mu[0] = z[0] = 0.0;
        
        for (int i = 1; i < n; i++) {
            l[i] = 2.0 * (x_[i + 1] - x_[i - 1]) - h_[i - 1] * mu[i - 1];
            mu[i] = h_[i] / l[i];
            z[i] = (alpha[i - 1] - h_[i - 1] * z[i - 1]) / l[i];
        }
        
        // Natural spline boundary conditions
        l[n] = 1.0;
        z[n] = c_[n] = 0.0;
        
        // Back substitution
        for (int j = n - 1; j >= 0; j--) {
            c_[j] = z[j] - mu[j] * c_[j + 1];
            b_[j] = (a_[j + 1] - a_[j]) / h_[j] - h_[j] * (c_[j + 1] + 2.0 * c_[j]) / 3.0;
            d_[j] = (c_[j + 1] - c_[j]) / (3.0 * h_[j]);
        }
    }
}

double CubicSpline::interpolate(double x) const {
    if (!initialized_) {
        throw std::runtime_error("Spline not initialized");
    }
    
    // Handle boundary cases
    if (x <= x_.front()) return y_.front();
    if (x >= x_.back()) return y_.back();
    
    size_t i = findIndex(x);
    double dx = x - x_[i];
    
    // Evaluate cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
    return a_[i] + b_[i] * dx + c_[i] * dx * dx + d_[i] * dx * dx * dx;
}

double CubicSpline::derivative(double x) const {
    if (!initialized_) {
        throw std::runtime_error("Spline not initialized");
    }
    
    // Handle boundary cases
    if (x <= x_.front()) {
        double dx = x_[1] - x_[0];
        return b_[0];
    }
    if (x >= x_.back()) {
        size_t last_idx = b_.size() - 1;
        return b_[last_idx] + 2.0 * c_[last_idx] * h_[last_idx] + 3.0 * d_[last_idx] * h_[last_idx] * h_[last_idx];
    }
    
    size_t i = findIndex(x);
    double dx = x - x_[i];
    
    // Derivative: b + 2*c*dx + 3*d*dx^2
    return b_[i] + 2.0 * c_[i] * dx + 3.0 * d_[i] * dx * dx;
}

double CubicSpline::secondDerivative(double x) const {
    if (!initialized_) {
        throw std::runtime_error("Spline not initialized");
    }
    
    // Handle boundary cases
    if (x <= x_.front()) return 2.0 * c_[0];
    if (x >= x_.back()) return 2.0 * c_.back();
    
    size_t i = findIndex(x);
    double dx = x - x_[i];
    
    // Second derivative: 2*c + 6*d*dx
    return 2.0 * c_[i] + 6.0 * d_[i] * dx;
}

std::pair<double, double> CubicSpline::getRange() const {
    if (!initialized_ || x_.empty()) {
        return {0.0, 0.0};
    }
    return {x_.front(), x_.back()};
}

size_t CubicSpline::findIndex(double x) const {
    // Binary search for the correct interval
    auto it = std::lower_bound(x_.begin(), x_.end(), x);
    size_t index = std::distance(x_.begin(), it);
    
    // Adjust for interpolation
    if (index > 0) index--;
    if (index >= x_.size() - 1) index = x_.size() - 2;
    
    return index;
}

//==============================================================================
// CubicSplinePath Implementation
//==============================================================================

CubicSplinePath::CubicSplinePath(const std::vector<double>& x, const std::vector<double>& y) 
    : total_length_(0.0), initialized_(false) {
    setWaypoints(x, y);
}

void CubicSplinePath::setWaypoints(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        throw std::invalid_argument("Invalid waypoints: x and y must have same size and at least 2 points");
    }
    
    // Calculate cumulative arc lengths
    s_ = calculateArcLengths(x, y);
    total_length_ = s_.back();
    
    // Create splines for x and y coordinates parameterized by arc length
    spline_x_.setPoints(s_, x);
    spline_y_.setPoints(s_, y);
    
    initialized_ = true;
}

std::vector<double> CubicSplinePath::calculateArcLengths(const std::vector<double>& x, 
                                                       const std::vector<double>& y) const {
    std::vector<double> arc_lengths(x.size());
    arc_lengths[0] = 0.0;
    
    for (size_t i = 1; i < x.size(); i++) {
        double dx = x[i] - x[i-1];
        double dy = y[i] - y[i-1];
        double segment_length = std::sqrt(dx * dx + dy * dy);
        arc_lengths[i] = arc_lengths[i-1] + segment_length;
    }
    
    return arc_lengths;
}

std::pair<double, double> CubicSplinePath::interpolatePosition(double s) const {
    if (!initialized_) {
        throw std::runtime_error("Spline path not initialized");
    }
    
    // Clamp s to valid range
    s = std::max(0.0, std::min(s, total_length_));
    
    double x = spline_x_.interpolate(s);
    double y = spline_y_.interpolate(s);
    
    return {x, y};
}

double CubicSplinePath::interpolateYaw(double s) const {
    if (!initialized_) {
        throw std::runtime_error("Spline path not initialized");
    }
    
    // Clamp s to valid range
    s = std::max(0.0, std::min(s, total_length_));
    
    double dx_ds = spline_x_.derivative(s);
    double dy_ds = spline_y_.derivative(s);
    
    return std::atan2(dy_ds, dx_ds);
}

double CubicSplinePath::interpolateCurvature(double s) const {
    if (!initialized_) {
        throw std::runtime_error("Spline path not initialized");
    }
    
    // Clamp s to valid range
    s = std::max(0.0, std::min(s, total_length_));
    
    double dx_ds = spline_x_.derivative(s);
    double dy_ds = spline_y_.derivative(s);
    double d2x_ds2 = spline_x_.secondDerivative(s);
    double d2y_ds2 = spline_y_.secondDerivative(s);
    
    // Curvature formula: κ = (x'*y'' - y'*x'') / (x'^2 + y'^2)^(3/2)
    double numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2;
    double denominator_sq = dx_ds * dx_ds + dy_ds * dy_ds;
    
    if (denominator_sq < 1e-8) {
        return 0.0;  // Avoid division by zero
    }
    
    return numerator / std::pow(denominator_sq, 1.5);
}

std::vector<std::array<double, 5>> CubicSplinePath::generateUniformPoints(double resolution) const {
    if (!initialized_) {
        throw std::runtime_error("Spline path not initialized");
    }
    
    std::vector<std::array<double, 5>> points;
    
    for (double s = 0.0; s <= total_length_; s += resolution) {
        auto pos = interpolatePosition(s);
        double yaw = interpolateYaw(s);
        double curvature = interpolateCurvature(s);
        
        points.push_back({s, pos.first, pos.second, yaw, curvature});
    }
    
    // Make sure we include the final point
    if (points.empty() || points.back()[0] < total_length_) {
        auto pos = interpolatePosition(total_length_);
        double yaw = interpolateYaw(total_length_);
        double curvature = interpolateCurvature(total_length_);
        
        points.push_back({total_length_, pos.first, pos.second, yaw, curvature});
    }
    
    return points;
}

} // namespace lattice_planner
