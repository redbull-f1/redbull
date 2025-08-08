#include "spline_planner_dh/spline_interpolator.hpp"
#include <stdexcept>

SplineInterpolator::SplineInterpolator(const std::vector<double>& x, const std::vector<double>& y)
    : x_(x), y_(y) {
    if (x.size() != y.size() || x.size() < 2) {
        throw std::invalid_argument("Invalid input data for spline interpolation");
    }
    compute_coefficients();
}

void SplineInterpolator::compute_coefficients() {
    int n = x_.size() - 1;
    a_.resize(n);
    b_.resize(n);
    c_.resize(n + 1);
    d_.resize(n);
    
    std::vector<double> h(n);
    for (int i = 0; i < n; ++i) {
        h[i] = x_[i + 1] - x_[i];
        a_[i] = y_[i];
    }
    
    // Natural spline (second derivative = 0 at endpoints)
    std::vector<double> alpha(n);
    for (int i = 1; i < n; ++i) {
        alpha[i] = 3.0 * ((y_[i + 1] - y_[i]) / h[i] - (y_[i] - y_[i - 1]) / h[i - 1]);
    }
    
    std::vector<double> l(n + 1), mu(n), z(n + 1);
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;
    
    for (int i = 1; i < n; ++i) {
        l[i] = 2.0 * (x_[i + 1] - x_[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    
    l[n] = 1.0;
    z[n] = 0.0;
    c_[n] = 0.0;
    
    for (int j = n - 1; j >= 0; --j) {
        c_[j] = z[j] - mu[j] * c_[j + 1];
        b_[j] = (y_[j + 1] - y_[j]) / h[j] - h[j] * (c_[j + 1] + 2.0 * c_[j]) / 3.0;
        d_[j] = (c_[j + 1] - c_[j]) / (3.0 * h[j]);
    }
}

double SplineInterpolator::interpolate(double x) const {
    int segment = find_segment(x);
    if (segment < 0 || segment >= static_cast<int>(a_.size())) {
        // Linear extrapolation
        if (x < x_[0]) {
            double slope = (y_[1] - y_[0]) / (x_[1] - x_[0]);
            return y_[0] + slope * (x - x_[0]);
        } else {
            int last = x_.size() - 1;
            double slope = (y_[last] - y_[last - 1]) / (x_[last] - x_[last - 1]);
            return y_[last] + slope * (x - x_[last]);
        }
    }
    
    double dx = x - x_[segment];
    return a_[segment] + b_[segment] * dx + c_[segment] * dx * dx + d_[segment] * dx * dx * dx;
}

std::vector<double> SplineInterpolator::interpolate(const std::vector<double>& x_vals) const {
    std::vector<double> result;
    result.reserve(x_vals.size());
    
    for (double x : x_vals) {
        result.push_back(interpolate(x));
    }
    
    return result;
}

int SplineInterpolator::find_segment(double x) const {
    for (int i = 0; i < static_cast<int>(x_.size()) - 1; ++i) {
        if (x >= x_[i] && x <= x_[i + 1]) {
            return i;
        }
    }
    return -1; // Not found
}
