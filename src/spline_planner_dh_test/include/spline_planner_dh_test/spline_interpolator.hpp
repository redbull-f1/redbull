#ifndef SPLINE_INTERPOLATOR_HPP
#define SPLINE_INTERPOLATOR_HPP

#include <vector>
#include <algorithm>

class SplineInterpolator {
public:
    SplineInterpolator(const std::vector<double>& x, const std::vector<double>& y);
    double interpolate(double x) const;
    std::vector<double> interpolate(const std::vector<double>& x_vals) const;
private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> a_, b_, c_, d_;
    void compute_coefficients();
    int find_segment(double x) const;
};

#endif // SPLINE_INTERPOLATOR_HPP
