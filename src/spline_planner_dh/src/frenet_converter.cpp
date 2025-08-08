#include "spline_planner_dh/frenet_converter.hpp"

FrenetConverter::FrenetConverter(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& psi)
    : x_(x), y_(y), psi_(psi) {
    compute_s_values();
}

void FrenetConverter::compute_s_values() {
    s_.resize(x_.size());
    s_[0] = 0.0;
    
    for (size_t i = 1; i < x_.size(); ++i) {
        double dx = x_[i] - x_[i-1];
        double dy = y_[i] - y_[i-1];
        s_[i] = s_[i-1] + std::sqrt(dx*dx + dy*dy);
    }
    
    max_s_ = s_.back();
}

std::pair<double, double> FrenetConverter::cartesian_to_frenet(double x, double y) const {
    int closest_idx = find_closest_point(x, y);
    
    // Get the closest point on the reference line
    double ref_x = x_[closest_idx];
    double ref_y = y_[closest_idx];
    double ref_psi = psi_[closest_idx];
    
    // Vector from reference point to target point
    double dx = x - ref_x;
    double dy = y - ref_y;
    
    // Calculate d (lateral offset)
    double d = dx * (-std::sin(ref_psi)) + dy * std::cos(ref_psi);
    
    // Calculate s (longitudinal distance along path)
    double s = s_[closest_idx];
    
    // Refine s calculation by projecting onto the path direction
    double ds = dx * std::cos(ref_psi) + dy * std::sin(ref_psi);
    s += ds;
    
    // Handle wraparound
    if (s < 0) s += max_s_;
    if (s > max_s_) s -= max_s_;
    
    return std::make_pair(s, d);
}

std::pair<double, double> FrenetConverter::frenet_to_cartesian(double s, double d) const {
    // Handle wraparound
    while (s < 0) s += max_s_;
    while (s > max_s_) s -= max_s_;
    
    // Find the segment that contains s
    int idx = 0;
    for (size_t i = 1; i < s_.size(); ++i) {
        if (s_[i] > s) {
            idx = i - 1;
            break;
        }
    }
    
    // Linear interpolation between waypoints
    double t = 0.0;
    if (idx < static_cast<int>(s_.size()) - 1) {
        double ds = s_[idx + 1] - s_[idx];
        if (ds > 1e-6) {
            t = (s - s_[idx]) / ds;
        }
    }
    
    // Interpolate position and heading
    double ref_x = x_[idx] + t * (x_[idx + 1] - x_[idx]);
    double ref_y = y_[idx] + t * (y_[idx + 1] - y_[idx]);
    double ref_psi = psi_[idx] + t * (psi_[idx + 1] - psi_[idx]);
    
    // Calculate Cartesian coordinates
    double x = ref_x + d * (-std::sin(ref_psi));
    double y = ref_y + d * std::cos(ref_psi);
    
    return std::make_pair(x, y);
}

int FrenetConverter::find_closest_point(double x, double y) const {
    double min_dist = std::numeric_limits<double>::max();
    int closest_idx = 0;
    
    for (size_t i = 0; i < x_.size(); ++i) {
        double dist = distance(x, y, x_[i], y_[i]);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    
    return closest_idx;
}

double FrenetConverter::distance(double x1, double y1, double x2, double y2) const {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx*dx + dy*dy);
}
