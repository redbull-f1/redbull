#include "lattice_planner_rb/frenet_converter.hpp"

namespace lattice_planner {

FrenetConverter::FrenetConverter() {}

void FrenetConverter::setReferencePath(const std::vector<Waypoint>& path) {
    reference_path_ = path;
    s_values_.clear();
    s_values_.resize(path.size());
    
    if (path.empty()) return;
    
    s_values_[0] = 0.0;
    for (size_t i = 1; i < path.size(); i++) {
        double ds = calculateDistance(path[i-1].x, path[i-1].y, path[i].x, path[i].y);
        s_values_[i] = s_values_[i-1] + ds;
    }
    
    // Enhanced Step 2: Calculate curvatures for all waypoints
    calculateCurvatures();
    
    RCLCPP_INFO(rclcpp::get_logger("FrenetConverter"), 
                "Reference path set with %zu points, total length: %.2f m", 
                path.size(), s_values_.back());
}

FrenetPoint FrenetConverter::cartesianToFrenet(double x, double y) const {
    FrenetPoint frenet_point;
    
    if (reference_path_.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("FrenetConverter"), "Reference path is empty");
        return frenet_point;
    }
    
    int closest_idx = findClosestPoint(x, y);
    
    // Get reference point
    const auto& ref_point = reference_path_[closest_idx];
    frenet_point.s = s_values_[closest_idx];
    
    // Calculate lateral offset (d)
    double dx = x - ref_point.x;
    double dy = y - ref_point.y;
    
    // Reference direction (simplified - using yaw from waypoint)
    double ref_yaw = ref_point.yaw;
    
    // Project onto normal direction
    frenet_point.d = -dx * sin(ref_yaw) + dy * cos(ref_yaw);
    
    // Initialize derivatives to zero for now
    frenet_point.s_dot = 0.0;
    frenet_point.s_ddot = 0.0;
    frenet_point.d_dot = 0.0;
    frenet_point.d_ddot = 0.0;
    
    return frenet_point;
}

CartesianPoint FrenetConverter::frenetToCartesian(const FrenetPoint& frenet_point) const {
    CartesianPoint cartesian_point;
    
    if (reference_path_.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("FrenetConverter"), "Reference path is empty");
        return cartesian_point;
    }
    
    // Find the reference point for given s
    int ref_idx = 0;
    for (size_t i = 0; i < s_values_.size(); i++) {
        if (s_values_[i] >= frenet_point.s) {
            ref_idx = i;
            break;
        }
        ref_idx = i;
    }
    
    const auto& ref_point = reference_path_[ref_idx];
    
    // Calculate position
    double ref_x = ref_point.x;
    double ref_y = ref_point.y;
    double ref_yaw = ref_point.yaw;
    
    cartesian_point.x = ref_x - frenet_point.d * sin(ref_yaw);
    cartesian_point.y = ref_y + frenet_point.d * cos(ref_yaw);
    cartesian_point.theta = ref_yaw;
    cartesian_point.curvature = getCurvatureAtS(frenet_point.s);  // Enhanced: use calculated curvature
    cartesian_point.velocity = ref_point.velocity;
    cartesian_point.acceleration = 0.0;
    
    return cartesian_point;
}

int FrenetConverter::findClosestPoint(double x, double y) const {
    int closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < reference_path_.size(); i++) {
        double dist = calculateDistance(x, y, reference_path_[i].x, reference_path_[i].y);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    
    return closest_idx;
}

double FrenetConverter::calculateDistance(double x1, double y1, double x2, double y2) const {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Enhanced Step 2 implementations
void FrenetConverter::calculateCurvatures() {
    if (reference_path_.size() < 3) {
        RCLCPP_WARN(rclcpp::get_logger("FrenetConverter"), 
                    "Not enough points to calculate curvatures");
        return;
    }
    
    for (size_t i = 0; i < reference_path_.size(); i++) {
        reference_path_[i].curvature = calculateCurvatureAtIndex(i);
    }
    
    RCLCPP_INFO(rclcpp::get_logger("FrenetConverter"), "Curvatures calculated for reference path");
}

double FrenetConverter::calculateCurvatureAtIndex(int index) const {
    if (index <= 0 || index >= static_cast<int>(reference_path_.size()) - 1) {
        return 0.0;  // Cannot calculate curvature at endpoints
    }
    
    // Use three consecutive points to calculate curvature
    const auto& p1 = reference_path_[index - 1];
    const auto& p2 = reference_path_[index];
    const auto& p3 = reference_path_[index + 1];
    
    // Calculate vectors
    double dx1 = p2.x - p1.x;
    double dy1 = p2.y - p1.y;
    double dx2 = p3.x - p2.x;
    double dy2 = p3.y - p2.y;
    
    // Calculate curvature using cross product formula
    double cross_product = dx1 * dy2 - dy1 * dx2;
    double ds1 = std::sqrt(dx1*dx1 + dy1*dy1);
    double ds2 = std::sqrt(dx2*dx2 + dy2*dy2);
    double ds_avg = (ds1 + ds2) / 2.0;
    
    if (ds_avg < 1e-6) {
        return 0.0;
    }
    
    return cross_product / (ds_avg * ds_avg * ds_avg);
}

void FrenetConverter::setRoadBoundaries(const std::vector<double>& left_widths, 
                                       const std::vector<double>& right_widths) {
    if (left_widths.size() != reference_path_.size() || 
        right_widths.size() != reference_path_.size()) {
        RCLCPP_WARN(rclcpp::get_logger("FrenetConverter"), 
                    "Road boundary size mismatch with reference path");
        return;
    }
    
    for (size_t i = 0; i < reference_path_.size(); i++) {
        reference_path_[i].road_width_left = left_widths[i];
        reference_path_[i].road_width_right = right_widths[i];
    }
    
    RCLCPP_INFO(rclcpp::get_logger("FrenetConverter"), "Road boundaries updated");
}

bool FrenetConverter::isValidFrenetPoint(const FrenetPoint& point) const {
    if (reference_path_.empty()) {
        return false;
    }
    
    // Check if s is within valid range
    if (point.s < 0.0 || point.s > s_values_.back()) {
        return false;
    }
    
    // Check if lateral offset is within road boundaries
    auto boundaries = getRoadBoundariesAtS(point.s);
    double max_left = boundaries.first;
    double max_right = boundaries.second;
    
    return (point.d >= -max_right && point.d <= max_left);
}

double FrenetConverter::getCurvatureAtS(double s) const {
    if (reference_path_.empty()) {
        return 0.0;
    }
    
    // Find the reference point for given s
    int ref_idx = 0;
    for (size_t i = 0; i < s_values_.size(); i++) {
        if (s_values_[i] >= s) {
            ref_idx = i;
            break;
        }
        ref_idx = i;
    }
    
    if (ref_idx >= static_cast<int>(reference_path_.size())) {
        ref_idx = reference_path_.size() - 1;
    }
    
    return reference_path_[ref_idx].curvature;
}

double FrenetConverter::getMaxLateralOffset(double s) const {
    auto boundaries = getRoadBoundariesAtS(s);
    return std::min(boundaries.first, boundaries.second);  // Return the smaller of left/right for safety
}

std::pair<double, double> FrenetConverter::getRoadBoundariesAtS(double s) const {
    if (reference_path_.empty()) {
        return {2.0, 2.0};  // Default boundaries
    }
    
    // Find the reference point for given s
    int ref_idx = 0;
    for (size_t i = 0; i < s_values_.size(); i++) {
        if (s_values_[i] >= s) {
            ref_idx = i;
            break;
        }
        ref_idx = i;
    }
    
    if (ref_idx >= static_cast<int>(reference_path_.size())) {
        ref_idx = reference_path_.size() - 1;
    }
    
    const auto& ref_point = reference_path_[ref_idx];
    return {ref_point.road_width_left, ref_point.road_width_right};
}

} // namespace lattice_planner
