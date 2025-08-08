#ifndef FRENET_CONVERTER_HPP
#define FRENET_CONVERTER_HPP

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <cmath>

namespace lattice_planner {

struct Waypoint {
    double x, y, z;
    double yaw;
    double velocity;
    double curvature = 0.0;          // Enhanced: curvature at this point
    double road_width_left = 2.0;    // Enhanced: left road boundary
    double road_width_right = 2.0;   // Enhanced: right road boundary
    int index;
};

struct FrenetPoint {
    double s;       // longitudinal distance
    double d;       // lateral distance  
    double s_dot = 0.0;   // longitudinal velocity
    double d_dot = 0.0;   // lateral velocity
    double s_ddot = 0.0;  // longitudinal acceleration
    double d_ddot = 0.0;  // lateral acceleration
};

struct CartesianPoint {
    double x, y;
    double theta;
    double curvature = 0.0;  // Enhanced: curvature (kappa)
    double velocity = 0.0;   // Enhanced: velocity
    double acceleration = 0.0; // Enhanced: acceleration
};

class FrenetConverter {
public:
    FrenetConverter();
    
    void setReferencePath(const std::vector<Waypoint>& path);
    FrenetPoint cartesianToFrenet(double x, double y) const;
    CartesianPoint frenetToCartesian(const FrenetPoint& frenet_point) const;
    
    // Enhanced Step 2 functions
    void calculateCurvatures();
    void setRoadBoundaries(const std::vector<double>& left_widths, const std::vector<double>& right_widths);
    bool isValidFrenetPoint(const FrenetPoint& point) const;
    double getCurvatureAtS(double s) const;
    double getMaxLateralOffset(double s) const;
    
private:
    std::vector<Waypoint> reference_path_;
    std::vector<double> s_values_;  // cumulative arc length
    
    int findClosestPoint(double x, double y) const;
    double calculateDistance(double x1, double y1, double x2, double y2) const;
    
    // Enhanced Step 2 private functions
    double calculateCurvatureAtIndex(int index) const;
    std::pair<double, double> getRoadBoundariesAtS(double s) const;
};

} // namespace lattice_planner

#endif // FRENET_CONVERTER_HPP
