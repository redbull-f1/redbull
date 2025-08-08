#ifndef WAYPOINT_LOADER_HPP_
#define WAYPOINT_LOADER_HPP_

#include <vector>
#include <string>
#include <memory>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace global_path_follower {

struct Waypoint {
    double s_m = 0.0;          // Path length coordinate (frenet s)
    double x_m = 0.0;          // X position in global frame
    double y_m = 0.0;          // Y position in global frame  
    double d_right = 0.0;      // Distance to right boundary
    double d_left = 0.0;       // Distance to left boundary
    double psi_rad = 0.0;      // Heading angle in radians
    double kappa_radpm = 0.0;  // Curvature in rad/m
    double vx_mps = 0.0;       // Velocity in m/s
    double ax_mps2 = 0.0;      // Acceleration in m/s^2
    
    // Legacy compatibility fields (to be deprecated)
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double speed = 0.0;
    double curvature = 0.0;
    int index = 0;
};

class WaypointLoader {
public:
    WaypointLoader();
    ~WaypointLoader() = default;
    
    bool loadWaypoints(const std::string& csv_file_path);
    const std::vector<Waypoint>& getWaypoints() const { return waypoints_; }
    const std::vector<Waypoint>& getGlobalWaypoints() const { return waypoints_; }
    
    // Get waypoints around a specific position
    std::vector<Waypoint> getLocalWaypoints(double ego_x, double ego_y, double lookahead_distance) const;
    
    // Find closest waypoint index
    int findClosestWaypointIndex(double ego_x, double ego_y) const;

private:
    std::vector<Waypoint> waypoints_;
};

} // namespace global_path_follower

#endif // WAYPOINT_LOADER_HPP_
