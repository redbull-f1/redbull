#include "global_path_follower/waypoint_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace global_path_follower {

WaypointLoader::WaypointLoader() {
}

bool WaypointLoader::loadWaypoints(const std::string& csv_file_path) {
    waypoints_.clear();
    
    std::ifstream file(csv_file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open waypoint file: " << csv_file_path << std::endl;
        return false;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        // Skip comment lines and header line
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Skip first non-comment line (header)
        if (first_line) {
            first_line = false;
            continue;
        }
        
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        
        // Use semicolon as delimiter
        while (std::getline(ss, cell, ';')) {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t"));
            cell.erase(cell.find_last_not_of(" \t") + 1);
            row.push_back(cell);
        }
        
        if (row.size() >= 9) {  // Ensure we have enough columns
            Waypoint wp;
            try {
                wp.s_m = std::stod(row[0]);
                wp.x_m = std::stod(row[1]);
                wp.y_m = std::stod(row[2]);
                wp.psi_rad = std::stod(row[3]);
                wp.kappa_radpm = std::stod(row[4]);
                wp.vx_mps = std::stod(row[5]);
                wp.ax_mps2 = std::stod(row[6]);
                wp.d_right = std::stod(row[7]);
                wp.d_left = std::stod(row[8]);
                
                waypoints_.push_back(wp);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing waypoint line: " << line << std::endl;
                continue;
            }
        }
    }
    
    file.close();
    std::cout << "Loaded " << waypoints_.size() << " waypoints from " << csv_file_path << std::endl;
    return !waypoints_.empty();
}

std::vector<Waypoint> WaypointLoader::getLocalWaypoints(double ego_x, double ego_y, double lookahead_distance) const {
    std::vector<Waypoint> local_waypoints;
    
    if (waypoints_.empty()) {
        return local_waypoints;
    }
    
    int closest_idx = findClosestWaypointIndex(ego_x, ego_y);
    if (closest_idx < 0) {
        return local_waypoints;
    }
    
    // Add waypoints within lookahead distance
    for (size_t i = closest_idx; i < waypoints_.size(); ++i) {
        double dx = waypoints_[i].x_m - ego_x;
        double dy = waypoints_[i].y_m - ego_y;
        double distance = std::sqrt(dx*dx + dy*dy);
        
        if (distance <= lookahead_distance) {
            local_waypoints.push_back(waypoints_[i]);
        } else {
            break;  // Waypoints are sequential, so we can break
        }
    }
    
    return local_waypoints;
}

int WaypointLoader::findClosestWaypointIndex(double ego_x, double ego_y) const {
    if (waypoints_.empty()) {
        return -1;
    }
    
    int closest_idx = -1;
    double min_distance = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < waypoints_.size(); ++i) {
        double dx = waypoints_[i].x_m - ego_x;
        double dy = waypoints_[i].y_m - ego_y;
        double distance = std::sqrt(dx*dx + dy*dy);
        
        if (distance < min_distance) {
            min_distance = distance;
            closest_idx = static_cast<int>(i);
        }
    }
    
    return closest_idx;
}

} // namespace global_path_follower
