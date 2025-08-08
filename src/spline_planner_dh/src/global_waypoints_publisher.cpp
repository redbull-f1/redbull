#include <rclcpp/rclcpp.hpp>
#include "spline_planner_dh/msg/wpnt_array.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <ament_index_cpp/get_package_share_directory.hpp>

class GlobalWaypointsPublisher : public rclcpp::Node {
public:
    GlobalWaypointsPublisher() : Node("global_waypoints_publisher") {
        publisher_ = this->create_publisher<spline_planner_dh::msg::WpntArray>(
            "/global_waypoints", 10);
        
        // Add visualization publisher for global waypoints
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/global_waypoints/visualization", 10);
        
        // Declare parameter for CSV file path
        this->declare_parameter<std::string>("csv_file", "redbull_0.csv");
        
        // Load waypoints from CSV
        load_waypoints();
        
        // Create timer to publish waypoints periodically
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&GlobalWaypointsPublisher::publish_waypoints, this));
        
        RCLCPP_INFO(this->get_logger(), "Global waypoints publisher initialized");
    }

private:
    void load_waypoints() {
        std::string csv_file = this->get_parameter("csv_file").as_string();
        
        // Try to get the full path
        std::string package_share_directory;
        try {
            package_share_directory = ament_index_cpp::get_package_share_directory("spline_planner_dh");
            csv_file = package_share_directory + "/waypoints/" + csv_file;
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Could not find package share directory, using relative path");
        }
        
        RCLCPP_INFO(this->get_logger(), "Loading waypoints from: %s", csv_file.c_str());
        
        std::ifstream file(csv_file);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_file.c_str());
            return;
        }
        
        waypoints_msg_.header.frame_id = "map";
        waypoints_msg_.wpnts.clear();
        
        std::string line;
        bool first_line = true;
        int id = 0;
        
        while (std::getline(file, line)) {
            // Skip comment lines
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> values;
            
            while (std::getline(ss, cell, ';')) {
                // Remove leading/trailing whitespace
                cell.erase(0, cell.find_first_not_of(" \t"));
                cell.erase(cell.find_last_not_of(" \t") + 1);
                
                try {
                    values.push_back(std::stod(cell));
                } catch (const std::exception& e) {
                    RCLCPP_WARN(this->get_logger(), "Error parsing value: %s", cell.c_str());
                    continue;
                }
            }
            
            // Expect: s_m; x_m; y_m; d_right; d_left; psi_rad; kappa_radpm; vx_mps; ax_mps2
            if (values.size() >= 8) {
                spline_planner_dh::msg::Wpnt waypoint;
                waypoint.id = id++;
                waypoint.s_m = values[0];     // s_m
                waypoint.x_m = values[1];     // x_m  
                waypoint.y_m = values[2];     // y_m
                waypoint.d_right = values[3]; // d_right
                waypoint.d_left = values[4];  // d_left
                waypoint.psi_rad = values[5]; // psi_rad
                waypoint.kappa_radpm = values[6]; // kappa_radpm
                waypoint.vx_mps = values[7];  // vx_mps
                waypoint.d_m = 0.0;          // d offset (center of track)
                
                waypoints_msg_.wpnts.push_back(waypoint);
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints", waypoints_msg_.wpnts.size());
        file.close();
    }
    
    void publish_waypoints() {
        if (!waypoints_msg_.wpnts.empty()) {
            waypoints_msg_.header.stamp = this->get_clock()->now();
            publisher_->publish(waypoints_msg_);
            
            // Publish visualization marker
            publish_visualization_marker();
            
            RCLCPP_DEBUG(this->get_logger(), "Published %zu waypoints", waypoints_msg_.wpnts.size());
        }
    }
    
    void publish_visualization_marker() {
        visualization_msgs::msg::MarkerArray marker_array;
        
        // Create line strip marker for the path
        visualization_msgs::msg::Marker line_marker;
        line_marker.header.frame_id = "map";
        line_marker.header.stamp = this->get_clock()->now();
        line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::msg::Marker::ADD;
        line_marker.id = 0;
        line_marker.ns = "global_path";
        
        // Set line marker properties
        line_marker.scale.x = 0.1;  // Line width
        line_marker.color.a = 0.8;  // Alpha
        line_marker.color.r = 0.0;  // Red
        line_marker.color.g = 1.0;  // Green  
        line_marker.color.b = 0.0;  // Blue (green line)
        
        // Add all waypoint positions to create a line strip
        for (const auto& waypoint : waypoints_msg_.wpnts) {
            geometry_msgs::msg::Point point;
            point.x = waypoint.x_m;
            point.y = waypoint.y_m;
            point.z = 0.0;
            line_marker.points.push_back(point);
        }
        
        // Close the loop by connecting last point to first
        if (!waypoints_msg_.wpnts.empty()) {
            geometry_msgs::msg::Point first_point;
            first_point.x = waypoints_msg_.wpnts[0].x_m;
            first_point.y = waypoints_msg_.wpnts[0].y_m;
            first_point.z = 0.0;
            line_marker.points.push_back(first_point);
        }
        
        marker_array.markers.push_back(line_marker);
        
        // Add individual waypoint markers (small spheres)
        for (size_t i = 0; i < waypoints_msg_.wpnts.size(); i += 10) { // Show every 10th waypoint to avoid clutter
            const auto& waypoint = waypoints_msg_.wpnts[i];
            
            visualization_msgs::msg::Marker point_marker;
            point_marker.header.frame_id = "map";
            point_marker.header.stamp = this->get_clock()->now();
            point_marker.type = visualization_msgs::msg::Marker::SPHERE;
            point_marker.action = visualization_msgs::msg::Marker::ADD;
            point_marker.id = static_cast<int>(i + 1);
            point_marker.ns = "waypoints";
            
            point_marker.pose.position.x = waypoint.x_m;
            point_marker.pose.position.y = waypoint.y_m;
            point_marker.pose.position.z = 0.05;
            point_marker.pose.orientation.w = 1.0;
            
            point_marker.scale.x = 0.2;
            point_marker.scale.y = 0.2;
            point_marker.scale.z = 0.2;
            
            point_marker.color.a = 0.7;
            point_marker.color.r = 0.0;
            point_marker.color.g = 0.8;
            point_marker.color.b = 1.0;  // Light blue
            
            marker_array.markers.push_back(point_marker);
        }
        
        marker_pub_->publish(marker_array);
    }
    
    rclcpp::Publisher<spline_planner_dh::msg::WpntArray>::SharedPtr publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    spline_planner_dh::msg::WpntArray waypoints_msg_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<GlobalWaypointsPublisher>();
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Starting global waypoints publisher");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}
