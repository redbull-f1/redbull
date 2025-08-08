#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <fstream>
#include <sstream>
#include <vector>

class WaypointsToPathConverter : public rclcpp::Node {
public:
    WaypointsToPathConverter() : Node("waypoints_to_path_converter") {
        // Publisher for global_waypoints and global_path
        global_waypoints_pub_ = this->create_publisher<nav_msgs::msg::Path>("/global_waypoints", 10);
        global_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/global_path", 10);
        
        // Timer to periodically publish waypoints and path
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(2000),  // 2초마다 publish
            std::bind(&WaypointsToPathConverter::publishCallback, this));
        
        // Load waypoints from CSV
        loadWaypointsFromCSV();
        
        RCLCPP_INFO(this->get_logger(), "Waypoints to Path Converter initialized");
    }

private:
    struct WaypointData {
        double x, y, z;
        double yaw;
        double velocity;
        double curvature;
        double d_right, d_left;
        int index;
    };
    
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_waypoints_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::vector<WaypointData> waypoints_;
    
    void loadWaypointsFromCSV() {
        std::string csv_file_path = "/home/jeong/sim_ws/src/planning_ws/src/lattice_planner_rb/waypoints/redbull_0.csv";
        
        std::ifstream file(csv_file_path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_file_path.c_str());
            return;
        }
        
        std::string line;
        waypoints_.clear();
        
        // Skip header lines
        std::getline(file, line); // # comment
        std::getline(file, line); // # comment
        std::getline(file, line); // header
        
        int index = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::stringstream ss(line);
            std::string item;
            std::vector<std::string> tokens;
            
            // Split by semicolon
            while (std::getline(ss, item, ';')) {
                // Remove whitespace
                item.erase(0, item.find_first_not_of(" \t"));
                item.erase(item.find_last_not_of(" \t") + 1);
                tokens.push_back(item);
            }
            
            if (tokens.size() >= 9) {
                WaypointData wp;
                wp.x = std::stod(tokens[1]);  // x_m
                wp.y = std::stod(tokens[2]);  // y_m
                wp.z = 0.0;
                wp.yaw = std::stod(tokens[5]);  // psi_rad
                wp.velocity = std::stod(tokens[7]);  // vx_mps
                wp.curvature = std::stod(tokens[6]);  // kappa_radpm
                wp.d_right = std::stod(tokens[3]);  // d_right
                wp.d_left = std::stod(tokens[4]);   // d_left
                wp.index = index++;
                
                waypoints_.push_back(wp);
            }
        }
        
        file.close();
        
        RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints from CSV file", waypoints_.size());
    }
    
    void publishCallback() {
        if (waypoints_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No waypoints to publish");
            return;
        }
        
        // Create global_waypoints message (with track boundary info in header)
        nav_msgs::msg::Path waypoints_msg;
        waypoints_msg.header.stamp = this->get_clock()->now();
        waypoints_msg.header.frame_id = "map";
        
        // Create global_path message (standard path message)
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->get_clock()->now();
        path_msg.header.frame_id = "map";
        
        for (const auto& wp : waypoints_) {
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = waypoints_msg.header.stamp;
            pose_stamped.header.frame_id = "map";
            
            // Position
            pose_stamped.pose.position.x = wp.x;
            pose_stamped.pose.position.y = wp.y;
            pose_stamped.pose.position.z = wp.z;
            
            // Orientation from yaw
            tf2::Quaternion q;
            q.setRPY(0, 0, wp.yaw);
            pose_stamped.pose.orientation = tf2::toMsg(q);
            
            // Add to both messages
            waypoints_msg.poses.push_back(pose_stamped);
            path_msg.poses.push_back(pose_stamped);
        }
        
        // Publish both messages
        global_waypoints_pub_->publish(waypoints_msg);
        global_path_pub_->publish(path_msg);
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Publishing %zu waypoints and path", waypoints_.size());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WaypointsToPathConverter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
