#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include "spline_planner_dh_test/msg/wpnt_array.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ament_index_cpp/get_package_share_directory.hpp>
//#include <yaml-cpp/yaml.h>
#include <chrono>

class WaypointVisualizer : public rclcpp::Node
{
public:
    WaypointVisualizer() : Node("waypoint_visualizer")
    {
        // Parameter declaration
        this->declare_parameter<std::string>("csv_file", "Spielberg_raceline.csv");
        //this->declare_parameter<std::string>("obstacles_file", "obstacles_spielberg.yaml");
        this->declare_parameter<double>("publish_rate", 1.0);

        // Get parameters
        csv_file_ = this->get_parameter("csv_file").as_string();
        //obstacles_file_ = this->get_parameter("obstacles_file").as_string();
        double publish_rate = this->get_parameter("publish_rate").as_double();

        // Create publisher for visualization markers
        visualization_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/global_waypoints/visualization", 10);

        // Load waypoints and obstacles
        load_waypoints_from_csv();
        //load_obstacles_from_yaml();

        // Create timer for publishing visualization
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_rate),
            std::bind(&WaypointVisualizer::publish_visualization_marker, this));

        RCLCPP_INFO(this->get_logger(), "Waypoint visualizer started");
    }

private:
    void load_waypoints_from_csv()
    {
        try {
            std::string package_share_directory = ament_index_cpp::get_package_share_directory("spline_planner_dh_test");
            std::string full_path = package_share_directory + "/waypoints/" + csv_file_;
            
            std::ifstream file(full_path);
            if (!file.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Could not open CSV file: %s", full_path.c_str());
                return;
            }

            std::string line;
            while (std::getline(file, line)) {
                if (line.empty() || line[0] == '#') continue;
                
                std::stringstream ss(line);
                std::string token;
                std::vector<std::string> tokens;
                
                while (std::getline(ss, token, ';')) {
                    tokens.push_back(token);
                }
                
                if (tokens.size() >= 3) {
                    spline_planner_dh_test::msg::Wpnt waypoint;
                    waypoint.s_m = std::stod(tokens[0]);
                    waypoint.x_m = std::stod(tokens[1]);
                    waypoint.y_m = std::stod(tokens[2]);
                    
                    if (tokens.size() >= 4) {
                        waypoint.d_right = std::stod(tokens[3]);
                    }
                    if (tokens.size() >= 5) {
                        waypoint.d_left = std::stod(tokens[4]);
                    }
                    if (tokens.size() >= 6) {
                        waypoint.psi_rad = std::stod(tokens[5]);
                    }
                    if (tokens.size() >= 7) {
                        waypoint.kappa_radpm = std::stod(tokens[6]);
                    }
                    if (tokens.size() >= 8) {
                        waypoint.vx_mps = std::stod(tokens[7]);
                    }
                    
                    waypoints_.push_back(waypoint);
                }
            }
            
            file.close();
            RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints for visualization", waypoints_.size());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading waypoints: %s", e.what());
        }
    }

    //void load_obstacles_from_yaml()
    //{
    //    try {
    //        std::string package_share_directory = ament_index_cpp::get_package_share_directory("spline_planner_dh_test");
    //        std::string full_path = package_share_directory + "/config/" + obstacles_file_;
    //        
    //        YAML::Node config = YAML::LoadFile(full_path);
    //        
    //        if (config["static_obstacles"]) {
    //            for (const auto& obstacle : config["static_obstacles"]) {
    //                geometry_msgs::msg::Point point;
    //                point.x = obstacle["x"].as<double>();
    //                point.y = obstacle["y"].as<double>();
    //                point.z = 0.0;
    //                obstacles_.push_back(point);
    //            }
    //            RCLCPP_INFO(this->get_logger(), "Loaded %zu static obstacles", obstacles_.size());
    //        }
    //    } catch (const std::exception& e) {
    //        RCLCPP_ERROR(this->get_logger(), "Error loading obstacles: %s", e.what());
    //    }
    //}

    void publish_visualization_marker()
    {
        visualization_msgs::msg::MarkerArray marker_array;

        // Clear previous markers
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header.frame_id = "map";
        clear_marker.header.stamp = this->get_clock()->now();
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        if (!waypoints_.empty()) {
            // Line strip for raceline - RED
            visualization_msgs::msg::Marker line_marker;
            line_marker.header.frame_id = "map";
            line_marker.header.stamp = this->get_clock()->now();
            line_marker.ns = "raceline";
            line_marker.id = 0;
            line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            line_marker.action = visualization_msgs::msg::Marker::ADD;
            line_marker.scale.x = 0.1;  // Line width
            line_marker.color.r = 1.0;  // Red
            line_marker.color.g = 0.0;
            line_marker.color.b = 0.0;
            line_marker.color.a = 1.0;
            line_marker.pose.orientation.w = 1.0;

            for (const auto& waypoint : waypoints_) {
                geometry_msgs::msg::Point point;
                point.x = waypoint.x_m;
                point.y = waypoint.y_m;
                point.z = 0.0;
                line_marker.points.push_back(point);
            }
            marker_array.markers.push_back(line_marker);

            // Points for waypoints - LIGHT GREEN
            visualization_msgs::msg::Marker waypoint_marker;
            waypoint_marker.header.frame_id = "map";
            waypoint_marker.header.stamp = this->get_clock()->now();
            waypoint_marker.ns = "waypoints";
            waypoint_marker.id = 1;
            waypoint_marker.type = visualization_msgs::msg::Marker::POINTS;
            waypoint_marker.action = visualization_msgs::msg::Marker::ADD;
            waypoint_marker.scale.x = 0.05;  // Point width
            waypoint_marker.scale.y = 0.05;  // Point height
            waypoint_marker.color.r = 0.5;   // Light green
            waypoint_marker.color.g = 1.0;
            waypoint_marker.color.b = 0.5;
            waypoint_marker.color.a = 1.0;
            waypoint_marker.pose.orientation.w = 1.0;

            for (size_t i = 0; i < waypoints_.size(); i += 1) {
                geometry_msgs::msg::Point point;
                point.x = waypoints_[i].x_m;
                point.y = waypoints_[i].y_m;
                point.z = 0.0;
                waypoint_marker.points.push_back(point);
            }
            marker_array.markers.push_back(waypoint_marker);
        }

        /* 
        Add obstacle markers - BLUE SPHERES + LABELS

        if (!obstacles_.empty()) {
            for (size_t i = 0; i < obstacles_.size(); ++i) {
                // 1) 파란 구
                visualization_msgs::msg::Marker obstacle_marker;
                obstacle_marker.header.frame_id = "map";
                obstacle_marker.header.stamp = this->get_clock()->now();
                obstacle_marker.ns = "static_obstacles";
                obstacle_marker.id = static_cast<int>(i) + 100;  // Offset to avoid ID conflicts
                obstacle_marker.type = visualization_msgs::msg::Marker::SPHERE;
                obstacle_marker.action = visualization_msgs::msg::Marker::ADD;
                obstacle_marker.pose.position = obstacles_[i];
                obstacle_marker.pose.orientation.w = 1.0;
                obstacle_marker.scale.x = 0.5;  // Sphere size
                obstacle_marker.scale.y = 0.5;
                obstacle_marker.scale.z = 0.5;
                obstacle_marker.color.r = 0.0;
                obstacle_marker.color.g = 0.0;
                obstacle_marker.color.b = 1.0;  // Blue
                obstacle_marker.color.a = 0.8;
                marker_array.markers.push_back(obstacle_marker);

                // 2) 텍스트 라벨 (obstacle1, obstacle2, ...)
                visualization_msgs::msg::Marker label_marker;
                label_marker.header.frame_id = "map";
                label_marker.header.stamp = this->get_clock()->now();
                label_marker.ns = "static_obstacles_label";
                label_marker.id = static_cast<int>(i) + 1100; // 구와 겹치지 않게 다른 ID 대역
                label_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                label_marker.action = visualization_msgs::msg::Marker::ADD;

                label_marker.pose.position = obstacles_[i];
                label_marker.pose.position.z += 0.6;          // 구 위로 살짝 띄우기
                label_marker.pose.orientation.w = 1.0;

                // TEXT_VIEW_FACING는 scale.z만 사용(글자 높이)
                label_marker.scale.z = 0.35;

                // 글자색(검정) + 불투명
                label_marker.color.r = 0.0;
                label_marker.color.g = 0.0;
                label_marker.color.b = 0.0;
                label_marker.color.a = 1.0;

                label_marker.text = "obstacle" + std::to_string(i + 1);
                marker_array.markers.push_back(label_marker);
            }
        }
        */
        visualization_pub_->publish(marker_array);
    }

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualization_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<spline_planner_dh_test::msg::Wpnt> waypoints_;
    //std::vector<geometry_msgs::msg::Point> obstacles_;
    std::string csv_file_;
    //std::string obstacles_file_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WaypointVisualizer>());
    rclcpp::shutdown();
    return 0;
}
