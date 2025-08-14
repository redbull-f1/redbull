#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include "spline_planner_dh_test/msg/wpnt_array.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ament_index_cpp/get_package_share_directory.hpp>

class GlobalWaypointsPublisher : public rclcpp::Node
{
public:
    GlobalWaypointsPublisher() : Node("global_waypoints_publisher")
    {
        // Parameter declaration
        this->declare_parameter<std::string>("csv_file", "Spielberg_raceline.csv");
        this->declare_parameter<double>("publish_rate", 1.0);

        // Get parameters
        csv_file_ = this->get_parameter("csv_file").as_string();
        double publish_rate = this->get_parameter("publish_rate").as_double();

        // Create publisher
        publisher_ = this->create_publisher<spline_planner_dh_test::msg::WpntArray>("/global_waypoints", 10);

        // Load waypoints from CSV
        load_waypoints_from_csv();

        // Create timer for publishing
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_rate),
            std::bind(&GlobalWaypointsPublisher::publish_waypoints, this));

        RCLCPP_INFO(this->get_logger(), "Global waypoints publisher started");
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
                if (line.empty() || line[0] == '#') continue;  // Skip empty lines and comments
                
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
                    
                    waypoints_msg_.wpnts.push_back(waypoint);
                }
            }
            
            file.close();
            RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints from %s", waypoints_msg_.wpnts.size(), csv_file_.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading waypoints: %s", e.what());
        }
    }

    void publish_waypoints()
    {
        if (!waypoints_msg_.wpnts.empty()) {
            waypoints_msg_.header.stamp = this->get_clock()->now();
            waypoints_msg_.header.frame_id = "map";
            publisher_->publish(waypoints_msg_);
        }
    }

    rclcpp::Publisher<spline_planner_dh_test::msg::WpntArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    spline_planner_dh_test::msg::WpntArray waypoints_msg_;
    std::string csv_file_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalWaypointsPublisher>());
    rclcpp::shutdown();
    return 0;
}
