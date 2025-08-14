#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include "spline_planner_dh_test/msg/obstacle_array.hpp"
#include "spline_planner_dh_test/msg/obstacle_wpnt.hpp"


//====================================================================================
// obstacles_spielberg.yaml 파일을 읽어와서 정적 장애물 정보 퍼블리시하는 노드
//====================================================================================
class StaticObstaclePublisher : public rclcpp::Node {
public:
    StaticObstaclePublisher() : Node("static_obstacle_publisher") {
        // Publisher for obstacle array
        obstacle_pub_ = this->create_publisher<spline_planner_dh_test::msg::ObstacleArray>(
            "/obstacles", 10);
        
        // Timer to publish obstacles periodically
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&StaticObstaclePublisher::publish_obstacles, this));
        
        // Load obstacles from config file
        load_obstacles_from_config();
        
        RCLCPP_INFO(this->get_logger(), "Static Obstacle Publisher started with %zu obstacles", 
                    static_obstacles_.size());
    }

private:
    void load_obstacles_from_config() {
        try {
            // Get the package share directory path
            std::string config_path = "/home/jeong/sim_ws/src/planning_ws/src/spline_planner_dh_test/config/obstacles_spielberg.yaml";
            
            YAML::Node config = YAML::LoadFile(config_path);
            
            if (config["static_obstacles"]) {
                for (const auto& obs_node : config["static_obstacles"]) {
                    spline_planner_dh_test::msg::ObstacleWpnt obstacle;
                    obstacle.id = obs_node["id"].as<int>();
                    obstacle.x = obs_node["x"].as<double>();
                    obstacle.y = obs_node["y"].as<double>();
                    obstacle.vx = obs_node["vx"].as<double>();
                    obstacle.vy = obs_node["vy"].as<double>();
                    obstacle.yaw = obs_node["yaw"].as<double>();
                    obstacle.size = obs_node["size"].as<double>();
                    
                    static_obstacles_.push_back(obstacle);
                    
                    RCLCPP_INFO(this->get_logger(), 
                               "Loaded obstacle %d at (%.2f, %.2f) with size %.2f",
                               obstacle.id, obstacle.x, obstacle.y, obstacle.size);
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load obstacles config: %s", e.what());
        }
    }
    
    void publish_obstacles() {
        spline_planner_dh_test::msg::ObstacleArray obstacle_array;
        obstacle_array.header.stamp = this->get_clock()->now();
        obstacle_array.header.frame_id = "map";
        obstacle_array.obstacles = static_obstacles_;
        
        obstacle_pub_->publish(obstacle_array);
    }
    
    rclcpp::Publisher<spline_planner_dh_test::msg::ObstacleArray>::SharedPtr obstacle_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<spline_planner_dh_test::msg::ObstacleWpnt> static_obstacles_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StaticObstaclePublisher>());
    rclcpp::shutdown();
    return 0;
}
