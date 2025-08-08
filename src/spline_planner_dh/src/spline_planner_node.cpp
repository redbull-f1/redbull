#include "spline_planner_dh/spline_planner.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<SplinePlanner>();
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Starting spline planner node");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}
