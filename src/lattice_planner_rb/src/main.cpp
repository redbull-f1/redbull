#include "lattice_planner_rb/lattice_planner.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<lattice_planner::LatticePlanner>();
    
    RCLCPP_INFO(node->get_logger(), "Starting Lattice Planner...");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}
