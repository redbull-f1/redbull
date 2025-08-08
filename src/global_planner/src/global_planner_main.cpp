#include "global_planner_node.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
