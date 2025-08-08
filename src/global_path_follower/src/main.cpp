#include <rclcpp/rclcpp.hpp>
#include "global_path_follower/global_path_follower.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<global_path_follower::GlobalPathFollower>();
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}
