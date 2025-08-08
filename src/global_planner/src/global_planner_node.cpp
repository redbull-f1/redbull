#include "global_planner_node.hpp"

GlobalPlanner::GlobalPlanner(const std::string &node_name, const rclcpp::NodeOptions &options)
    : Node(node_name, options)  {
    
    // QoS init
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
    
    // Parameters
    this->declare_parameter("global_planner/loop_rate_hz", 100.0);

    ProcessParams();
    RCLCPP_INFO(this->get_logger(), "loop_rate_hz: %f", cfg_.loop_rate_hz);
    
    // Subscriber init
    s_car_state_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/car_state/odom", qos_profile, std::bind(&GlobalPlanner::CallbackCarStateOdom, this, std::placeholders::_1));
    
    // Publisher init
    p_global_waypoints_ = this->create_publisher<f110_msgs::msg::WpntArray>(
        "/global_waypoints", qos_profile);
    p_local_waypoints_ = this->create_publisher<f110_msgs::msg::WpntArray>(
        "/local_waypoints", qos_profile);
    p_car_state_odom_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/car_state/frenet/odom", qos_profile);
    
    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); });
}

GlobalPlanner::~GlobalPlanner() {}

void GlobalPlanner::ProcessParams() {
    this->get_parameter("global_planner/loop_rate_hz", cfg_.loop_rate_hz);
}

void GlobalPlanner::Run() {
    auto current_time = this->now();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "Running ...");
    ProcessParams();

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get subscribe variables
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    if (b_is_car_state_odom_ == false) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 1000, "Waiting for car state odom...");
        return;
    }

    mutex_car_state_odom_.lock();
    auto car_state_odom = i_car_state_odom_;
    mutex_car_state_odom_.unlock();
}

int main(int argc, char **argv) {
    std::string node_name = "global_planner";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalPlanner>(node_name));
    rclcpp::shutdown();
    return 0;
}