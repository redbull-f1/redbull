#ifndef GLOBAL_PLANNER_NODE_HPP
#define GLOBAL_PLANNER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>  
#include <chrono>
#include <functional>
#include <mutex>

// Parameter Header
#include "global_planner_config.hpp"

class GlobalPlanner : public rclcpp::Node {
    public:
        explicit GlobalPlanner(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
        virtual ~GlobalPlanner();

        void ProcessParams();
        void Run();

    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Functions     
        inline void CallbackCarStateOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
            mutex_car_state_odom_.lock();
            i_car_state_odom_ = *msg;
            b_is_car_state_odom_ = true;
            mutex_car_state_odom_.unlock();
        }

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables
        
        // Subscribers
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr s_car_state_odom_;

        // Input
        nav_msgs::msg::Odometry i_car_state_odom_;
        
        // Publishers  
        rclcpp::Publisher<f110_msgs::msg::WpntArray>::SharedPtr p_global_waypoints_;
        rclcpp::Publisher<f110_msgs::msg::WpntArray>::SharedPtr p_local_waypoints_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr p_car_state_odom_;

        // Mutex
        std::mutex mutex_car_state_odom_;
        
        // Timer
        rclcpp::TimerBase::SharedPtr t_run_node_;

        // Config
        GlobalPlannerConfig cfg_;

        // Flag
        bool b_is_car_state_odom_ = false;
};

#endif // GLOBAL_PLANNER_NODE_HPP