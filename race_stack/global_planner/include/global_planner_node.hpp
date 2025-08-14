#ifndef GLOBAL_PLANNER_NODE_HPP
#define GLOBAL_PLANNER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>  
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <vector>
#include <sstream>
#include <mutex>
#include <limits>
#include <algorithm>
#include <cmath>

#include "global_planner_config.hpp"

class GlobalPlanner : public rclcpp::Node {
    public:
        // Frenet coordinate structure
        struct FrenetCoordinates {
            double s;  // longitudinal coordinate
            double d;  // lateral coordinate
        };

        GlobalPlanner(const std::string &node_name, const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
        ~GlobalPlanner();

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
        inline void CallbackEgoRaceCarOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
            mutex_ego_racecar_odom_.lock();
            i_ego_racecar_odom_ = *msg;
            b_is_ego_racecar_odom_ = true;
            mutex_ego_racecar_odom_.unlock();
        }

        // Load global trajectory from CSV file
        bool LoadGlobalTrajectoryFromCSV(const std::string& csv_filename);
        void PublishGlobalWaypoints();

        // Frenet coordinate conversion
        int                         FindClosestWaypoint(double vehicle_x, double vehicle_y);
        FrenetCoordinates           CartesianToFrenet(double x, double y);
        std::pair<double, double>   CartesianVelocityToFrenet(double vx, double vy, double vehicle_theta);
        void                        PublishFrenetOdometry(const builtin_interfaces::msg::Time& timestamp,
                                                         const FrenetCoordinates& frenet_coords,
                                                         double vs, double vd);
        void                        PublishLocalWaypoints(const FrenetCoordinates& current_frenet);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables

        // Global trajectory data
        std::vector<std::vector<double>> global_trajectory_;

        // Frenet coordinate conversion
        int last_closest_index_ = 0;
        
        // Subscribers
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr            s_car_state_odom_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr            s_ego_racecar_odom_;
        
        // Subscriber data
        nav_msgs::msg::Odometry         i_car_state_odom_;
        nav_msgs::msg::Odometry         i_ego_racecar_odom_;

        // mutex
        std::mutex mutex_car_state_odom_;
        std::mutex mutex_ego_racecar_odom_;
        
        // Publishers
        rclcpp::Publisher<f110_msgs::msg::WpntArray>::SharedPtr             p_global_waypoints_;
        rclcpp::Publisher<f110_msgs::msg::WpntArray>::SharedPtr             p_local_waypoints_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr               p_car_state_odom_;
        
        // Timer
        rclcpp::TimerBase::SharedPtr t_run_node_;
        
        // Configuration
        GlobalPlannerConfig cfg_;
        
        // Status flags
        bool b_is_car_state_odom_ = false;
        bool b_is_ego_racecar_odom_ = false;
};

#endif // GLOBAL_PLANNER_NODE_HPP