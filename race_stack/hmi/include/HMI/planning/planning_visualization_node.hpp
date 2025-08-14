#ifndef PLANNING_VISUALIZATION_NODE_HPP
#define PLANNING_VISUALIZATION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <mutex>

class PlanningVisualizationNode : public rclcpp::Node {
    public:
        PlanningVisualizationNode(const std::string &node_name, const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
        ~PlanningVisualizationNode();

        void ProcessParams();
        void Run();

    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Functions
        bool LoadCenterlineFromCSV(const std::string& csv_filename);
        void PublishCenterlineMarkers();
        void PublishGlobalWaypointsMarkers();
        void PublishLocalWaypointsMarkers();

        // Callbacks (inline functions)
        inline void CallbackGlobalWaypoints(const f110_msgs::msg::WpntArray::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(mutex_global_waypoints_);
            global_waypoints_ = *msg;
            b_is_global_waypoints_ = true;
            RCLCPP_DEBUG(this->get_logger(), "Received global waypoints: %zu points", msg->wpnts.size());
        }
        
        inline void CallbackLocalWaypoints(const f110_msgs::msg::WpntArray::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(mutex_local_waypoints_);
            local_waypoints_ = *msg;
            b_is_local_waypoints_ = true;
            RCLCPP_DEBUG(this->get_logger(), "Received local waypoints: %zu points", msg->wpnts.size());
        }

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables

        // Centerline data (x, y, w_tr_right, w_tr_left)
        std::vector<std::vector<double>> centerline_data_;

        // Waypoints data
        f110_msgs::msg::WpntArray global_waypoints_;
        f110_msgs::msg::WpntArray local_waypoints_;
        
        // Status flags
        bool b_is_global_waypoints_ = false;
        bool b_is_local_waypoints_ = false;
        
        // Mutexes
        std::mutex mutex_global_waypoints_;
        std::mutex mutex_local_waypoints_;

        // Subscribers
        rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr s_global_waypoints_;
        rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr s_local_waypoints_;

        // Publishers
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  p_centerline_markers_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  p_global_waypoints_markers_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  p_local_waypoints_markers_;

        // Timer
        rclcpp::TimerBase::SharedPtr t_run_node_;

        // Parameters
        double loop_rate_hz_;
        std::string centerline_csv_file_;
};

#endif // PLANNING_VISUALIZATION_NODE_HPP
