#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>
#include <cmath>

class SimplePathFollower : public rclcpp::Node {
public:
    SimplePathFollower() : Node("simple_path_follower") {
        // Parameters
        target_speed_ = 2.0;  // m/s - 테스트를 위해 속도 증가
        lookahead_distance_ = 2.0;  // meters
        max_steering_angle_ = 0.5;  // radians (~28.6 degrees)
        
        // Subscribers
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/global_waypoints", 10,
            std::bind(&SimplePathFollower::pathCallback, this, std::placeholders::_1));
        
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10,
            std::bind(&SimplePathFollower::odomCallback, this, std::placeholders::_1));
        
        // Publisher
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        
        // Timer for control loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),  // 20 Hz
            std::bind(&SimplePathFollower::controlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "Simple Path Follower initialized");
    }

private:
    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (msg->poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty path");
            return;
        }
        
        global_path_ = *msg;
        path_received_ = true;
        
        RCLCPP_INFO(this->get_logger(), "Received path with %zu waypoints", msg->poses.size());
    }
    
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_pose_ = msg->pose.pose;
        odom_received_ = true;
    }
    
    void controlLoop() {
        if (!path_received_ || !odom_received_ || global_path_.poses.empty()) {
            return;
        }
        
        // Find closest waypoint
        int closest_idx = findClosestWaypoint();
        if (closest_idx == -1) {
            return;
        }
        
        // Find lookahead point
        int lookahead_idx = findLookaheadPoint(closest_idx);
        if (lookahead_idx == -1) {
            // End of path reached
            publishStop();
            return;
        }
        
        // Calculate steering angle using pure pursuit
        double steering_angle = calculateSteeringAngle(lookahead_idx);
        
        // Publish drive command
        publishDriveCommand(steering_angle);
    }
    
    int findClosestWaypoint() {
        if (global_path_.poses.empty()) return -1;
        
        double min_dist = std::numeric_limits<double>::max();
        int closest_idx = 0;
        
        for (size_t i = 0; i < global_path_.poses.size(); i++) {
            double dx = global_path_.poses[i].pose.position.x - current_pose_.position.x;
            double dy = global_path_.poses[i].pose.position.y - current_pose_.position.y;
            double dist = std::sqrt(dx*dx + dy*dy);
            
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }
        
        return closest_idx;
    }
    
    int findLookaheadPoint(int start_idx) {
        for (size_t i = start_idx; i < global_path_.poses.size(); i++) {
            double dx = global_path_.poses[i].pose.position.x - current_pose_.position.x;
            double dy = global_path_.poses[i].pose.position.y - current_pose_.position.y;
            double dist = std::sqrt(dx*dx + dy*dy);
            
            if (dist >= lookahead_distance_) {
                return i;
            }
        }
        
        // If no point found at lookahead distance, return last point
        if (global_path_.poses.size() > 0) {
            return global_path_.poses.size() - 1;
        }
        
        return -1;
    }
    
    double calculateSteeringAngle(int lookahead_idx) {
        // Get lookahead point
        auto& lookahead_pose = global_path_.poses[lookahead_idx].pose;
        
        // Transform lookahead point to vehicle frame
        double dx = lookahead_pose.position.x - current_pose_.position.x;
        double dy = lookahead_pose.position.y - current_pose_.position.y;
        
        // Get current yaw
        double current_yaw = tf2::getYaw(current_pose_.orientation);
        
        // Transform to vehicle frame
        double local_x = dx * cos(current_yaw) + dy * sin(current_yaw);
        double local_y = -dx * sin(current_yaw) + dy * cos(current_yaw);
        
        // Pure pursuit algorithm
        double ld = std::sqrt(local_x*local_x + local_y*local_y);  // actual lookahead distance
        if (ld < 0.1) {  // too close
            return 0.0;
        }
        
        double curvature = 2.0 * local_y / (ld * ld);
        double wheelbase = 0.33;  // F1TENTH wheelbase
        double steering_angle = atan(wheelbase * curvature);
        
        // Limit steering angle
        steering_angle = std::max(-max_steering_angle_, std::min(max_steering_angle_, steering_angle));
        
        return steering_angle;
    }
    
    void publishDriveCommand(double steering_angle) {
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->get_clock()->now();
        drive_msg.header.frame_id = "base_link";
        
        drive_msg.drive.speed = target_speed_;
        drive_msg.drive.steering_angle = steering_angle;
        
        drive_pub_->publish(drive_msg);
        
        // Log every 1 second
        static auto last_log_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count() > 1000) {
            RCLCPP_INFO(this->get_logger(), "Speed: %.2f m/s, Steering: %.3f rad (%.1f deg)", 
                        target_speed_, steering_angle, steering_angle * 180.0 / M_PI);
            last_log_time = now;
        }
    }
    
    void publishStop() {
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->get_clock()->now();
        drive_msg.header.frame_id = "base_link";
        
        drive_msg.drive.speed = 0.0;
        drive_msg.drive.steering_angle = 0.0;
        
        drive_pub_->publish(drive_msg);
    }

    // ROS2 subscribers and publishers
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // State variables
    nav_msgs::msg::Path global_path_;
    geometry_msgs::msg::Pose current_pose_;
    bool path_received_ = false;
    bool odom_received_ = false;
    
    // Parameters
    double target_speed_;
    double lookahead_distance_;
    double max_steering_angle_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimplePathFollower>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
