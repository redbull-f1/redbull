#include "global_path_follower/global_path_follower.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace global_path_follower {

GlobalPathFollower::GlobalPathFollower(const rclcpp::NodeOptions& options)
    : Node("global_path_follower", options),
      odom_received_(false) {
    
    // Load parameters
    loadParameters();
    
    // Initialize waypoint loader
    waypoint_loader_ = std::make_unique<WaypointLoader>();
    
    // Load waypoints
    std::string package_share_directory;
    try {
        package_share_directory = ament_index_cpp::get_package_share_directory("global_path_follower");
        std::string full_waypoint_path = package_share_directory + "/waypoints/" + waypoint_file_;
        
        if (!waypoint_loader_->loadWaypoints(full_waypoint_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load waypoints from: %s", full_waypoint_path.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Successfully loaded waypoints from: %s", full_waypoint_path.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error loading waypoints: %s", e.what());
        return;
    }
    
    // Create global path from waypoints
    global_path_.header.frame_id = map_frame_;
    const auto& waypoints = waypoint_loader_->getWaypoints();
    for (const auto& wp : waypoints) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = map_frame_;
        pose.pose.position.x = wp.x_m;
        pose.pose.position.y = wp.y_m;
        pose.pose.position.z = 0.0;
        
        tf2::Quaternion q;
        q.setRPY(0, 0, wp.psi_rad);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        
        global_path_.poses.push_back(pose);
    }
    
    // Initialize subscribers
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/ego_racecar/odom", 10,
        std::bind(&GlobalPathFollower::odomCallback, this, std::placeholders::_1));
    
    // Initialize publishers
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/global_path", 10);
    vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/global_path_vis", 10);
    global_waypoints_pub_ = this->create_publisher<global_path_follower::msg::WpntArray>("/global_waypoints", 10);
    
    // Initialize timer
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / control_frequency_)),
        std::bind(&GlobalPathFollower::timerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), "Global Path Follower initialized");
}

void GlobalPathFollower::loadParameters() {
    waypoint_file_ = this->declare_parameter<std::string>("waypoint_file", "redbull_0.csv");
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    control_frequency_ = this->declare_parameter<double>("control_frequency", 20.0);
    lookahead_distance_ = this->declare_parameter<double>("lookahead_distance", 2.0);  // Reduced from 3.0
    max_speed_ = this->declare_parameter<double>("max_speed", 1.5);  // Reduced from 2.0
    min_speed_ = this->declare_parameter<double>("min_speed", 0.3);  // Reduced from 0.5
    wheelbase_ = this->declare_parameter<double>("wheelbase", 0.33);
}

void GlobalPathFollower::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    vehicle_state_.x = msg->pose.pose.position.x;
    vehicle_state_.y = msg->pose.pose.position.y;
    vehicle_state_.yaw = tf2::getYaw(msg->pose.pose.orientation);
    vehicle_state_.speed = std::sqrt(
        msg->twist.twist.linear.x * msg->twist.twist.linear.x +
        msg->twist.twist.linear.y * msg->twist.twist.linear.y
    );
    
    odom_received_ = true;
    
    // Debug output
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Ego pose: x=%.2f, y=%.2f, yaw=%.3f, speed=%.2f", 
                         vehicle_state_.x, vehicle_state_.y, vehicle_state_.yaw, vehicle_state_.speed);
}

void GlobalPathFollower::timerCallback() {
    if (!odom_received_) {
        return;
    }
    
    // Publish global path for visualization
    global_path_.header.stamp = this->get_clock()->now();
    path_pub_->publish(global_path_);
    
    // Publish global waypoints
    publishGlobalWaypoints();
    
    // Follow global path
    followGlobalPath();
    
    // Publish visualization
    publishVisualization();
}

void GlobalPathFollower::followGlobalPath() {
    auto drive_cmd = calculatePurePursuitCommand();
    drive_pub_->publish(drive_cmd);
}

ackermann_msgs::msg::AckermannDriveStamped GlobalPathFollower::calculatePurePursuitCommand() {
    ackermann_msgs::msg::AckermannDriveStamped drive_cmd;
    drive_cmd.header.stamp = this->get_clock()->now();
    drive_cmd.header.frame_id = "base_link";
    
    if (!odom_received_ || global_path_.poses.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "No odom received or empty path. odom_received: %d, path_size: %zu", 
                            odom_received_, global_path_.poses.size());
        return drive_cmd;  // Return zero command
    }
    
    // Find target waypoint using lookahead distance
    geometry_msgs::msg::Point target_point;
    bool target_found = false;
    double min_distance = std::numeric_limits<double>::max();
    int closest_idx = -1;
    
    // Find closest waypoint first
    for (size_t i = 0; i < global_path_.poses.size(); ++i) {
        double dx = global_path_.poses[i].pose.position.x - vehicle_state_.x;
        double dy = global_path_.poses[i].pose.position.y - vehicle_state_.y;
        double distance = std::sqrt(dx*dx + dy*dy);
        
        if (distance < min_distance) {
            min_distance = distance;
            closest_idx = i;
        }
    }
    
    // Find target waypoint starting from closest point
    for (size_t i = closest_idx; i < global_path_.poses.size(); ++i) {
        double dx = global_path_.poses[i].pose.position.x - vehicle_state_.x;
        double dy = global_path_.poses[i].pose.position.y - vehicle_state_.y;
        double distance = std::sqrt(dx*dx + dy*dy);
        
        if (distance >= lookahead_distance_) {
            target_point = global_path_.poses[i].pose.position;
            target_found = true;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                               "Found target at idx %zu, distance: %.2f", i, distance);
            break;
        }
    }
    
    if (!target_found && closest_idx >= 0 && closest_idx < (int)global_path_.poses.size()) {
        // Use a point ahead of the closest point
        size_t target_idx = std::min(closest_idx + 10, (int)global_path_.poses.size() - 1);
        target_point = global_path_.poses[target_idx].pose.position;
        target_found = true;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "Using ahead waypoint at idx %zu as target", target_idx);
    }
    
    if (target_found) {
        // Calculate pure pursuit steering
        double dx = target_point.x - vehicle_state_.x;
        double dy = target_point.y - vehicle_state_.y;
        double actual_lookahead = std::sqrt(dx*dx + dy*dy);
        
        // Limit the actual lookahead distance to prevent too large values
        if (actual_lookahead > 2.0 * lookahead_distance_) {
            // If too far, use a closer point
            actual_lookahead = lookahead_distance_;
            double ratio = lookahead_distance_ / std::sqrt(dx*dx + dy*dy);
            dx *= ratio;
            dy *= ratio;
            target_point.x = vehicle_state_.x + dx;
            target_point.y = vehicle_state_.y + dy;
        }
        
        // Transform to vehicle coordinate frame
        double target_x_vehicle = dx * cos(vehicle_state_.yaw) + dy * sin(vehicle_state_.yaw);
        double target_y_vehicle = -dx * sin(vehicle_state_.yaw) + dy * cos(vehicle_state_.yaw);
        
        // Pure pursuit formula
        double curvature = 2.0 * target_y_vehicle / (actual_lookahead * actual_lookahead);
        double steering_angle = atan(wheelbase_ * curvature);
        
        // Limit steering angle
        double max_steering = 0.4;  // radians
        steering_angle = std::max(-max_steering, std::min(max_steering, steering_angle));
        
        // Set Ackermann drive command
        drive_cmd.drive.speed = std::max(min_speed_, std::min(max_speed_, max_speed_));
        drive_cmd.drive.steering_angle = steering_angle;
        
        // Reduce speed when turning
        if (std::abs(steering_angle) > 0.2) {
            drive_cmd.drive.speed *= 0.7;
        }
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                           "Ego: (%.2f,%.2f), Target: (%.2f,%.2f), actual_lookahead=%.2f, target_vehicle=(%.2f,%.2f), Steering: %.3f, Speed: %.2f", 
                           vehicle_state_.x, vehicle_state_.y, target_point.x, target_point.y, 
                           actual_lookahead, target_x_vehicle, target_y_vehicle, steering_angle, drive_cmd.drive.speed);
    } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "No target found! Closest distance: %.2f", min_distance);
    }
    
    return drive_cmd;
}

void GlobalPathFollower::publishVisualization() {
    visualization_msgs::msg::MarkerArray marker_array;
    
    // Clear previous markers
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);
    
    // Visualize global waypoints as spheres
    visualization_msgs::msg::Marker waypoints_marker;
    waypoints_marker.header.frame_id = map_frame_;
    waypoints_marker.header.stamp = this->get_clock()->now();
    waypoints_marker.id = 0;
    waypoints_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    waypoints_marker.action = visualization_msgs::msg::Marker::ADD;
    waypoints_marker.scale.x = 0.2;
    waypoints_marker.scale.y = 0.2;
    waypoints_marker.scale.z = 0.2;
    waypoints_marker.color.r = 0.0;
    waypoints_marker.color.g = 1.0;
    waypoints_marker.color.b = 0.0;
    waypoints_marker.color.a = 0.8;
    
    const auto& waypoints = waypoint_loader_->getWaypoints();
    for (const auto& wp : waypoints) {
        geometry_msgs::msg::Point point;
        point.x = wp.x_m;
        point.y = wp.y_m;
        point.z = 0.0;
        waypoints_marker.points.push_back(point);
    }
    marker_array.markers.push_back(waypoints_marker);
    
    // Visualize global path as line strip
    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = map_frame_;
    path_marker.header.stamp = this->get_clock()->now();
    path_marker.id = 1;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.scale.x = 0.1;
    path_marker.color.r = 1.0;
    path_marker.color.g = 0.0;
    path_marker.color.b = 0.0;
    path_marker.color.a = 1.0;
    
    for (const auto& pose : global_path_.poses) {
        geometry_msgs::msg::Point point;
        point.x = pose.pose.position.x;
        point.y = pose.pose.position.y;
        point.z = 0.0;
        path_marker.points.push_back(point);
    }
    marker_array.markers.push_back(path_marker);
    
    // Visualize ego vehicle
    visualization_msgs::msg::Marker ego_marker;
    ego_marker.header.frame_id = map_frame_;
    ego_marker.header.stamp = this->get_clock()->now();
    ego_marker.id = 2;
    ego_marker.type = visualization_msgs::msg::Marker::ARROW;
    ego_marker.action = visualization_msgs::msg::Marker::ADD;
    
    ego_marker.pose.position.x = vehicle_state_.x;
    ego_marker.pose.position.y = vehicle_state_.y;
    ego_marker.pose.position.z = 0.0;
    
    tf2::Quaternion q;
    q.setRPY(0, 0, vehicle_state_.yaw);
    ego_marker.pose.orientation.x = q.x();
    ego_marker.pose.orientation.y = q.y();
    ego_marker.pose.orientation.z = q.z();
    ego_marker.pose.orientation.w = q.w();
    
    ego_marker.scale.x = 1.0;
    ego_marker.scale.y = 0.3;
    ego_marker.scale.z = 0.3;
    ego_marker.color.r = 0.0;
    ego_marker.color.g = 0.0;
    ego_marker.color.b = 1.0;
    ego_marker.color.a = 1.0;
    
    marker_array.markers.push_back(ego_marker);
    
    vis_pub_->publish(marker_array);
}

void GlobalPathFollower::publishGlobalWaypoints() {
    if (!waypoint_loader_ || waypoint_loader_->getGlobalWaypoints().empty()) {
        RCLCPP_WARN(this->get_logger(), "No waypoints to publish");
        return;
    }
    
    global_path_follower::msg::WpntArray wpnt_array_msg;
    wpnt_array_msg.header.stamp = this->get_clock()->now();
    wpnt_array_msg.header.frame_id = map_frame_;
    
    const auto& waypoints = waypoint_loader_->getGlobalWaypoints();
    
    for (const auto& wp : waypoints) {
        global_path_follower::msg::Wpnt wpnt_msg;
        wpnt_msg.s_m = wp.s_m;
        wpnt_msg.x_m = wp.x_m;
        wpnt_msg.y_m = wp.y_m;
        wpnt_msg.d_right = wp.d_right;
        wpnt_msg.d_left = wp.d_left;
        wpnt_msg.psi_rad = wp.psi_rad;
        wpnt_msg.kappa_radpm = wp.kappa_radpm;
        wpnt_msg.vx_mps = wp.vx_mps;
        wpnt_msg.ax_mps2 = wp.ax_mps2;
        
        wpnt_array_msg.waypoints.push_back(wpnt_msg);
    }
    
    global_waypoints_pub_->publish(wpnt_array_msg);
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                         "Published %zu global waypoints", waypoints.size());
}

} // namespace global_path_follower
