#include "spline_planner_dh/spline_planner.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

SplinePlanner::SplinePlanner() : Node("spline_planner_node"),
    cur_s_(0.0), cur_d_(0.0), cur_vs_(0.0),
    lookahead_(10.0), evasion_dist_(0.65), obs_traj_thresh_(0.3), spline_bound_mindist_(0.2),
    gb_max_s_(0.0), gb_max_idx_(0), gb_vmax_(0.0),
    pre_apex_0_(-4.0), pre_apex_1_(-3.0), pre_apex_2_(-1.5),
    post_apex_0_(2.0), post_apex_1_(3.0), post_apex_2_(4.0),
    last_ot_side_(""), last_switch_time_(this->get_clock()->now()) {
    
    // Initialize subscribers
    obstacles_sub_ = this->create_subscription<spline_planner_dh::msg::ObstacleArray>(
        "/perception/obstacles", 10,
        std::bind(&SplinePlanner::obstacles_callback, this, std::placeholders::_1));
    
    ego_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/ego_racecar/odom", 10,
        std::bind(&SplinePlanner::ego_odom_callback, this, std::placeholders::_1));
    
    global_waypoints_sub_ = this->create_subscription<spline_planner_dh::msg::WpntArray>(
        "/global_waypoints", 10,
        std::bind(&SplinePlanner::global_waypoints_callback, this, std::placeholders::_1));
    
    // Initialize publishers
    local_trajectory_pub_ = this->create_publisher<spline_planner_dh::msg::OTWpntArray>(
        "/local_trajectory", 10);
    
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planner/avoidance/markers", 10);
    
    closest_obs_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/planner/avoidance/considered_OBS", 10);
    
    reference_path_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planner/reference_path", 10);
    
    // Wait for required messages
    RCLCPP_INFO(this->get_logger(), "Waiting for required messages...");
    if (!wait_for_messages()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to receive required messages");
        return;
    }
    
    // Initialize Frenet converter
    initialize_converter();
    
    // Create timer for main processing loop
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50), // 20 Hz
        std::bind(&SplinePlanner::timer_callback, this));
    
    RCLCPP_INFO(this->get_logger(), "Spline planner initialized successfully");
}

void SplinePlanner::obstacles_callback(const spline_planner_dh::msg::ObstacleArray::SharedPtr msg) {
    obstacles_ = msg;
}

void SplinePlanner::ego_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    ego_odom_ = msg;
    
    // Convert ego position to Frenet coordinates
    if (frenet_converter_) {
        auto frenet_pos = frenet_converter_->cartesian_to_frenet(
            msg->pose.pose.position.x, msg->pose.pose.position.y);
        cur_s_ = frenet_pos.first;
        cur_d_ = frenet_pos.second;
        cur_vs_ = std::sqrt(
            msg->twist.twist.linear.x * msg->twist.twist.linear.x +
            msg->twist.twist.linear.y * msg->twist.twist.linear.y);
    }
}

void SplinePlanner::global_waypoints_callback(const spline_planner_dh::msg::WpntArray::SharedPtr msg) {
    global_waypoints_ = msg;
    
    if (!msg->wpnts.empty()) {
        gb_max_idx_ = msg->wpnts.back().id;
        gb_max_s_ = msg->wpnts.back().s_m;
        
        // Calculate maximum velocity
        gb_vmax_ = 0.0;
        for (const auto& waypoint : msg->wpnts) {
            if (waypoint.vx_mps > gb_vmax_) {
                gb_vmax_ = waypoint.vx_mps;
            }
        }
        
        RCLCPP_INFO(this->get_logger(), 
            "Received global waypoints: %zu points, max_s: %.2f, max_v: %.2f",
            msg->wpnts.size(), gb_max_s_, gb_vmax_);
    }
}

void SplinePlanner::timer_callback() {
    process_spline_planning();
}

void SplinePlanner::process_spline_planning() {
    // Check for minimum required data (ego_odom and global_waypoints)
    if (!ego_odom_ || !global_waypoints_ || !frenet_converter_) {
        return;
    }
    
    // Always publish reference path when we have waypoints
    publish_reference_path();
    
    spline_planner_dh::msg::OTWpntArray trajectory_msg;
    visualization_msgs::msg::MarkerArray markers_msg;
    
    // Only process obstacles if they exist
    if (obstacles_) {
        // Convert obstacles to Frenet coordinates
        auto frenet_obstacles = convert_obstacles_to_frenet(*obstacles_);
        
        // Filter obstacles that are close and relevant
        auto close_obstacles = filter_close_obstacles(frenet_obstacles);
        
        if (!close_obstacles.empty()) {
            // Find closest obstacle
            auto closest_obstacle = *std::min_element(close_obstacles.begin(), close_obstacles.end(),
                [this](const ObstacleFrenet& a, const ObstacleFrenet& b) {
                    double dist_a = std::fmod(a.s_center - cur_s_ + gb_max_s_, gb_max_s_);
                    double dist_b = std::fmod(b.s_center - cur_s_ + gb_max_s_, gb_max_s_);
                    return dist_a < dist_b;
                });
            
            // Generate evasion trajectory
            trajectory_msg = generate_evasion_trajectory(closest_obstacle);
            
            // Create visualization markers
            markers_msg = create_trajectory_markers(trajectory_msg);
            
            // Publish closest obstacle marker
            auto obs_marker = create_obstacle_marker(closest_obstacle);
            closest_obs_pub_->publish(obs_marker);
            
            RCLCPP_DEBUG(this->get_logger(), 
                "Generated evasion trajectory with %zu waypoints", 
                trajectory_msg.waypoints.size());
        } else {
            // Obstacles exist but none are close - following reference path
            trajectory_msg.header.stamp = this->get_clock()->now();
            trajectory_msg.header.frame_id = "map";
            
            // Create status marker
            visualization_msgs::msg::Marker status_marker;
            status_marker.header.frame_id = "map";
            status_marker.header.stamp = this->get_clock()->now();
            status_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            status_marker.action = visualization_msgs::msg::Marker::ADD;
            status_marker.id = 999;
            status_marker.ns = "planner_status";
            
            status_marker.pose.position.x = ego_odom_->pose.pose.position.x + 2.0;
            status_marker.pose.position.y = ego_odom_->pose.pose.position.y + 1.0;
            status_marker.pose.position.z = 1.0;
            status_marker.pose.orientation.w = 1.0;
            
            status_marker.scale.z = 0.5;
            status_marker.color.a = 0.8;
            status_marker.color.r = 0.0;
            status_marker.color.g = 1.0;
            status_marker.color.b = 0.0;
            
            status_marker.text = "SPLINE PLANNER: NO CLOSE OBSTACLES";
            
            markers_msg.markers.push_back(status_marker);
        }
    } else {
        // No obstacle data available - still show planner is active
        trajectory_msg.header.stamp = this->get_clock()->now();
        trajectory_msg.header.frame_id = "map";
        
        // Create status marker
        visualization_msgs::msg::Marker status_marker;
        status_marker.header.frame_id = "map";
        status_marker.header.stamp = this->get_clock()->now();
        status_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        status_marker.action = visualization_msgs::msg::Marker::ADD;
        status_marker.id = 999;
        status_marker.ns = "planner_status";
        
        status_marker.pose.position.x = ego_odom_->pose.pose.position.x + 2.0;
        status_marker.pose.position.y = ego_odom_->pose.pose.position.y + 1.0;
        status_marker.pose.position.z = 1.0;
        status_marker.pose.orientation.w = 1.0;
        
        status_marker.scale.z = 0.5;
        status_marker.color.a = 0.8;
        status_marker.color.r = 1.0;
        status_marker.color.g = 0.5;
        status_marker.color.b = 0.0;  // Orange color to indicate no obstacle data
        
        status_marker.text = "SPLINE PLANNER: WAITING FOR OBSTACLE DATA";
        
        markers_msg.markers.push_back(status_marker);
        
        RCLCPP_DEBUG(this->get_logger(), "No obstacle data - following reference path");
    }
    
    // Publish results
    local_trajectory_pub_->publish(trajectory_msg);
    markers_pub_->publish(markers_msg);
}

bool SplinePlanner::wait_for_messages() {
    rclcpp::Rate rate(10); // 10 Hz
    int timeout_count = 0;
    const int max_timeout = 100; // 10 seconds
    
    while (rclcpp::ok() && timeout_count < max_timeout) {
        rclcpp::spin_some(this->get_node_base_interface());
        
        if (ego_odom_ && global_waypoints_) {
            RCLCPP_INFO(this->get_logger(), "All required messages received");
            return true;
        }
        
        rate.sleep();
        timeout_count++;
    }
    
    return false;
}

void SplinePlanner::initialize_converter() {
    if (!global_waypoints_ || global_waypoints_->wpnts.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Cannot initialize converter: no global waypoints");
        return;
    }
    
    std::vector<double> x, y, psi;
    for (const auto& waypoint : global_waypoints_->wpnts) {
        x.push_back(waypoint.x_m);
        y.push_back(waypoint.y_m);
        psi.push_back(waypoint.psi_rad);
    }
    
    frenet_converter_ = std::make_unique<FrenetConverter>(x, y, psi);
    RCLCPP_INFO(this->get_logger(), "Frenet converter initialized");
}

std::vector<ObstacleFrenet> SplinePlanner::convert_obstacles_to_frenet(
    const spline_planner_dh::msg::ObstacleArray& obstacles) {
    
    std::vector<ObstacleFrenet> frenet_obstacles;
    
    for (const auto& obs : obstacles.obstacles) {
        ObstacleFrenet frenet_obs;
        frenet_obs.id = obs.id;
        
        // Convert all obstacle points to Frenet coordinates
        auto center_frenet = frenet_converter_->cartesian_to_frenet(obs.x_center, obs.y_center);
        auto left_frenet = frenet_converter_->cartesian_to_frenet(obs.x_left, obs.y_left);
        auto right_frenet = frenet_converter_->cartesian_to_frenet(obs.x_right, obs.y_right);
        auto start_frenet = frenet_converter_->cartesian_to_frenet(obs.x_start, obs.y_start);
        auto end_frenet = frenet_converter_->cartesian_to_frenet(obs.x_end, obs.y_end);
        
        frenet_obs.s_center = center_frenet.first;
        frenet_obs.d_center = center_frenet.second;
        frenet_obs.s_left = left_frenet.first;
        frenet_obs.d_left = left_frenet.second;
        frenet_obs.s_right = right_frenet.first;
        frenet_obs.d_right = right_frenet.second;
        frenet_obs.s_start = start_frenet.first;
        frenet_obs.d_start = start_frenet.second;
        frenet_obs.s_end = end_frenet.first;
        frenet_obs.d_end = end_frenet.second;
        
        frenet_obstacles.push_back(frenet_obs);
    }
    
    return frenet_obstacles;
}

std::vector<ObstacleFrenet> SplinePlanner::filter_close_obstacles(
    const std::vector<ObstacleFrenet>& obstacles) {
    
    std::vector<ObstacleFrenet> close_obstacles;
    
    for (const auto& obs : obstacles) {
        // Check if obstacle is within trajectory threshold
        if (std::abs(obs.d_center) < obs_traj_thresh_) {
            // Check if obstacle is within lookahead distance
            double dist_in_front = std::fmod(obs.s_center - cur_s_ + gb_max_s_, gb_max_s_);
            if (dist_in_front < lookahead_) {
                close_obstacles.push_back(obs);
            }
        }
    }
    
    return close_obstacles;
}

std::pair<std::string, double> SplinePlanner::calculate_more_space(const ObstacleFrenet& obstacle) {
    // Find corresponding global waypoint
    double wpnt_dist = gb_max_s_ / gb_max_idx_;
    int gb_idx = static_cast<int>((obstacle.s_center / wpnt_dist)) % gb_max_idx_;
    
    if (gb_idx >= static_cast<int>(global_waypoints_->wpnts.size())) {
        gb_idx = global_waypoints_->wpnts.size() - 1;
    }
    
    const auto& waypoint = global_waypoints_->wpnts[gb_idx];
    
    double left_gap = std::abs(waypoint.d_left - obstacle.d_left);
    double right_gap = std::abs(waypoint.d_right - obstacle.d_right);
    double min_space = evasion_dist_ + spline_bound_mindist_;
    
    if (right_gap > min_space && left_gap < min_space) {
        // Go right
        double d_apex_right = obstacle.d_right - evasion_dist_;
        if (d_apex_right > 0) d_apex_right = 0;
        return std::make_pair("right", d_apex_right);
    } else if (left_gap > min_space && right_gap < min_space) {
        // Go left
        double d_apex_left = obstacle.d_left + evasion_dist_;
        if (d_apex_left < 0) d_apex_left = 0;
        return std::make_pair("left", d_apex_left);
    } else {
        // Both sides have space - choose the one closer to raceline
        double candidate_d_apex_left = obstacle.d_left + evasion_dist_;
        double candidate_d_apex_right = obstacle.d_right - evasion_dist_;
        
        if (std::abs(candidate_d_apex_left) <= std::abs(candidate_d_apex_right)) {
            if (candidate_d_apex_left < 0) candidate_d_apex_left = 0;
            return std::make_pair("left", candidate_d_apex_left);
        } else {
            if (candidate_d_apex_right > 0) candidate_d_apex_right = 0;
            return std::make_pair("right", candidate_d_apex_right);
        }
    }
}

spline_planner_dh::msg::OTWpntArray SplinePlanner::generate_evasion_trajectory(
    const ObstacleFrenet& closest_obstacle) {
    
    spline_planner_dh::msg::OTWpntArray trajectory;
    trajectory.header.stamp = this->get_clock()->now();
    trajectory.header.frame_id = "map";
    
    // Calculate apex position
    double s_apex = (closest_obstacle.s_end + closest_obstacle.s_start) / 2.0;
    if (closest_obstacle.s_end < closest_obstacle.s_start) {
        s_apex = (closest_obstacle.s_end + gb_max_s_ + closest_obstacle.s_start) / 2.0;
    }
    
    // Determine which side has more space
    auto [more_space, d_apex] = calculate_more_space(closest_obstacle);
    
    // Create spline points
    std::vector<double> spline_params = {
        pre_apex_0_, pre_apex_1_, pre_apex_2_, 0.0,
        post_apex_0_, post_apex_1_, post_apex_2_
    };
    
    std::vector<double> evasion_s, evasion_d;
    for (size_t i = 0; i < spline_params.size(); ++i) {
        double dst = spline_params[i];
        // Scale distance based on current speed
        dst *= std::clamp(1.0 + cur_vs_ / gb_vmax_, 1.0, 1.5);
        
        double si = s_apex + dst;
        double di = (dst == 0.0) ? d_apex : 0.0;
        
        evasion_s.push_back(si);
        evasion_d.push_back(di);
    }
    
    // Create spline interpolator
    SplineInterpolator spline(evasion_s, evasion_d);
    
    // Generate trajectory points
    double spline_resolution = 0.1;
    std::vector<double> trajectory_s;
    for (double s = evasion_s[0]; s <= evasion_s.back(); s += spline_resolution) {
        trajectory_s.push_back(s);
    }
    
    auto trajectory_d = spline.interpolate(trajectory_s);
    
    // Clip d values to stay within bounds
    for (size_t i = 0; i < trajectory_d.size(); ++i) {
        if (d_apex < 0) {
            trajectory_d[i] = std::clamp(trajectory_d[i], d_apex, 0.0);
        } else {
            trajectory_d[i] = std::clamp(trajectory_d[i], 0.0, d_apex);
        }
    }
    
    // Convert to Cartesian and create waypoints
    for (size_t i = 0; i < trajectory_s.size(); ++i) {
        double s = std::fmod(trajectory_s[i], gb_max_s_);
        if (s < 0) s += gb_max_s_;
        
        auto cartesian = frenet_converter_->frenet_to_cartesian(s, trajectory_d[i]);
        
        // Get velocity from global waypoints
        double wpnt_dist = gb_max_s_ / gb_max_idx_;
        int gb_idx = static_cast<int>((s / wpnt_dist)) % gb_max_idx_;
        if (gb_idx >= static_cast<int>(global_waypoints_->wpnts.size())) {
            gb_idx = global_waypoints_->wpnts.size() - 1;
        }
        
        double velocity = global_waypoints_->wpnts[gb_idx].vx_mps;
        
        spline_planner_dh::msg::OTWpnt waypoint;
        waypoint.id = static_cast<int>(trajectory.waypoints.size());
        waypoint.x_m = cartesian.first;
        waypoint.y_m = cartesian.second;
        waypoint.s_m = s;
        waypoint.d_m = trajectory_d[i];
        waypoint.vx_mps = velocity;
        
        trajectory.waypoints.push_back(waypoint);
    }
    
    // Set trajectory metadata
    trajectory.ot_side = more_space;
    trajectory.side_switch = (last_ot_side_ != more_space);
    trajectory.last_switch_time = last_switch_time_;
    
    if (trajectory.side_switch) {
        last_switch_time_ = this->get_clock()->now();
    }
    last_ot_side_ = more_space;
    
    return trajectory;
}

visualization_msgs::msg::MarkerArray SplinePlanner::create_trajectory_markers(
    const spline_planner_dh::msg::OTWpntArray& trajectory) {
    
    visualization_msgs::msg::MarkerArray markers;
    
    if (trajectory.waypoints.empty()) {
        return markers;
    }
    
    // Create line strip for the trajectory
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = "map";
    line_marker.header.stamp = this->get_clock()->now();
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.id = 0;
    line_marker.ns = "local_trajectory_line";
    
    line_marker.scale.x = 0.1;  // Line width
    line_marker.color.a = 1.0;
    line_marker.color.r = 1.0;  // Red
    line_marker.color.g = 0.0;
    line_marker.color.b = 1.0;  // Magenta line
    
    // Add all trajectory points to line strip
    for (const auto& waypoint : trajectory.waypoints) {
        geometry_msgs::msg::Point point;
        point.x = waypoint.x_m;
        point.y = waypoint.y_m;
        point.z = 0.02;
        line_marker.points.push_back(point);
    }
    
    markers.markers.push_back(line_marker);
    
    // Add individual waypoint markers (cylinders) - show every 3rd point to avoid clutter
    for (size_t i = 0; i < trajectory.waypoints.size(); i += 3) {
        const auto& waypoint = trajectory.waypoints[i];
        
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.id = static_cast<int>(i + 1);
        marker.ns = "local_trajectory_points";
        
        marker.pose.position.x = waypoint.x_m;
        marker.pose.position.y = waypoint.y_m;
        marker.pose.position.z = waypoint.vx_mps / (gb_vmax_ / 2.0);
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = 0.15;
        marker.scale.y = 0.15;
        marker.scale.z = std::max(0.1, waypoint.vx_mps / gb_vmax_);
        
        marker.color.a = 0.8;
        marker.color.r = 0.75;
        marker.color.g = 0.0;
        marker.color.b = 0.75;
        
        markers.markers.push_back(marker);
    }
    
    return markers;
}

visualization_msgs::msg::Marker SplinePlanner::create_obstacle_marker(const ObstacleFrenet& obstacle) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->get_clock()->now();
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    
    // Convert obstacle center back to Cartesian
    auto cartesian = frenet_converter_->frenet_to_cartesian(obstacle.s_center, obstacle.d_center);
    
    marker.pose.position.x = cartesian.first;
    marker.pose.position.y = cartesian.second;
    marker.pose.position.z = 0.01;
    marker.pose.orientation.w = 1.0;
    
    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 0.5;
    
    marker.color.a = 0.8;
    marker.color.r = 1.0;
    marker.color.g = 0.65;
    marker.color.b = 0.65;
    
    return marker;
}

void SplinePlanner::publish_reference_path() {
    if (!global_waypoints_ || global_waypoints_->wpnts.empty()) {
        return;
    }
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // Create reference line (raceline)
    visualization_msgs::msg::Marker raceline_marker;
    raceline_marker.header.frame_id = "map";
    raceline_marker.header.stamp = this->get_clock()->now();
    raceline_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    raceline_marker.action = visualization_msgs::msg::Marker::ADD;
    raceline_marker.id = 0;
    raceline_marker.ns = "reference_path";
    
    raceline_marker.scale.x = 0.05;  // Thin line
    raceline_marker.color.a = 0.6;
    raceline_marker.color.r = 1.0;
    raceline_marker.color.g = 1.0;
    raceline_marker.color.b = 0.0;  // Yellow raceline
    
    // Add raceline points
    for (const auto& waypoint : global_waypoints_->wpnts) {
        geometry_msgs::msg::Point point;
        point.x = waypoint.x_m;
        point.y = waypoint.y_m;
        point.z = 0.01;
        raceline_marker.points.push_back(point);
    }
    
    // Close the loop
    if (!global_waypoints_->wpnts.empty()) {
        geometry_msgs::msg::Point first_point;
        first_point.x = global_waypoints_->wpnts[0].x_m;
        first_point.y = global_waypoints_->wpnts[0].y_m;
        first_point.z = 0.01;
        raceline_marker.points.push_back(first_point);
    }
    
    marker_array.markers.push_back(raceline_marker);
    
    // Create track boundaries
    for (int side = 0; side < 2; ++side) {
        visualization_msgs::msg::Marker boundary_marker;
        boundary_marker.header.frame_id = "map";
        boundary_marker.header.stamp = this->get_clock()->now();
        boundary_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        boundary_marker.action = visualization_msgs::msg::Marker::ADD;
        boundary_marker.id = side + 1;
        boundary_marker.ns = "track_boundaries";
        
        boundary_marker.scale.x = 0.03;
        boundary_marker.color.a = 0.4;
        boundary_marker.color.r = 0.8;
        boundary_marker.color.g = 0.8;
        boundary_marker.color.b = 0.8;  // Gray boundaries
        
        for (const auto& waypoint : global_waypoints_->wpnts) {
            // Calculate boundary points using Frenet coordinates
            double d_offset = (side == 0) ? waypoint.d_left : waypoint.d_right;
            auto boundary_point = frenet_converter_->frenet_to_cartesian(waypoint.s_m, d_offset);
            
            geometry_msgs::msg::Point point;
            point.x = boundary_point.first;
            point.y = boundary_point.second;
            point.z = 0.005;
            boundary_marker.points.push_back(point);
        }
        
        // Close the loop
        if (!global_waypoints_->wpnts.empty()) {
            const auto& first_waypoint = global_waypoints_->wpnts[0];
            double d_offset = (side == 0) ? first_waypoint.d_left : first_waypoint.d_right;
            auto boundary_point = frenet_converter_->frenet_to_cartesian(first_waypoint.s_m, d_offset);
            
            geometry_msgs::msg::Point first_point;
            first_point.x = boundary_point.first;
            first_point.y = boundary_point.second;
            first_point.z = 0.005;
            boundary_marker.points.push_back(first_point);
        }
        
        marker_array.markers.push_back(boundary_marker);
    }
    
    reference_path_pub_->publish(marker_array);
}
