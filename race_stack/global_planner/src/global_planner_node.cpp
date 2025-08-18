// Lee Minwon

#include "global_planner_node.hpp"

GlobalPlanner::GlobalPlanner(const std::string &node_name, const rclcpp::NodeOptions &options)
    : Node(node_name, options)  {
    
    // QoS init
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
    
    // Parameters
    this->declare_parameter("global_planner/loop_rate_hz", 40.0);
    this->declare_parameter("global_planner/trajectory_csv_file", "redbull_1.csv");

    ProcessParams();
    RCLCPP_INFO(this->get_logger(), "loop_rate_hz: %f", cfg_.loop_rate_hz);
    RCLCPP_INFO(this->get_logger(), "trajectory_csv_file: %s", cfg_.trajectory_csv_file.c_str());
    
    // Load global trajectory from CSV
    if (LoadGlobalTrajectoryFromCSV(cfg_.trajectory_csv_file)) {
        RCLCPP_INFO(this->get_logger(), "Successfully loaded %zu waypoints from CSV", global_trajectory_.size());
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to load trajectory from CSV file");
    }
    
    // Subscriber init
    s_car_state_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/car_state/odom", qos_profile, std::bind(&GlobalPlanner::CallbackCarStateOdom, this, std::placeholders::_1));
    // s_ego_racecar_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
    //     "/ego_racecar/odom", qos_profile, std::bind(&GlobalPlanner::CallbackEgoRaceCarOdom, this, std::placeholders::_1));
  
    // Publisher init
    p_global_waypoints_ = this->create_publisher<f110_msgs::msg::WpntArray>(
        "/global_waypoints", qos_profile);
    // p_local_waypoints_ = this->create_publisher<f110_msgs::msg::WpntArray>(
    //     "/local_waypoints", qos_profile);
    p_car_state_odom_ = this->create_publisher<nav_msgs::msg::Odometry>(  // frenet 으로 이름 바꾸기
        "/car_state/frenet/odom", qos_profile);
    
    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); });
}

GlobalPlanner::~GlobalPlanner() {}

void GlobalPlanner::ProcessParams() {
    this->get_parameter("global_planner/loop_rate_hz", cfg_.loop_rate_hz);
    this->get_parameter("global_planner/trajectory_csv_file", cfg_.trajectory_csv_file);
}

void GlobalPlanner::Run() {
    auto current_time = this->now();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "Running ...");
    ProcessParams();

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get ego_racecar_odom and extract all required data
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    if (b_is_car_state_odom_ == false) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 1000, "Waiting for car state odom...");
        return;
    }

    // if (b_is_ego_racecar_odom_ == false) {
    //     RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 1000, "Waiting for ego racecar odom...");
    //     return;
    // }

    mutex_car_state_odom_.lock();
    auto car_state_odom = i_car_state_odom_;
    mutex_car_state_odom_.unlock(); 

    // mutex_ego_racecar_odom_.lock();
    // auto ego_racecar_odom = i_ego_racecar_odom_;
    // mutex_ego_racecar_odom_.unlock();

    // Extract position and velocities from ego_racecar_odom
    double vehicle_x = car_state_odom.pose.pose.position.x;
    double vehicle_y = car_state_odom.pose.pose.position.y;
    double vx = car_state_odom.twist.twist.linear.x;
    double vy = car_state_odom.twist.twist.linear.y;

    // double vehicle_x = ego_racecar_odom.pose.pose.position.x;
    // double vehicle_y = ego_racecar_odom.pose.pose.position.y;
    // double vx = ego_racecar_odom.twist.twist.linear.x;
    // double vy = ego_racecar_odom.twist.twist.linear.y;
    
    // Extract vehicle heading from quaternion
    auto q = car_state_odom.pose.pose.orientation;
    // auto q = ego_racecar_odom.pose.pose.orientation;
    double vehicle_theta = atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z));

    // Convert position to Frenet coordinates
    FrenetCoordinates frenet_coords = CartesianToFrenet(vehicle_x, vehicle_y);
    
    // Convert velocities to Frenet coordinates
    auto frenet_vel = CartesianVelocityToFrenet(vx, vy, vehicle_theta);
    double vs = frenet_vel.first;   // longitudinal velocity
    double vd = frenet_vel.second;  // lateral velocity
    
    // Publish Frenet coordinates with velocities
    PublishFrenetOdometry(car_state_odom.header.stamp, frenet_coords, vs, vd);
    // PublishFrenetOdometry(ego_racecar_odom.header.stamp, frenet_coords, vs, vd);

    // Publish local waypoints (next 80 waypoints from current position)
    // PublishLocalWaypoints(frenet_coords);

    // // Publish global waypoints
    PublishGlobalWaypoints();
}

bool GlobalPlanner::LoadGlobalTrajectoryFromCSV(const std::string& csv_filename) {
    try {
        std::string csv_path = csv_filename;
        
        RCLCPP_INFO(this->get_logger(), "Loading trajectory from: %s", csv_path.c_str());
        
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open CSV file: %s", csv_path.c_str());
            return false;
        }
        
        global_trajectory_.clear();
        std::string line;
        
        // Skip header lines that start with '#'
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            
            // Parse CSV line (s_m; x_m; y_m; d_right; d_left; psi_rad; kappa_radpm; vx_mps; ax_mps2)
            while (std::getline(ss, cell, ';')) {
                try {
                    // Trim whitespace
                    cell.erase(0, cell.find_first_not_of(" \t"));
                    cell.erase(cell.find_last_not_of(" \t") + 1);
                    double value = std::stod(cell);
                    row.push_back(value);
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Error parsing CSV value: %s", cell.c_str());
                    return false;
                }
            }
            
            // Expect 9 columns: s_m, x_m, y_m, d_right, d_left, psi_rad, kappa_radpm, vx_mps, ax_mps2
            if (row.size() == 9) {
                global_trajectory_.push_back(row);
            } else {
                RCLCPP_WARN(this->get_logger(), "Skipping invalid CSV line with %zu columns", row.size());
            }
        }
        
        file.close();
        
        if (global_trajectory_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No valid waypoints found in CSV file");
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception while loading CSV: %s", e.what());
        return false;
    }
}

void GlobalPlanner::PublishGlobalWaypoints() {
    if (global_trajectory_.empty()) {
        return;
    }
    
    f110_msgs::msg::WpntArray waypoint_array;
    waypoint_array.header.stamp = this->now();
    waypoint_array.header.frame_id = "map";
    
    for (size_t i = 0; i < global_trajectory_.size(); ++i) {
        const auto& row = global_trajectory_[i];
        
        f110_msgs::msg::Wpnt waypoint;
        waypoint.id = static_cast<int32_t>(i);  // Sequential ID starting from 0
        waypoint.s_m = row[0];     // s_m
        waypoint.d_m = 0.0;        // d_m (set to 0 as requested)
        waypoint.x_m = row[1];     // x_m
        waypoint.y_m = row[2];     // y_m
        waypoint.d_right = row[3]; // d_right
        waypoint.d_left = row[4];  // d_left
        waypoint.psi_rad = row[5]; // psi_rad
        waypoint.kappa_radpm = row[6]; // kappa_radpm
        waypoint.vx_mps = row[7];  // vx_mps
        waypoint.ax_mps2 = row[8]; // ax_mps2
        
        waypoint_array.wpnts.push_back(waypoint);
    }
    
    p_global_waypoints_->publish(waypoint_array);
}

int GlobalPlanner::FindClosestWaypoint(double vehicle_x, double vehicle_y) {
    if (global_trajectory_.empty()) return -1;
    
    // For circular tracks, expand search range when near boundaries
    int search_range = 20;  // ±20개 waypoint 검색
    int trajectory_size = (int)global_trajectory_.size();
    
    // If we're near the end of trajectory, also check the beginning
    bool check_wrap_around = (last_closest_index_ > trajectory_size * 0.8) || 
                            (last_closest_index_ < trajectory_size * 0.2);
    
    double min_distance_sq = std::numeric_limits<double>::max();
    int closest_index = last_closest_index_;
    
    if (check_wrap_around) {
        // Search entire trajectory when near boundaries to handle wrap-around
        for (int i = 0; i < trajectory_size; ++i) {
            double wp_x = global_trajectory_[i][1];  // x_m
            double wp_y = global_trajectory_[i][2];  // y_m
            
            double dx = vehicle_x - wp_x;
            double dy = vehicle_y - wp_y;
            double distance_sq = dx * dx + dy * dy;
            
            if (distance_sq < min_distance_sq) {
                min_distance_sq = distance_sq;
                closest_index = i;
            }
        }
        RCLCPP_DEBUG(this->get_logger(), "Wrap-around search used, found closest: %d", closest_index);
    } else {
        // Normal local search
        int start_idx = std::max(0, last_closest_index_ - search_range);
        int end_idx = std::min(trajectory_size - 1, last_closest_index_ + search_range);
        
        for (int i = start_idx; i <= end_idx; ++i) {
            double wp_x = global_trajectory_[i][1];  // x_m
            double wp_y = global_trajectory_[i][2];  // y_m
            
            double dx = vehicle_x - wp_x;
            double dy = vehicle_y - wp_y;
            double distance_sq = dx * dx + dy * dy;
            
            if (distance_sq < min_distance_sq) {
                min_distance_sq = distance_sq;
                closest_index = i;
            }
        }
        RCLCPP_DEBUG(this->get_logger(), 
            "Local search range: [%d, %d], Found closest: %d", 
            start_idx, end_idx, closest_index);
    }
    
    // 다음 검색을 위해 저장
    last_closest_index_ = closest_index;
    
    return closest_index;
}

GlobalPlanner::FrenetCoordinates GlobalPlanner::CartesianToFrenet(double x, double y) {
    FrenetCoordinates frenet;
    frenet.s = 0.0;
    frenet.d = 0.0;
    
    if (global_trajectory_.empty()) {
        RCLCPP_WARN(this->get_logger(), "Global trajectory is empty, cannot convert to Frenet");
        return frenet;
    }
    
    // Find closest waypoint
    int closest_idx = FindClosestWaypoint(x, y);
    
    if (closest_idx < 0) {
        RCLCPP_WARN(this->get_logger(), "Could not find closest waypoint");
        return frenet;
    }
    
    // Current waypoint data
    double wp_x = global_trajectory_[closest_idx][1];    // x_m
    double wp_y = global_trajectory_[closest_idx][2];    // y_m
    double wp_s = global_trajectory_[closest_idx][0];    // s_m
    double wp_psi = global_trajectory_[closest_idx][5];  // psi_rad
    
    // Calculate track direction using adjacent waypoints for more accuracy
    double track_psi = wp_psi;
    
    // Use adjacent waypoints to calculate actual track direction
    if (closest_idx > 0 && closest_idx < (int)global_trajectory_.size() - 1) {
        double prev_x = global_trajectory_[closest_idx - 1][1];
        double prev_y = global_trajectory_[closest_idx - 1][2];
        double next_x = global_trajectory_[closest_idx + 1][1];
        double next_y = global_trajectory_[closest_idx + 1][2];
        
        // Calculate track direction from trajectory points
        track_psi = atan2(next_y - prev_y, next_x - prev_x);
        
        RCLCPP_DEBUG(this->get_logger(), 
            "Track direction: stored_psi=%.3f, calculated_psi=%.3f", wp_psi, track_psi);
    }
    
    // Vector from waypoint to vehicle
    double dx = x - wp_x;
    double dy = y - wp_y;
    
    // Project onto track tangent to get s offset
    double ds = dx * cos(track_psi) + dy * sin(track_psi);
    
    // Project onto track normal to get d offset
    // Convention: positive d = left side of track (when looking in direction of travel)
    double dd = -dx * sin(track_psi) + dy * cos(track_psi);
    
    // Final Frenet coordinates
    frenet.s = wp_s + ds;
    frenet.d = dd;
    
    // Handle track wrapping for closed circuits
    if (!global_trajectory_.empty()) {
        double max_s = global_trajectory_.back()[0];  // Last waypoint's s value
        
        // If we're near the end of the track and close to the beginning
        if (frenet.s > max_s * 0.9) {  // Near end of track
            // Check if we're actually closer to the beginning
            double first_wp_x = global_trajectory_[0][1];
            double first_wp_y = global_trajectory_[0][2];
            double dist_to_start = sqrt((x - first_wp_x) * (x - first_wp_x) + 
                                       (y - first_wp_y) * (y - first_wp_y));
            
            double current_wp_x = global_trajectory_[closest_idx][1];
            double current_wp_y = global_trajectory_[closest_idx][2];
            double dist_to_current = sqrt((x - current_wp_x) * (x - current_wp_x) + 
                                         (y - current_wp_y) * (y - current_wp_y));
            
            // If closer to start, wrap around
            if (dist_to_start < dist_to_current * 1.5 && closest_idx > global_trajectory_.size() * 0.8) {
                frenet.s = frenet.s - max_s;  // Wrap to beginning
                RCLCPP_DEBUG(this->get_logger(), "Track wrapping applied: s wrapped from %.3f to %.3f", 
                            wp_s + ds, frenet.s);
            }
        }
        
        // Ensure s is always positive
        if (frenet.s < 0.0) {
            frenet.s = 0.0;
        }
    }
    
    // Detailed debugging output
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000,
        "[Frenet] XY(%.2f,%.2f)  ClosestWP[%d]:XY(%.2f,%.2f)  TrackPsi:%.3f  dX:%.2f dY:%.2f  ds:%.2f dd:%.2f  =>  s:%.2f d:%.2f",
        x, y, closest_idx, wp_x, wp_y, track_psi, dx, dy, ds, dd, frenet.s, frenet.d);
    
    return frenet;
}

std::pair<double, double> GlobalPlanner::CartesianVelocityToFrenet(double vx, double vy, double vehicle_theta) {
    // base_link 기준 속도를 map 기준 속도로 변환
    // vx, vy: base_link 기준 (x: 전방, y: 좌측)
    // vehicle_theta: map 기준 차량 헤딩 (x축 기준 CCW)
    if (global_trajectory_.empty()) {
        RCLCPP_WARN(this->get_logger(), "Global trajectory is empty, cannot convert velocity to Frenet");
        return {0.0, 0.0};
    }

    int closest_idx = last_closest_index_;
    if (closest_idx < 0 || closest_idx >= (int)global_trajectory_.size()) {
        RCLCPP_WARN(this->get_logger(), "Invalid closest waypoint index for velocity conversion");
        return {0.0, 0.0};
    }

    // 트랙 방향(psi, map 기준)
    double track_psi = global_trajectory_[closest_idx][5];

    // 1. base_link -> map 변환
    double vx_map = vx * cos(vehicle_theta) - vy * sin(vehicle_theta);
    double vy_map = vx * sin(vehicle_theta) + vy * cos(vehicle_theta);

    // 2. map -> frenet 변환
    // vd: 트랙 좌측이 +인 횡방향, vs: 트랙 종방향(Reference Line 방향)
    double vd = vx_map * cos(track_psi) + vy_map * sin(track_psi);
    double vs = -vx_map * sin(track_psi) + vy_map * cos(track_psi);

    RCLCPP_DEBUG(this->get_logger(),
        "Velocity conversion: BaseLink(vx=%.3f, vy=%.3f, theta=%.3f) -> Map(vx=%.3f, vy=%.3f) -> Track_psi=%.3f -> Frenet(vs=%.3f, vd=%.3f)",
        vx, vy, vehicle_theta, vx_map, vy_map, track_psi, vs, vd);

    return {vs, vd};
}

void GlobalPlanner::PublishFrenetOdometry(const builtin_interfaces::msg::Time& timestamp,
                                         const FrenetCoordinates& frenet_coords,
                                         double vs, double vd) {
    // Create new Frenet odometry message
    nav_msgs::msg::Odometry frenet_odom;
    
    // Header
    frenet_odom.header.stamp = timestamp;
    frenet_odom.header.frame_id = "frenet";
    frenet_odom.child_frame_id = "base_link_frenet";
    
    // Pose: s, d coordinates
    frenet_odom.pose.pose.position.x = frenet_coords.s;  // s coordinate
    frenet_odom.pose.pose.position.y = frenet_coords.d;  // d coordinate
    frenet_odom.pose.pose.position.z = 0.0;
    
    // Orientation (keep as identity quaternion for Frenet frame)
    frenet_odom.pose.pose.orientation.x = 0.0;
    frenet_odom.pose.pose.orientation.y = 0.0;
    frenet_odom.pose.pose.orientation.z = 0.0;
    frenet_odom.pose.pose.orientation.w = 1.0;
    
    // Twist: vs, vd velocities
    frenet_odom.twist.twist.linear.x = vs;  // longitudinal velocity
    frenet_odom.twist.twist.linear.y = vd;  // lateral velocity
    frenet_odom.twist.twist.linear.z = 0.0;
    frenet_odom.twist.twist.angular.x = 0.0;
    frenet_odom.twist.twist.angular.y = 0.0;
    frenet_odom.twist.twist.angular.z = 0.0;
    
    // Covariances (set to zero)
    for (int i = 0; i < 36; ++i) {
        frenet_odom.pose.covariance[i] = 0.0;
        frenet_odom.twist.covariance[i] = 0.0;
    }
    
    // Publish the Frenet odometry message
    p_car_state_odom_->publish(frenet_odom);
    
    RCLCPP_DEBUG(this->get_logger(), 
        "Published Frenet odom: s=%.3f, d=%.3f, vs=%.3f, vd=%.3f", 
        frenet_coords.s, frenet_coords.d, vs, vd);
}

void GlobalPlanner::PublishLocalWaypoints(const FrenetCoordinates& current_frenet) {
    if (global_trajectory_.empty()) {
        return;
    }
    
    f110_msgs::msg::WpntArray local_waypoints;
    local_waypoints.header.stamp = this->now();
    local_waypoints.header.frame_id = "map";
    
    // Find the closest waypoint index based on current s coordinate
    int start_idx = last_closest_index_;
    
    // Extract next 80 waypoints from current position
    constexpr int num_local_waypoints = 80;
    int trajectory_size = static_cast<int>(global_trajectory_.size());
    
    // Pre-allocate vector for better performance
    local_waypoints.wpnts.reserve(num_local_waypoints);
    
    // Check if we can copy continuously without wrap-around
    if (start_idx + num_local_waypoints <= trajectory_size) {
        // Continuous copy - no wrap-around needed
        for (int i = 0; i < num_local_waypoints; ++i) {
            const auto& row = global_trajectory_[start_idx + i];
            
            local_waypoints.wpnts.emplace_back();
            auto& waypoint = local_waypoints.wpnts.back();
            
            waypoint.id = static_cast<int32_t>(i);
            waypoint.s_m = row[0];
            waypoint.d_m = 0.0;
            waypoint.x_m = row[1];
            waypoint.y_m = row[2];
            waypoint.d_right = row[3];
            waypoint.d_left = row[4];
            waypoint.psi_rad = row[5];
            waypoint.kappa_radpm = row[6];
            waypoint.vx_mps = row[7];
            waypoint.ax_mps2 = row[8];
        }
    } else {
        // Need wrap-around for circular track
        for (int i = 0; i < num_local_waypoints; ++i) {
            int wp_idx = (start_idx + i) % trajectory_size;
            const auto& row = global_trajectory_[wp_idx];
            
            local_waypoints.wpnts.emplace_back();
            auto& waypoint = local_waypoints.wpnts.back();
            
            waypoint.id = static_cast<int32_t>(i);
            waypoint.s_m = row[0];
            waypoint.d_m = 0.0;
            waypoint.x_m = row[1];
            waypoint.y_m = row[2];
            waypoint.d_right = row[3];
            waypoint.d_left = row[4];
            waypoint.psi_rad = row[5];
            waypoint.kappa_radpm = row[6];
            waypoint.vx_mps = row[7];
            waypoint.ax_mps2 = row[8];
        }
    }
    
    p_local_waypoints_->publish(local_waypoints);
    
    RCLCPP_DEBUG(this->get_logger(), 
        "Published %d local waypoints starting from index %d (s=%.2f)", 
        num_local_waypoints, start_idx, current_frenet.s);
}

int main(int argc, char **argv) {
    std::string node_name = "global_planner";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GlobalPlanner>(node_name));
    rclcpp::shutdown();
    return 0;
}