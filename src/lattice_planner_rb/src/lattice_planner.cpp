#include "lattice_planner_rb/lattice_planner.hpp"
#include <fstream>
#include <sstream>
#include <tf2/utils.h>

namespace lattice_planner {

LatticePlanner::LatticePlanner() : Node("lattice_planner") {
    // Initialize parameters
    planning_horizon_ = 20.0;  // meters
    time_step_ = 0.1;  // seconds
    num_lateral_samples_ = 5;  // 5 paths: center + 4 lateral offsets
    lateral_offset_max_ = 1.5;  // meters (will be adjusted based on track width)
    longitudinal_step_ = 0.5;  // finer resolution for smooth paths
    lattice_planning_distance_ = 15.0;  // 파란색 구간 - 격자 계획 거리
    spline_smoothing_distance_ = 8.0;   // 빨간색 구간 - 스플라인 보간 거리
    
    // 5개 경로의 Frenet d 오프셋 설정 (중앙 + 좌우 2개씩)
    // 트랙 폭 6m 기준으로 안전한 d 오프셋 설정 (1m보다 작게)
    lateral_offsets_ = {-0.8, -0.4, 0.0, 0.4, 0.8};  // Frenet d coordinates in meters
    
    // Enhanced Step 2: Cost weights
    w_progress_ = 1.0;
    w_smoothness_ = 0.8;     
    w_lateral_ = 0.5;        
    w_collision_ = 10.0;
    w_curvature_ = 0.3;
    
    global_path_received_ = false;
    odom_received_ = false;
    
    // TF2 setup
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // ====================================================
    // Subscribers
    // ====================================================
    global_waypoints_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "/global_waypoints", 10,
        std::bind(&LatticePlanner::globalWaypointsCallback, this, std::placeholders::_1));
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/ego_racecar/odom", 10,
        std::bind(&LatticePlanner::odomCallback, this, std::placeholders::_1));
    // ====================================================
    // Publishers
    // ====================================================
    planned_trajectory_pub_ = this->create_publisher<nav_msgs::msg::Path>(
        "/planned_trajectory", 10);
    
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/lattice_trajectories", 10);
    
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
        "/drive", 10);
    
    // Timer for planning loop
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),  // 10 Hz
        std::bind(&LatticePlanner::planningTimerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), "Lattice Planner initialized");
    
    // Load global path from CSV (Step 1)
    loadGlobalPath();
}

void LatticePlanner::globalWaypointsCallback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (msg->poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "Received empty global waypoints");
        return;
    }
    
    global_path_.clear();
    
    for (size_t i = 0; i < msg->poses.size(); i++) {
        Waypoint wp;
        wp.x = msg->poses[i].pose.position.x;
        wp.y = msg->poses[i].pose.position.y;
        wp.z = msg->poses[i].pose.position.z;
        
        // Extract yaw from quaternion
        auto& q = msg->poses[i].pose.orientation;
        wp.yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 
                       1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        
        wp.velocity = 2.0;  // Default velocity
        wp.index = i;
        
        global_path_.push_back(wp);
    }
    
    // Set reference path for Frenet converter
    frenet_converter_.setReferencePath(global_path_);
    global_path_received_ = true;
    
    RCLCPP_INFO(this->get_logger(), "Global waypoints received with %zu waypoints", global_path_.size());
}

void LatticePlanner::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    current_odom_ = *msg;
    odom_received_ = true;
}

void LatticePlanner::planningTimerCallback() {
    if (!global_path_received_ || !odom_received_) {
        return;
    }
    
    // 매번 ego 위치 기준으로 spline path 재생성
    generateSplinePaths();
    
    // Get current position in Frenet coordinates
    double current_x = current_odom_.pose.pose.position.x;
    double current_y = current_odom_.pose.pose.position.y;
    
    FrenetPoint current_frenet = frenet_converter_.cartesianToFrenet(current_x, current_y);
    
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Current position: (%.2f, %.2f) -> Frenet: (s=%.2f, d=%.2f)",
                         current_x, current_y, current_frenet.s, current_frenet.d);
    
    // Generate trajectories from ego-connected spline paths
    std::vector<LatticeTrajectory> sample_trajectories = generateTrajectories(current_frenet);
    
    if (!sample_trajectories.empty()) {
        // Select best trajectory
        LatticeTrajectory best_trajectory = selectBestTrajectory(sample_trajectories);
        
        // Publish trajectory and visualization
        publishTrajectory(best_trajectory);
        publishVisualization(sample_trajectories);
    }
}

void LatticePlanner::loadGlobalPath() {
    // CSV 파일에서 글로벌 경로 로드
    std::string csv_file_path = "/home/jeong/sim_ws/src/planning_ws/src/lattice_planner_rb/waypoints/redbull_0.csv";
    
    std::ifstream file(csv_file_path);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_file_path.c_str());
        return;
    }
    
    std::string line;
    global_path_.clear();
    
    // Skip header lines
    std::getline(file, line); // # comment
    std::getline(file, line); // # comment
    std::getline(file, line); // header
    
    int index = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> tokens;
        
        // Split by semicolon
        while (std::getline(ss, item, ';')) {
            // Remove whitespace
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            tokens.push_back(item);
        }
        
        if (tokens.size() >= 9) {
            Waypoint wp;
            wp.x = std::stod(tokens[1]);  // x_m
            wp.y = std::stod(tokens[2]);  // y_m
            wp.z = 0.0;
            wp.yaw = std::stod(tokens[5]);  // psi_rad
            wp.velocity = std::stod(tokens[7]);  // vx_mps
            wp.curvature = std::stod(tokens[6]);  // kappa_radpm
            wp.road_width_right = std::stod(tokens[3]);  // d_right
            wp.road_width_left = std::stod(tokens[4]);   // d_left
            wp.index = index++;
            
            global_path_.push_back(wp);
        }
    }
    
    file.close();
    
    if (!global_path_.empty()) {
        // 🔄 CSV 데이터가 시계방향이므로 반시계방향으로 뒤집기
        std::reverse(global_path_.begin(), global_path_.end());
        
        // 뒤집은 후 각 waypoint의 yaw 방향도 반대로 수정
        for (auto& wp : global_path_) {
            wp.yaw += M_PI;  // 180도 회전
            if (wp.yaw > M_PI) wp.yaw -= 2 * M_PI;  // [-π, π] 범위로 정규화
        }
        
        // 인덱스 재할당
        for (size_t i = 0; i < global_path_.size(); i++) {
            global_path_[i].index = i;
        }
        
        // Set reference path for Frenet converter
        frenet_converter_.setReferencePath(global_path_);
        global_path_received_ = true;
        
        RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints from CSV file (reversed for counter-clockwise)", global_path_.size());
        RCLCPP_INFO(this->get_logger(), "First waypoint: (%.3f, %.3f) yaw=%.3f", 
                    global_path_[0].x, global_path_[0].y, global_path_[0].yaw);
        RCLCPP_INFO(this->get_logger(), "Ego-connected lattice planner ready");
    } else {
        RCLCPP_ERROR(this->get_logger(), "No valid waypoints loaded from CSV file");
    }
}

std::vector<LatticeTrajectory> LatticePlanner::generateTrajectories(const FrenetPoint& current_state) {
    std::vector<LatticeTrajectory> trajectories;
    
    if (spline_paths_.empty()) {
        RCLCPP_WARN(this->get_logger(), "No spline paths available");
        return trajectories;
    }
    
    // 현재 위치에서 가장 가까운 waypoint 찾기
    double current_x = current_odom_.pose.pose.position.x;
    double current_y = current_odom_.pose.pose.position.y;
    int closest_idx = findClosestWaypointIndex(current_x, current_y);
    
    // 각 스플라인 경로에 대해 궤적 생성
    for (int path_idx = 0; path_idx < static_cast<int>(spline_paths_.size()); path_idx++) {
        LatticeTrajectory traj;
        traj.path_index = path_idx - 2;  // -2, -1, 0, +1, +2
        
        // 스무스한 궤적 생성 (spline interpolation 사용)
        std::vector<CartesianPoint> trajectory_points = generateSmoothTrajectory(path_idx, current_state);
        
        if (!trajectory_points.empty() && trajectory_points.size() > 3) {
            traj.points = trajectory_points;
            traj.cost = calculateCost(traj);
            trajectories.push_back(traj);
        }
    }
    
    // 생성된 궤적의 세부 정보 로깅
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                         "🎯 Generated %zu lattice trajectories from %zu spline paths", 
                         trajectories.size(), spline_paths_.size());
    
    for (size_t i = 0; i < trajectories.size(); i++) {
        const char* color_name = "";
        switch (trajectories[i].path_index) {
            case -2: color_name = "🔴RED"; break;
            case -1: color_name = "🟠ORANGE"; break;
            case 0:  color_name = "🟢GREEN"; break;
            case 1:  color_name = "🔵BLUE"; break;
            case 2:  color_name = "🟣PURPLE"; break;
            default: color_name = "⚪GRAY"; break;
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                             "  Trajectory %zu: %s path_index=%d, points=%zu, cost=%.3f", 
                             i, color_name, trajectories[i].path_index, trajectories[i].points.size(), trajectories[i].cost);
    }
    
    return trajectories;
}

//=======================================================================
// Spline 으로 smooth한 궤적 생성 
//=======================================================================
// 현재 위치에서 가장 가까운 spline 상의 점을 찾아서 planning horizon 까지의
// 부드러운 궤적을 생성합니다.
// 이 함수는 현재 위치와 가장 가까운 spline 상의 점을 찾아서, 해당
// 점부터 planning horizon 까지의 궤적을 생성합니다.
// 이때, spline interpolation을 사용하여 부드러운 궤적을 생성합니다
std::vector<CartesianPoint> LatticePlanner::generateSmoothTrajectory(int path_index, const FrenetPoint& current_state) {
    std::vector<CartesianPoint> trajectory_points;
    
    if (path_index < 0 || path_index >= static_cast<int>(spline_paths_.size())) {
        return trajectory_points;
    }
    
    const CubicSplinePath& spline_path = spline_paths_[path_index];
    double current_x = current_odom_.pose.pose.position.x;
    double current_y = current_odom_.pose.pose.position.y;
    
    // 현재 위치에서 가장 가까운 spline 상의 점 찾기
    double min_dist = std::numeric_limits<double>::max();
    double start_s = 0.0;
    
    // 균등하게 분포된 점들을 생성하여 가장 가까운 점 찾기
    auto uniform_points = spline_path.generateUniformPoints(0.1);
    
    for (const auto& point : uniform_points) {
        double dx = point[1] - current_x;  // x coordinate
        double dy = point[2] - current_y;  // y coordinate
        double dist = std::sqrt(dx*dx + dy*dy);
        
        if (dist < min_dist) {
            min_dist = dist;
            start_s = point[0];  // s coordinate
        }
    }
    
    // 현재 위치부터 planning horizon까지의 궤적 생성
    double target_s = start_s + lattice_planning_distance_;
    double total_length = spline_path.getTotalLength();
    
    // Cubic spline을 통한 부드러운 궤적 생성
    for (double s = start_s; s <= target_s && s <= total_length; s += longitudinal_step_) {
        CartesianPoint point;
        
        // Cubic spline 보간으로 위치 계산
        auto pos = spline_path.interpolatePosition(s);
        point.x = pos.first;
        point.y = pos.second;
        point.theta = spline_path.interpolateYaw(s);
        point.curvature = spline_path.interpolateCurvature(s);
        point.velocity = 2.0;  // 기본 속도
        point.acceleration = 0.0;
        
        trajectory_points.push_back(point);
    }
    
    return trajectory_points;
}

int LatticePlanner::findClosestWaypointIndex(double x, double y) {
    if (global_path_.empty()) {
        return 0;
    }
    
    double min_dist = std::numeric_limits<double>::max();
    int closest_idx = 0;
    
    for (size_t i = 0; i < global_path_.size(); i++) {
        double dx = global_path_[i].x - x;
        double dy = global_path_[i].y - y;
        double dist = std::sqrt(dx*dx + dy*dy);
        
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    
    return closest_idx;
}

//=======================================================================
// 5개중에 select path 
//=======================================================================
// 현재 위치에서 가장 가까운 스플라인 경로를 선택합니다.
// 이 함수는 현재 위치에서 가장 가까운 스플라인 경로를 찾아서,
// 해당 경로의 인덱스를 반환합니다.
LatticeTrajectory LatticePlanner::selectBestTrajectory(const std::vector<LatticeTrajectory>& trajectories) {
    if (trajectories.empty()) {
        return LatticeTrajectory();
    }
    
    // Find trajectory with minimum cost
    auto best_it = std::min_element(trajectories.begin(), trajectories.end(),
                                    [](const LatticeTrajectory& a, const LatticeTrajectory& b) {
                                        return a.cost < b.cost;
                                    });
    
    return *best_it;
}


//=======================================================================
// cost 계산
//=======================================================================
double LatticePlanner::calculateCost(const LatticeTrajectory& trajectory) {
    if (trajectory.points.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    // 새로운 비용 계산 - 스플라인 경로 기반
    double lateral_cost = 0.0;
    double smoothness_cost = 0.0;
    double curvature_cost = 0.0;
    double progress_cost = 0.0;
    double center_preference_cost = 0.0;
    
    // 1. 중앙 경로 선호 비용 (center path preference)
    center_preference_cost = std::abs(trajectory.path_index) * 0.5;
    
    // 2. 곡률 비용 (curvature cost)
    for (const auto& point : trajectory.points) {
        curvature_cost += std::abs(point.curvature);
    }
    curvature_cost /= trajectory.points.size();  // 평균 곡률
    
    // 3. 부드러움 비용 (smoothness cost) - 연속된 점들 간의 각도 변화
    for (size_t i = 1; i < trajectory.points.size() - 1; i++) {
        double dx1 = trajectory.points[i].x - trajectory.points[i-1].x;
        double dy1 = trajectory.points[i].y - trajectory.points[i-1].y;
        double dx2 = trajectory.points[i+1].x - trajectory.points[i].x;
        double dy2 = trajectory.points[i+1].y - trajectory.points[i].y;
        
        double angle1 = atan2(dy1, dx1);
        double angle2 = atan2(dy2, dx2);
        double angle_diff = std::abs(angle2 - angle1);
        if (angle_diff > M_PI) angle_diff = 2 * M_PI - angle_diff;
        
        smoothness_cost += angle_diff * angle_diff;
    }
    smoothness_cost /= std::max(1.0, static_cast<double>(trajectory.points.size() - 2));
    
    // 4. 진행 비용 (progress cost) - 충분한 거리 진행을 장려
    if (trajectory.points.size() >= 2) {
        double dx = trajectory.points.back().x - trajectory.points.front().x;
        double dy = trajectory.points.back().y - trajectory.points.front().y;
        double actual_progress = std::sqrt(dx*dx + dy*dy);
        progress_cost = std::max(0.0, lattice_planning_distance_ - actual_progress);
    }
    
    // 5. 속도 기반 비용 (velocity-based cost) - 높은 곡률에서 속도 감소
    double velocity_cost = 0.0;
    for (const auto& point : trajectory.points) {
        if (std::abs(point.curvature) > 0.1) {  // 높은 곡률
            velocity_cost += std::abs(point.curvature) * 2.0;
        }
    }
    
    // 전체 비용 계산 (가중치 적용)
    double total_cost = w_lateral_ * center_preference_cost + 
                       w_smoothness_ * smoothness_cost + 
                       w_curvature_ * curvature_cost +
                       w_progress_ * progress_cost * 0.1 +  // 진행 비용은 작게
                       0.3 * velocity_cost;  // 속도 비용
    
    return total_cost;
}

void LatticePlanner::generateSplinePaths() {
    spline_paths_.clear();
    
    if (global_path_.empty() || !odom_received_) {
        RCLCPP_WARN(this->get_logger(), "Global path empty or no odometry");
        return;
    }
    
    // 1. 현재 ego 위치와 방향
    double ego_x = current_odom_.pose.pose.position.x;
    double ego_y = current_odom_.pose.pose.position.y;
    double ego_yaw = tf2::getYaw(current_odom_.pose.pose.orientation);
    
    // 2. ego에서 가장 가까운 global waypoint 찾기
    int closest_idx = findClosestWaypointIndex(ego_x, ego_y);
    
    // 3. ego 앞쪽 planning horizon 구간 선택
    std::vector<Waypoint> future_waypoints;
    double accumulated_distance = 0.0;
    
    for (int i = closest_idx; i < (int)global_path_.size() && accumulated_distance < lattice_planning_distance_; i++) {
        if (i > closest_idx) {
            double dx = global_path_[i].x - global_path_[i-1].x;
            double dy = global_path_[i].y - global_path_[i-1].y;
            accumulated_distance += std::sqrt(dx*dx + dy*dy);
        }
        future_waypoints.push_back(global_path_[i]);
    }
    
    if (future_waypoints.empty()) {
        RCLCPP_WARN(this->get_logger(), "No future waypoints found");
        return;
    }
    
    // 4. 5개의 d_offset에 대해 각각 ego → future_path 연결
    lateral_offsets_ = {-0.8, -0.4, 0.0, 0.4, 0.8};
    
    for (double d_offset : lateral_offsets_) {
        std::vector<double> path_xs, path_ys;
        
        // 4-1. ego 현재 위치 추가
        path_xs.push_back(ego_x);
        path_ys.push_back(ego_y);
        
        // 4-2. ego 앞쪽 연결점 생성 (ego 방향으로 조금 전진)
        double connection_distance = 1.5;  // 1.5m 앞에서 연결
        double connection_x = ego_x + connection_distance * cos(ego_yaw);
        double connection_y = ego_y + connection_distance * sin(ego_yaw);
        path_xs.push_back(connection_x);
        path_ys.push_back(connection_y);
        
        // 4-3. future waypoints에 offset 적용해서 추가
        for (const auto& wp : future_waypoints) {
            // offset 적용 (waypoint의 법선 방향으로)
            double offset_x = wp.x - sin(wp.yaw) * d_offset;
            double offset_y = wp.y + cos(wp.yaw) * d_offset;
            
            // ego에서 너무 가까운 점들은 제외 (중복 방지)
            double dist_from_ego = std::sqrt((offset_x - ego_x)*(offset_x - ego_x) + 
                                           (offset_y - ego_y)*(offset_y - ego_y));
            if (dist_from_ego > 2.0) {  // 2m 이상 떨어진 점만 추가
                path_xs.push_back(offset_x);
                path_ys.push_back(offset_y);
            }
        }
        
        // 4-4. 충분한 점이 있는지 확인
        if (path_xs.size() < 3) {
            RCLCPP_WARN(this->get_logger(), "Not enough points for offset %.2f (only %zu points)", 
                        d_offset, path_xs.size());
            continue;
        }
        
        // 4-5. parametric spline 생성 (arc-length 기반)
        try {
            std::vector<double> s_values;  // arc-length parameter
            s_values.push_back(0.0);
            
            // arc-length 계산
            for (size_t i = 1; i < path_xs.size(); i++) {
                double dx = path_xs[i] - path_xs[i-1];
                double dy = path_ys[i] - path_ys[i-1];
                double ds = std::sqrt(dx*dx + dy*dy);
                s_values.push_back(s_values.back() + ds);
            }
            
            // s를 기준으로 x, y를 각각 spline 보간
            CubicSplinePath spline_x(s_values, path_xs);
            CubicSplinePath spline_y(s_values, path_ys);
            
            // 최종 uniform sampling으로 spline path 생성
            std::vector<double> uniform_xs, uniform_ys;
            double total_length = s_values.back();
            
            for (double s = 0.0; s <= total_length; s += 0.3) {  // 0.3m 간격
                auto x_interp = spline_x.interpolatePosition(s);
                auto y_interp = spline_y.interpolatePosition(s);
                uniform_xs.push_back(x_interp.first);
                uniform_ys.push_back(y_interp.first);
            }
            
            // x 좌표 기준으로 정렬하여 단조증가 보장
            std::vector<std::pair<double, double>> xy_pairs;
            for (size_t i = 0; i < uniform_xs.size(); i++) {
                xy_pairs.push_back({uniform_xs[i], uniform_ys[i]});
            }
            std::sort(xy_pairs.begin(), xy_pairs.end());
            
            // 중복된 x 값 제거
            std::vector<double> final_xs, final_ys;
            for (size_t i = 0; i < xy_pairs.size(); i++) {
                if (i == 0 || xy_pairs[i].first > xy_pairs[i-1].first + 0.01) {  // 1cm 이상 차이
                    final_xs.push_back(xy_pairs[i].first);
                    final_ys.push_back(xy_pairs[i].second);
                }
            }
            
            if (final_xs.size() >= 3) {
                CubicSplinePath final_spline(final_xs, final_ys);
                spline_paths_.push_back(final_spline);
                
                RCLCPP_DEBUG(this->get_logger(), "✅ Created spline path for offset %.2f with %zu points", 
                             d_offset, final_xs.size());
            } else {
                RCLCPP_WARN(this->get_logger(), "Not enough unique x points for offset %.2f", d_offset);
            }
            
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "❌ Failed to create spline for offset %.2f: %s", d_offset, e.what());
        }
    }
    
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "🛣️  Generated %zu ego-connected spline paths from ego position (%.2f, %.2f)", 
                         spline_paths_.size(), ego_x, ego_y);
}

//=======================================================================
// Frenet 좌표계 기반 d 오프셋 경로 생성
//=======================================================================
// 기준 경로(global waypoints)의 각 점을 Frenet 좌표계로 변환한 후,
// d 좌표에 오프셋을 추가하여 다시 Cartesian 좌표계로 변환하여 경로 생성
// 이 방식은 spline의 s 좌표를 따라 법선 방향으로 정확한 d 오프셋을 적용
CubicSplinePath LatticePlanner::createFrenetOffsetPath(const std::vector<Waypoint>& reference_path, double d_offset) {
    std::vector<double> x_values, y_values;
    
    for (size_t i = 0; i < reference_path.size(); i++) {
        const auto& wp = reference_path[i];
        
        // 1. 기준 waypoint를 Frenet 좌표계로 변환
        FrenetPoint frenet_point = frenet_converter_.cartesianToFrenet(wp.x, wp.y);
        
        // 2. d 좌표에 오프셋 추가 (s는 그대로 유지)
        FrenetPoint offset_frenet;
        offset_frenet.s = frenet_point.s;
        offset_frenet.d = frenet_point.d + d_offset;  // Frenet d 오프셋 적용
        offset_frenet.s_dot = 0.0;
        offset_frenet.d_dot = 0.0;
        offset_frenet.s_ddot = 0.0;
        offset_frenet.d_ddot = 0.0;
        
        // 3. 오프셋된 Frenet 좌표를 Cartesian 좌표계로 변환
        auto cartesian_point = frenet_converter_.frenetToCartesian(offset_frenet);
        
        x_values.push_back(cartesian_point.x);
        y_values.push_back(cartesian_point.y);
        
        // 디버깅을 위한 주기적 로깅 (처음 몇 개 점만)
        if (i < 3) {
            RCLCPP_DEBUG(this->get_logger(), 
                        "Waypoint %zu: Original(%.2f,%.2f) -> Frenet(s=%.2f,d=%.2f) -> Offset_Frenet(s=%.2f,d=%.2f) -> Cartesian(%.2f,%.2f)",
                        i, wp.x, wp.y, frenet_point.s, frenet_point.d, 
                        offset_frenet.s, offset_frenet.d, cartesian_point.x, cartesian_point.y);
        }
    }
    
    // CubicSplinePath 생성 (Frenet 기반 정확한 d 오프셋 경로)
    CubicSplinePath spline_path(x_values, y_values);
    
    return spline_path;
}

nav_msgs::msg::Path LatticePlanner::trajectoryToPath(const LatticeTrajectory& trajectory) {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->get_clock()->now();
    path_msg.header.frame_id = "map";
    
    for (const auto& point : trajectory.points) {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = path_msg.header;
        pose_stamped.pose.position.x = point.x;
        pose_stamped.pose.position.y = point.y;
        pose_stamped.pose.position.z = 0.0;
        
        // Convert theta to quaternion
        tf2::Quaternion q;
        q.setRPY(0, 0, point.theta);
        pose_stamped.pose.orientation = tf2::toMsg(q);
        
        path_msg.poses.push_back(pose_stamped);
    }
    
    return path_msg;
}

//=======================================================================
// 궤적 마커 생성 (시각화용)
//=======================================================================
// 선택된 궤적을 굵고 파란색으로 표시하고, 나머지 궤적은 얇고 회색으로 표시
// 현재 위치는 노란색 구체로 표시합니다.
// 이 함수는 궤적의 시각화를 위한 마커 배열을 생성
visualization_msgs::msg::MarkerArray LatticePlanner::createTrajectoryMarkers(
    const std::vector<LatticeTrajectory>& trajectories) {
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // 선택된 궤적의 인덱스 찾기 (최소 비용)
    int best_trajectory_idx = 0;
    if (!trajectories.empty()) {
        double min_cost = trajectories[0].cost;
        for (size_t i = 1; i < trajectories.size(); i++) {
            if (trajectories[i].cost < min_cost) {
                min_cost = trajectories[i].cost;
                best_trajectory_idx = i;
            }
        }
    }
    
    for (size_t i = 0; i < trajectories.size(); i++) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "lattice_trajectories";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // 5개 궤적을 서로 다른 색상으로 구분 (더 얇은 선으로)
        marker.scale.x = 0.08;  // 더 얇게 변경 (0.15 → 0.08)
        marker.color.a = 0.8;   // 약간 투명하게
        
        // path_index에 따른 색상 구분 (RGB 값 더 명확하게)
        if (i < trajectories.size()) {
            int path_idx = trajectories[i].path_index;
            switch (path_idx) {
                case -2:  // 좌측 끝 - 밝은 빨간색
                    marker.color.r = 1.0;
                    marker.color.g = 0.0;
                    marker.color.b = 0.0;
                    break;
                case -1:  // 좌측 중간 - 주황색
                    marker.color.r = 1.0;
                    marker.color.g = 0.65;
                    marker.color.b = 0.0;
                    break;
                case 0:   // 중앙 (global path) - 밝은 초록색
                    marker.color.r = 0.0;
                    marker.color.g = 1.0;
                    marker.color.b = 0.0;
                    marker.scale.x = 0.12;  // 중앙 경로는 조금만 굵게 (0.2 → 0.12)
                    break;
                case 1:   // 우측 중간 - 밝은 파란색
                    marker.color.r = 0.0;
                    marker.color.g = 0.5;
                    marker.color.b = 1.0;
                    break;
                case 2:   // 우측 끝 - 밝은 보라색
                    marker.color.r = 0.8;
                    marker.color.g = 0.0;
                    marker.color.b = 1.0;
                    break;
                default:  // 기본값 - 노란색 (문제 감지용)
                    marker.color.r = 1.0;
                    marker.color.g = 1.0;
                    marker.color.b = 0.0;
                    RCLCPP_WARN(this->get_logger(), "Unknown path_index: %d", path_idx);
                    break;
            }
        }
        
        // 선택된 궤적 추가 강조 (테두리 효과)
        if (static_cast<int>(i) == best_trajectory_idx) {
            marker.scale.x *= 1.3;  // 선택된 궤적은 약간만 굵게 (1.5 → 1.3)
            marker.color.a = 1.0;   // 완전 불투명
            
            // 선택된 궤적 로깅
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "⭐ Selected trajectory %zu (path_index=%d) with cost=%.3f", 
                                 i, trajectories[i].path_index, trajectories[i].cost);
        }
        
        for (const auto& point : trajectories[i].points) {
            geometry_msgs::msg::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = 0.02 * i;  // 각 궤적을 약간 다른 높이에 표시하여 겹침 방지
            marker.points.push_back(p);
        }
        
        marker_array.markers.push_back(marker);
    }
    
    // 현재 위치 마커 추가
    visualization_msgs::msg::Marker ego_marker;
    ego_marker.header.frame_id = "map";
    ego_marker.header.stamp = this->get_clock()->now();
    ego_marker.ns = "ego_position";
    ego_marker.id = 1000;
    ego_marker.type = visualization_msgs::msg::Marker::SPHERE;
    ego_marker.action = visualization_msgs::msg::Marker::ADD;
    
    ego_marker.pose.position.x = current_odom_.pose.pose.position.x;
    ego_marker.pose.position.y = current_odom_.pose.pose.position.y;
    ego_marker.pose.position.z = 0.0;
    ego_marker.pose.orientation = current_odom_.pose.pose.orientation;
    
    ego_marker.scale.x = 0.5;
    ego_marker.scale.y = 0.5;
    ego_marker.scale.z = 0.2;
    ego_marker.color.r = 1.0;
    ego_marker.color.g = 1.0;
    ego_marker.color.b = 0.0;  // 노란색
    ego_marker.color.a = 1.0;
    
    marker_array.markers.push_back(ego_marker);
    
    return marker_array;
}

void LatticePlanner::publishTrajectory(const LatticeTrajectory& trajectory) {
    nav_msgs::msg::Path path_msg = trajectoryToPath(trajectory);
    planned_trajectory_pub_->publish(path_msg);
}

void LatticePlanner::publishVisualization(const std::vector<LatticeTrajectory>& trajectories) {
    visualization_msgs::msg::MarkerArray markers = createTrajectoryMarkers(trajectories);
    marker_pub_->publish(markers);
}

} // namespace lattice_planner

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<lattice_planner::LatticePlanner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
