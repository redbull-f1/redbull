#include "spline_planner_dh_test/spline_planner.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

SplinePlanner::SplinePlanner() : Node("spline_planner_node"),
    cur_s_(0.0), 
    cur_d_(0.0), 
    cur_vs_(0.0),
    cur_vd_(0.0),
    lookahead_(6.0), //10은 너무 멀어 -> 6로 바꿔서 ㄱㄱ 
    evasion_dist_(0.30), //파라미터 0.65 -> 0.6 -> 0.55-> 0.50 -> 0.30 바꾸면서 실험해보기 
    obs_traj_thresh_(0.5), //파라미터 1.0 -> 0.5로 바꾸기 
    spline_bound_mindist_(0.2), //파라미터 0.2 -> 0.1로 바꾸기 -> 다시 0.2로 바꾸기
    gb_max_s_(0.0), 
    gb_max_idx_(0),    // <- 이제 "웨이포인트 개수"로 사용
    gb_vmax_(0.0),
    pre_apex_0_(-4.0), 
    pre_apex_1_(-3.0), 
    pre_apex_2_(-1.5),
    post_apex_0_(2.0), 
    post_apex_1_(3.0), post_apex_2_(4.0),
    last_ot_side_(""), 
    last_switch_time_(this->get_clock()->now()) {
    
//====================================================================================
// subscribers, publishers 정의 
//====================================================================================
    //--------------------------------subscribers-------------------------------------
    obstacles_sub_ = this->create_subscription<spline_planner_dh_test::msg::ObstacleArray>(
        "/obstacles", 10,
        std::bind(&SplinePlanner::obstacles_callback, this, std::placeholders::_1));
    
    ego_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/ego_racecar/odom", 10,
        std::bind(&SplinePlanner::ego_odom_callback, this, std::placeholders::_1));
    
    global_waypoints_sub_ = this->create_subscription<spline_planner_dh_test::msg::WpntArray>(
        "/global_waypoints", 10,
        std::bind(&SplinePlanner::global_waypoints_callback, this, std::placeholders::_1));
    

    //-------------------------------publishers-------------------------------------
    local_trajectory_pub_ = this->create_publisher<spline_planner_dh_test::msg::OTWpntArray>(
        "/local_trajectory", 10);
    
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planner/avoidance/markers", 10);
    
    closest_obs_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/planner/avoidance/considered_OBS", 10);
    
    // 모든 장애물 시각화용 publisher 추가
    all_obstacles_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planner/all_obstacles", 10);
    
    reference_path_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planner/reference_path", 10);
    
    frenet_debug_pub_ = this->create_publisher<spline_planner_dh_test::msg::FrenetDebugArray>(
        "/planner/frenet_debug", 10);
    
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
        std::chrono::milliseconds(25), // 40 Hz
        std::bind(&SplinePlanner::timer_callback, this));
    
    RCLCPP_INFO(this->get_logger(), "Spline planner initialized successfully");
}

//=====================================================================================
// Callbacks 정의 (obstacle, ego_odom, global_waypoints)
//=====================================================================================
void SplinePlanner::obstacles_callback(const spline_planner_dh_test::msg::ObstacleArray::SharedPtr msg) {
    // 디버깅용: 수신 확인
    RCLCPP_INFO(this->get_logger(), "obstacles_callback: %zu obstacles", msg->obstacles.size());
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
    
    // Calculate yaw from quaternion
    ego_yaw_ = quaternion_to_yaw(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );
}

//------------------------global waypoint callback (gb_max_idx_, gb_max_s_, gb_vmax 계산)-----
void SplinePlanner::global_waypoints_callback(const spline_planner_dh_test::msg::WpntArray::SharedPtr msg) {
    global_waypoints_ = msg;
    
    if (!msg->wpnts.empty()) {
        // 마지막 id가 아니라 "웨이포인트 개수"를 저장
        gb_max_idx_ = static_cast<int>(msg->wpnts.size());
        gb_max_s_   = msg->wpnts.back().s_m;
        
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
            
        // 첫 5개 waypoint의 frenet 좌표 출력
        RCLCPP_INFO(this->get_logger(), "첫 5개 Global Waypoint Frenet 좌표:");
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), msg->wpnts.size()); i++) {
            const auto& wp = msg->wpnts[i];
            RCLCPP_INFO(this->get_logger(), 
                "Waypoint %zu: s=%.2f, d=%.2f, x=%.2f, y=%.2f, v=%.2f",
                i, wp.s_m, wp.d_m, wp.x_m, wp.y_m, wp.vx_mps);
        }
    }
}

void SplinePlanner::timer_callback() {
    process_spline_planning();
}

//===========================================================================================
//함수
//============================================================================================
void SplinePlanner::process_spline_planning() {
    // Check for minimum required data (ego_odom and global_waypoints)
    if (!ego_odom_ || !global_waypoints_ || !frenet_converter_) {
        return;
    }
    if (gb_max_idx_ <= 1 || gb_max_s_ <= 0.0) {
        // 웨이포인트가 충분치 않으면 계산 중단
        return;
    }
    
    // Always publish reference path when we have waypoints
    publish_reference_path();
    
    // Always publish ego frenet debug info
    std::vector<ObstacleFrenet> empty_obstacles;
    publish_frenet_debug_info(empty_obstacles);

    spline_planner_dh_test::msg::OTWpntArray trajectory_msg;
    visualization_msgs::msg::MarkerArray markers_msg;
    
    // ========================================================================================
    // 장애물이 있으면 frenet 좌표계로 변환하고, 가까운 장애물 filtering, 가장 가까운 장애물 선택 후 회피 경로 생성
    // ========================================================================================
    if (obstacles_) {
        // 장애물 Frenet 좌표계로 변환
        auto frenet_obstacles = convert_obstacles_to_frenet(*obstacles_);
        
        // Frenet 디버깅 정보 publish
        publish_frenet_debug_info(frenet_obstacles);
        // 모든 장애물 시각화 publish
        publish_all_obstacles(frenet_obstacles);

        // -------가까운 장애물 filtering 리스트 close_obstacles-----------------
        auto close_obstacles = filter_close_obstacles(frenet_obstacles);
        
        if (!close_obstacles.empty()) {
            RCLCPP_INFO(this->get_logger(), 
                "발견된 가까운 장애물 수: %zu개", close_obstacles.size());
            
            // ------------------ 가장 가까운 장애물 선택 후 회피 경로 생성 ------------------
            auto closest_obstacle = *std::min_element(close_obstacles.begin(), close_obstacles.end(),
                [this](const ObstacleFrenet& a, const ObstacleFrenet& b) {
                    double dist_a = std::fmod(a.s_center - cur_s_ + gb_max_s_, gb_max_s_);
                    double dist_b = std::fmod(b.s_center - cur_s_ + gb_max_s_, gb_max_s_);
                    return dist_a < dist_b;
                });

            trajectory_msg = generate_evasion_trajectory(closest_obstacle);
            
            // ----------evasion_trajectory 시각화----------------
            markers_msg = create_trajectory_markers(trajectory_msg);
            
            // ----------가장 가까운 장애물 시각화----------------
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

//===========================================================================================
//함수: 필수 메시지 대기(ego/odom, global_waypoints) loop(10Hz, 10초 동안 대기)
//===========================================================================================
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

//===========================================================================================
//함수: FrenetConverter 초기화
//===========================================================================================
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


//==============================================================================================
// obstacle map -> frenet 좌표계로 변경
// 새로운 ObstacleWpnt 메시지 형식: id, x, y, vx, vy, yaw, size
// 장애물의 중심점만 Frenet 좌표계로 변환하고, vs(s 방향 속도) 계산
//===============================================================================================
std::vector<ObstacleFrenet> SplinePlanner::convert_obstacles_to_frenet(
    const spline_planner_dh_test::msg::ObstacleArray& obstacles) {
    
    std::vector<ObstacleFrenet> frenet_obstacles;

    if (gb_max_idx_ <= 1 || gb_max_s_ <= 0.0) {
        return frenet_obstacles;
    }
    // 평균 간격: N개 포인트가 0~max_s를 커버 → 분모는 (N-1)
    const int denom = std::max(1, gb_max_idx_ - 1);
    const double wpnt_dist = gb_max_s_ / static_cast<double>(denom);
    
    for (const auto& obs : obstacles.obstacles) {
        ObstacleFrenet frenet_obs;
        frenet_obs.id = obs.id;
        frenet_obs.yaw = obs.yaw;
        frenet_obs.size = obs.size;
        
        // 중심점을 Frenet 좌표계로 변환
        auto center_frenet = frenet_converter_->cartesian_to_frenet(obs.x, obs.y);
        frenet_obs.s_center = center_frenet.first;
        frenet_obs.d_center = center_frenet.second;

        // 인덱스 계산 (가드 포함)
        int gb_idx = 0;
        if (wpnt_dist > 1e-9) {
            gb_idx = static_cast<int>(std::floor(frenet_obs.s_center / wpnt_dist));
            if (gb_idx < 0) gb_idx = 0;
            if (gb_idx >= gb_max_idx_) gb_idx = gb_max_idx_ - 1;
        }

        // 해당 위치에서의 트랙 방향(psi)
        double track_psi = global_waypoints_->wpnts[gb_idx].psi_rad;
        
        // Frenet 좌표계로 속도 변환:
        // vs = vx * cos(track_psi) + vy * sin(track_psi)  (s 방향 속도)
        // vd = -vx * sin(track_psi) + vy * cos(track_psi) (d 방향 속도)
        frenet_obs.vs = obs.vx * std::cos(track_psi) + obs.vy * std::sin(track_psi);
        frenet_obs.vd = -obs.vx * std::sin(track_psi) + obs.vy * std::cos(track_psi);
        
        // s wrap
        if (gb_max_s_ > 0) {
            frenet_obs.s_center = std::fmod(frenet_obs.s_center + gb_max_s_, gb_max_s_);
        }
        
        frenet_obstacles.push_back(frenet_obs);                                   
        RCLCPP_INFO(this->get_logger(), 
            "장애물 %d: Frenet좌표(s=%.2f, d=%.2f), 속도(vs=%.2f, vd=%.2f), size=%.2f",
            frenet_obs.id, frenet_obs.s_center, frenet_obs.d_center, 
            frenet_obs.vs, frenet_obs.vd, frenet_obs.size);
    }
    
    return frenet_obstacles;
}



//=========================================================================================
// 가까운 거리에 있는 장애물 filtering하는 함수 (lookahead = 전방 10m, obs_traj_thresh_ = 0.3m) 
//=========================================================================================
std::vector<ObstacleFrenet> SplinePlanner::filter_close_obstacles(
    const std::vector<ObstacleFrenet>& obstacles) {
    
    std::vector<ObstacleFrenet> close_obstacles;
    
    for (const auto& obs : obstacles) {
        // Check if obstacle is within trajectory threshold
        if (std::abs(obs.d_center) < obs_traj_thresh_) {
            // 0~gb_max_s_ 범위에서 장애물과 ego가 얼마나 떨어져 있는지 확인(dist_in_front)
            // ---------------------------------------------------------------------------
            // EX) ego s=190, obs = 10, gb_max_s_ = 200 -> dist_in_front = 20
            // ---------------------------------------------------------------------------
            double dist_in_front = std::fmod(obs.s_center - cur_s_ + gb_max_s_, gb_max_s_);
            if (dist_in_front < lookahead_) {
                close_obstacles.push_back(obs);
            }
        }
    }
    
    return close_obstacles;
}


//===========================================================================================
// 장애물의 위치에 따라 left, right gap 계산하고, 더 많은 공간 선택 
// +
// d_apex 계산
//===========================================================================================
std::pair<std::string, double> SplinePlanner::calculate_more_space(const ObstacleFrenet& obstacle) {
    if (gb_max_idx_ <= 1 || gb_max_s_ <= 0.0) {
        return {"right", 0.0};
    }

    const int denom = std::max(1, gb_max_idx_ - 1);
    const double wpnt_dist = gb_max_s_ / static_cast<double>(denom);
    int gb_idx = 0;
    if (wpnt_dist > 1e-9) {
        gb_idx = static_cast<int>(std::floor(obstacle.s_center / wpnt_dist));
        if (gb_idx < 0) gb_idx = 0;
        if (gb_idx >= gb_max_idx_) gb_idx = gb_max_idx_ - 1;
    }
    
    const auto& waypoint = global_waypoints_->wpnts[gb_idx];
    
    // 장애물의 좌우 경계를 size를 이용해 계산
    double obstacle_d_left = obstacle.d_center + obstacle.size;   // obstacle의 왼쪽 경계
    double obstacle_d_right = obstacle.d_center - obstacle.size; // obstacle의 오른쪽 경계 
    
    double left_gap = std::abs(waypoint.d_left - abs(obstacle_d_left));     // left 빈공간
    double right_gap = std::abs(waypoint.d_right - abs(obstacle_d_right));  // right 빈공간 (abs주기)
    double min_space = evasion_dist_ + spline_bound_mindist_; // 0.65 + 0.2
    
    //-------우선 한쪽만 넓으면 그쪽으로-------
    if (right_gap > min_space && left_gap < min_space) {
        double d_apex_right = obstacle_d_right - evasion_dist_;
        if (d_apex_right > 0) d_apex_right = 0;

        // ★ d_apex 및 해당 위치 트랙 경계 로그
        RCLCPP_INFO(this->get_logger(), "d_apex=%.3f, d_left=%.3f, d_right=%.3f",
                    d_apex_right, waypoint.d_left, waypoint.d_right);

        return std::make_pair("right", d_apex_right);

    } else if (left_gap > min_space && right_gap < min_space) {
        double d_apex_left = obstacle_d_left + evasion_dist_;
        if (d_apex_left < 0) d_apex_left = 0;

        // ★ d_apex 및 해당 위치 트랙 경계 로그
        RCLCPP_INFO(this->get_logger(), "d_apex=%.3f, d_left=%.3f, d_right=%.3f",
                    d_apex_left, waypoint.d_left, waypoint.d_right);

        return std::make_pair("left", d_apex_left);
    } else {
        //-------B. 양쪽 다 넓으면 더 많은 공간 선택-------
        if (right_gap > left_gap) {
            double d_apex_right = obstacle_d_right - evasion_dist_;
            if (d_apex_right > 0) d_apex_right = 0;

            // ★ d_apex 및 해당 위치 트랙 경계 로그
            RCLCPP_INFO(this->get_logger(), "d_apex=%.3f, d_left=%.3f, d_right=%.3f",
                        d_apex_right, waypoint.d_left, waypoint.d_right);

            return std::make_pair("right", d_apex_right);
        } else {
            double d_apex_left = obstacle_d_left + evasion_dist_;
            if (d_apex_left < 0) d_apex_left = 0;

            // ★ d_apex 및 해당 위치 트랙 경계 로그
            RCLCPP_INFO(this->get_logger(), "d_apex=%.3f, d_left=%.3f, d_right=%.3f",
                        d_apex_left, waypoint.d_left, waypoint.d_right);

            return std::make_pair("left", d_apex_left);
        }
    }
}

//=========================================================================================
// 함수: evasion trajectory (/local waypoints) 생성
// ---------1)SplineInterpolator 사용하여 spline 생성
// ---------2)d값 clip
// ---------3)Cartesian 변환
//=========================================================================================
spline_planner_dh_test::msg::OTWpntArray SplinePlanner::generate_evasion_trajectory(
    const ObstacleFrenet& closest_obstacle) {

    spline_planner_dh_test::msg::OTWpntArray local_waypoints;
    local_waypoints.header.stamp = this->get_clock()->now();
    local_waypoints.header.frame_id = "map";

    if (gb_max_idx_ <= 1 || gb_max_s_ <= 0.0) {
        return local_waypoints;
    }
    
    // Determine which side has more space
    auto [more_space, d_apex] = calculate_more_space(closest_obstacle);
    
    // 곡률 부호 합으로 outside 판단
    const int denom = std::max(1, gb_max_idx_ - 1);
    const double wpnt_dist = gb_max_s_ / static_cast<double>(denom);
    int gb_idx = 0;
    if (wpnt_dist > 1e-9) {
        gb_idx = static_cast<int>(std::floor(closest_obstacle.s_center / wpnt_dist));
        if (gb_idx < 0) gb_idx = 0;
        if (gb_idx >= gb_max_idx_) gb_idx = gb_max_idx_ - 1;
    }
    
    double kappa_sum = 0.0;
    for (int offset = -2; offset <= 2; ++offset) {
        int idx = gb_idx + offset;
        // 수동 wrap
        if (idx < 0) idx += gb_max_idx_;
        if (idx >= gb_max_idx_) idx -= gb_max_idx_;
        kappa_sum += global_waypoints_->wpnts[idx].kappa_radpm;
    }
    
    std::string outside = (kappa_sum < 0.0) ? "left" : "right";

    //===========================================================
    // 7개 apex 점 생성 (dst=apex 계열 파라미터)
    //===========================================================
    double s_apex = closest_obstacle.s_center;
    std::vector<double> spline_params = {
        pre_apex_0_, pre_apex_1_, pre_apex_2_, 0.0,
        post_apex_0_, post_apex_1_, post_apex_2_
    };
    
    std::vector<double> evasion_s, evasion_d;
    for (size_t i = 0; i < spline_params.size(); ++i) {
        double dst = spline_params[i];
        // 속도 기반 스케일
        dst *= std::clamp(1.0 + cur_vs_ / std::max(1e-3, gb_vmax_), 1.0, 1.5);
        
        // outside 추월 시 더 부드럽게
        double si = s_apex + ((outside == more_space) ? dst * 1.75 : dst);
        double di = (dst == 0.0) ? d_apex : 0.0;
        
        evasion_s.push_back(si);
        evasion_d.push_back(di);
    }
    
    //==========================================================================================
    // spline 보간 -> (trajectory_s, trajectory_d)
    //==========================================================================================
    SplineInterpolator spline(evasion_s, evasion_d);
    
    double spline_resolution = 0.1;
    std::vector<double> trajectory_s;
    for (double s = evasion_s.front(); s <= evasion_s.back(); s += spline_resolution) {
        trajectory_s.push_back(s);
    }
    
    auto trajectory_d = spline.interpolate(trajectory_s);
    
    // Clip d values to stay within [min(d_apex,0), max(d_apex,0)]
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
        
        //-----------------------------------------------------------------------
        // 가장 가까운 글로벌 웨이포인트 인덱스 (가장 가까운 글로벌 웨이포인트에서 속도 가져오기 위해)
        //-----------------------------------------------------------------------
        int gb_idx_local = 0;
        if (wpnt_dist > 1e-9) {
            gb_idx_local = static_cast<int>(std::floor(s / wpnt_dist));
            if (gb_idx_local < 0) gb_idx_local = 0;
            if (gb_idx_local >= gb_max_idx_) gb_idx_local = gb_max_idx_ - 1;
        }
        
        double base_velocity = global_waypoints_->wpnts[gb_idx_local].vx_mps;
        double velocity = (outside == more_space) ? base_velocity : base_velocity * 0.9;


        spline_planner_dh_test::msg::OTWpnt waypoint;
        waypoint.id = static_cast<int>(local_waypoints.waypoints.size());
        waypoint.x_m = cartesian.first;
        waypoint.y_m = cartesian.second;
        waypoint.s_m = s;
        waypoint.d_m = trajectory_d[i];
        waypoint.vx_mps = velocity;
        
        local_waypoints.waypoints.push_back(waypoint);
    }
    
    // 메타데이터
    local_waypoints.ot_side = more_space;
    local_waypoints.side_switch = (last_ot_side_ != more_space);
    local_waypoints.last_switch_time = last_switch_time_;
    
    if (local_waypoints.side_switch) {
        last_switch_time_ = this->get_clock()->now();
    }
    last_ot_side_ = more_space;
    
    return local_waypoints;
}


//=====================================================================================
// 시각화 부분들 
//=====================================================================================
visualization_msgs::msg::MarkerArray SplinePlanner::create_trajectory_markers(
    const spline_planner_dh_test::msg::OTWpntArray& local_waypoints) {
    
    visualization_msgs::msg::MarkerArray markers;
    
    if (local_waypoints.waypoints.empty()) {
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
    for (const auto& waypoint : local_waypoints.waypoints) {
        geometry_msgs::msg::Point point;
        point.x = waypoint.x_m;
        point.y = waypoint.y_m;
        point.z = 0.02;
        line_marker.points.push_back(point);
    }
    
    markers.markers.push_back(line_marker);
    
    // Add individual waypoint markers (cylinders) - show every 3rd point to avoid clutter
    for (size_t i = 0; i < local_waypoints.waypoints.size(); i += 3) {
        const auto& waypoint = local_waypoints.waypoints[i];
        
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.id = static_cast<int>(i + 1);
        marker.ns = "local_trajectory_points";
        
        marker.pose.position.x = waypoint.x_m;
        marker.pose.position.y = waypoint.y_m;
        marker.pose.position.z = waypoint.vx_mps / (std::max(1e-3, gb_vmax_) / 2.0);
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = 0.15;
        marker.scale.y = 0.15;
        marker.scale.z = std::max(0.1, waypoint.vx_mps / std::max(1e-3, gb_vmax_));
        
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

// 모든 장애물을 위한 MarkerArray 생성
visualization_msgs::msg::MarkerArray SplinePlanner::create_all_obstacles_markers(
    const std::vector<ObstacleFrenet>& obstacles) {
    
    visualization_msgs::msg::MarkerArray markers;
    
    for (size_t i = 0; i < obstacles.size(); ++i) {
        const auto& obstacle = obstacles[i];
        
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.id = static_cast<int>(i);
        marker.ns = "all_obstacles";
        
        // Convert obstacle center back to Cartesian
        auto cartesian = frenet_converter_->frenet_to_cartesian(obstacle.s_center, obstacle.d_center);
        
        marker.pose.position.x = cartesian.first;
        marker.pose.position.y = cartesian.second;
        marker.pose.position.z = 0.1;
        marker.pose.orientation.w = 1.0;
        
        // 크기를 장애물 size에 맞게 조정
        marker.scale.x = obstacle.size * 2.0;
        marker.scale.y = obstacle.size * 2.0;
        marker.scale.z = 0.3;
        
        // 장애물별로 다른 색상
        marker.color.a = 0.7;
        if (i % 3 == 0) {
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; // 빨강
        } else if (i % 3 == 1) {
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; // 초록
        } else {
            marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0; // 파랑
        }
        
        markers.markers.push_back(marker);
        
        // 장애물 ID 텍스트 마커 추가
        visualization_msgs::msg::Marker text_marker;
        text_marker.header.frame_id = "map";
        text_marker.header.stamp = this->get_clock()->now();
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.id = static_cast<int>(i + 1000);  // ID 겹치지 않게
        text_marker.ns = "obstacle_ids";
        
        text_marker.pose.position.x = cartesian.first;
        text_marker.pose.position.y = cartesian.second;
        text_marker.pose.position.z = 0.5;
        text_marker.pose.orientation.w = 1.0;
        
        text_marker.scale.z = 0.3;
        text_marker.color.a = 1.0;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        
        text_marker.text = "ID:" + std::to_string(obstacle.id) + 
                          "\ns=" + std::to_string(static_cast<int>(obstacle.s_center * 100) / 100.0) +
                          "\nd=" + std::to_string(static_cast<int>(obstacle.d_center * 100) / 100.0);
        
        markers.markers.push_back(text_marker);
    }
    
    return markers;
}

// 모든 장애물 publish
void SplinePlanner::publish_all_obstacles(const std::vector<ObstacleFrenet>& obstacles) {
    if (obstacles.empty()) return;
    
    auto markers = create_all_obstacles_markers(obstacles);
    all_obstacles_pub_->publish(markers);
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
//=====================================================================================
// 함수: yaw값 계산 (quaternion -> yaw) Utility Functions 
//=====================================================================================
double SplinePlanner::quaternion_to_yaw(double x, double y, double z, double w) {
    // Convert quaternion to yaw angle using atan2
    double siny_cosp = 2 * (w * z + x * y);
    double cosy_cosp = 1 - 2 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}

/*
//=====================================================================================
// 영향권 + 우선순위 기반 효율적 다중 장애물 회피 함수
// F1tenth 환경에 최적화: O(n) 연산량, 기존 spline 코드 재사용
//=====================================================================================
spline_planner_dh_test::msg::OTWpntArray SplinePlanner::generate_priority_based_evasion_trajectory(
    const std::vector<ObstacleFrenet>& obstacles) {
    
    if (obstacles.empty()) {
        spline_planner_dh_test::msg::OTWpntArray empty_trajectory;
        empty_trajectory.header.stamp = this->get_clock()->now();
        empty_trajectory.header.frame_id = "map";
        return empty_trajectory;
    }
    
    // 1단계: 우선순위 계산 및 최고 우선순위 장애물 선택 (O(n))
    auto primary_obstacle = select_primary_obstacle(obstacles);
    
    RCLCPP_INFO(this->get_logger(), 
        "주요 장애물 선택: ID=%d, s=%.2f, d=%.2f, 우선순위=%.3f",
        primary_obstacle.id, primary_obstacle.s_center, primary_obstacle.d_center, 
        primary_obstacle.priority);
    
    // 2단계: 주요 장애물로 기본 spline 생성 (기존 코드 재사용)
    auto base_trajectory = generate_evasion_trajectory(primary_obstacle);
    
    // 3단계: 영향권 내 다른 장애물들과 충돌 체크 및 조정 (O(k), k<<n)
    auto adjusted_trajectory = adjust_trajectory_for_influence_zone(base_trajectory, obstacles, primary_obstacle);
    
    RCLCPP_INFO(this->get_logger(), 
        "우선순위 기반 회피 완료: %zu개 waypoint, %zu개 장애물 고려",
        adjusted_trajectory.waypoints.size(), obstacles.size());
    
    return adjusted_trajectory;
}

//=====================================================================================
// 우선순위 계산 및 최고 우선순위 장애물 선택
//=====================================================================================
ObstacleFrenet SplinePlanner::select_primary_obstacle(const std::vector<ObstacleFrenet>& obstacles) {
    
    std::vector<ObstacleFrenet> obstacles_with_priority = obstacles;
    
    // 각 장애물의 우선순위 계산
    for (auto& obs : obstacles_with_priority) {
        double distance = std::fmod(obs.s_center - cur_s_ + gb_max_s_, gb_max_s_);
        double distance_weight = 1.0 / (distance + 0.1);  // 거리 역수 (가까울수록 높음)
        
        double threat_level = obs.size * 2.0;  // 크기가 클수록 위험
        
        double path_blocking_factor = 1.0 / (std::abs(obs.d_center) + 0.1);  // 경로에 가까울수록 높음
        
        obs.priority = distance_weight * threat_level * path_blocking_factor;
        
        RCLCPP_DEBUG(this->get_logger(), 
            "장애물 %d 우선순위: %.3f (거리가중치:%.2f, 위험도:%.2f, 경로차단:%.2f)",
            obs.id, obs.priority, distance_weight, threat_level, path_blocking_factor);
    }
    
    // 최고 우선순위 장애물 반환
    return *std::max_element(obstacles_with_priority.begin(), obstacles_with_priority.end(),
        [](const ObstacleFrenet& a, const ObstacleFrenet& b) {
            return a.priority < b.priority;
        });
}

//=====================================================================================
// 영향권 내 장애물들을 고려한 궤적 미세 조정
//=====================================================================================
spline_planner_dh_test::msg::OTWpntArray SplinePlanner::adjust_trajectory_for_influence_zone(
    const spline_planner_dh_test::msg::OTWpntArray& base_trajectory,
    const std::vector<ObstacleFrenet>& all_obstacles,
    const ObstacleFrenet& primary_obstacle) {
    
    auto adjusted_trajectory = base_trajectory;
    
    // 주요 장애물의 영향권 정의
    double influence_radius = primary_obstacle.size * 3.0 + 2.0;  // 안전 마진 포함
    double influence_s_min = primary_obstacle.s_center - influence_radius;
    double influence_s_max = primary_obstacle.s_center + influence_radius;
    
    RCLCPP_DEBUG(this->get_logger(), 
        "영향권 범위: s=[%.2f, %.2f], 반경=%.2f",
        influence_s_min, influence_s_max, influence_radius);
    
    // 영향권 내의 다른 장애물들 찾기
    std::vector<ObstacleFrenet> influence_obstacles;
    for (const auto& obs : all_obstacles) {
        if (obs.id != primary_obstacle.id &&
            obs.s_center >= influence_s_min && obs.s_center <= influence_s_max) {
            influence_obstacles.push_back(obs);
        }
    }
    
    if (influence_obstacles.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "영향권 내 추가 장애물 없음");
        return adjusted_trajectory;
    }
    
    RCLCPP_INFO(this->get_logger(), 
        "영향권 내 추가 장애물 %zu개 발견, 궤적 미세 조정",
        influence_obstacles.size());
    
    // 궤적 점들과 영향권 장애물들 간의 충돌 체크 및 조정
    for (auto& waypoint : adjusted_trajectory.waypoints) {
        for (const auto& obs : influence_obstacles) {
            double dist_to_obs = std::sqrt(
                std::pow(waypoint.s_m - obs.s_center, 2) + 
                std::pow(waypoint.d_m - obs.d_center, 2)
            );
            
            double safety_distance = obs.size + 0.5;  // 안전 거리
            
            if (dist_to_obs < safety_distance) {
                // 충돌 시 d 좌표 조정 (간단한 repulsion)
                double push_direction = (waypoint.d_m > obs.d_center) ? 1.0 : -1.0;
                double adjustment = (safety_distance - dist_to_obs) * push_direction;
                waypoint.d_m += adjustment;
                
                // d 좌표 범위 제한
                waypoint.d_m = std::clamp(waypoint.d_m, -3.0, 3.0);
                
                // Cartesian 좌표 재계산
                auto cartesian = frenet_converter_->frenet_to_cartesian(waypoint.s_m, waypoint.d_m);
                waypoint.x_m = cartesian.first;
                waypoint.y_m = cartesian.second;
                
                RCLCPP_DEBUG(this->get_logger(), 
                    "충돌 회피 조정: s=%.2f, d=%.2f->%.2f",
                    waypoint.s_m, waypoint.d_m - adjustment, waypoint.d_m);
            }
        }
    }
    
    return adjusted_trajectory;
}
*/


//=====================================================================================
// Frenet 디버깅 정보 publish 함수
//=====================================================================================
void SplinePlanner::publish_frenet_debug_info(const std::vector<ObstacleFrenet>& obstacles) {
    spline_planner_dh_test::msg::FrenetDebugArray debug_array;
    debug_array.header.stamp = this->get_clock()->now();
    debug_array.header.frame_id = "map";
    
    // Ego vehicle 정보 추가
    if (ego_odom_) {
        spline_planner_dh_test::msg::FrenetDebug ego_debug;
        ego_debug.id = 0;  // ego는 ID 0으로 설정
        ego_debug.object_type = "ego";
        
        // Cartesian 좌표
        ego_debug.x_cartesian = ego_odom_->pose.pose.position.x;
        ego_debug.y_cartesian = ego_odom_->pose.pose.position.y;
        
        // Frenet 좌표
        ego_debug.s_frenet = cur_s_;
        ego_debug.d_frenet = cur_d_;
        
        // 속도 정보
        ego_debug.vx_cartesian = ego_odom_->twist.twist.linear.x;
        ego_debug.vy_cartesian = ego_odom_->twist.twist.linear.y;
        ego_debug.vs_frenet = cur_vs_;
        ego_debug.vd_frenet = cur_vd_;
        
        ego_debug.yaw = ego_yaw_;
        ego_debug.size = 0.5;  // ego vehicle size estimate
        
        debug_array.objects.push_back(ego_debug);
    }
    
    // 장애물 정보 추가
    for (const auto& obs : obstacles) {
        spline_planner_dh_test::msg::FrenetDebug obs_debug;
        obs_debug.id = obs.id;
        obs_debug.object_type = "obstacle";
        
        // Cartesian 좌표 (frenet에서 역변환)
        auto cartesian = frenet_converter_->frenet_to_cartesian(obs.s_center, obs.d_center);
        obs_debug.x_cartesian = cartesian.first;
        obs_debug.y_cartesian = cartesian.second;
        
        // Frenet 좌표
        obs_debug.s_frenet = obs.s_center;
        obs_debug.d_frenet = obs.d_center;
        
        // 속도 정보
        obs_debug.vs_frenet = obs.vs;
        obs_debug.vd_frenet = obs.vd;
        
        obs_debug.yaw = obs.yaw;
        obs_debug.size = obs.size;
        
        debug_array.objects.push_back(obs_debug);
    }
    
    frenet_debug_pub_->publish(debug_array);
    
    // 콘솔에도 간단한 정보 출력
    RCLCPP_INFO(this->get_logger(), 
        "=== FRENET DEBUG INFO ===");
    RCLCPP_INFO(this->get_logger(), 
        "EGO: s=%.2f, d=%.2f (x=%.2f, y=%.2f)", 
        cur_s_, cur_d_, 
        ego_odom_ ? ego_odom_->pose.pose.position.x : 0.0,
        ego_odom_ ? ego_odom_->pose.pose.position.y : 0.0);
    
    for (const auto& obs : obstacles) {
        auto cartesian = frenet_converter_->frenet_to_cartesian(obs.s_center, obs.d_center);
        RCLCPP_INFO(this->get_logger(), 
            "OBS[%d]: s=%.2f, d=%.2f (x=%.2f, y=%.2f)", 
            obs.id, obs.s_center, obs.d_center, cartesian.first, cartesian.second);
    }
    RCLCPP_INFO(this->get_logger(), "========================");
}
