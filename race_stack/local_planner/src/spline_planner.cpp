#include "spline_planner_dh_test/spline_planner.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <functional>
#include <memory>

SplinePlanner::SplinePlanner() : Node("spline_planner_node"),
    cur_s_(0.0), 
    cur_d_(0.0), 
    cur_vs_(0.0),
    cur_vd_(0.0),
    lookahead_(3.0), //10은 너무 멀어 -> 6로 바꿔서 ㄱㄱ 
    evasion_dist_(0.3), //파라미터 0.65 -> 0.6 -> 0.55-> 0.50 -> 0.30 바꾸면서 실험해보기 
    obs_traj_thresh_(1.0), //파라미터 1.0 -> 0.5로 바꾸기 
    spline_bound_mindist_(0.2), //파라미터 0.2 -> 0.1로 바꾸기 -> 다시 0.2로 바꾸기
    gb_max_s_(0.0), 
    gb_max_idx_(0),    // <- 이제 "웨이포인트 개수"로 사용
    gb_vmax_(0.0),
    // [수정] 멤버 선언 순서에 맞춰 초기화 순서 정렬 + 신규 파라미터 초기화
    obs_outside_reject_margin_(0.05),      // [수정] 트랙 바깥 포인트 무시 허용 오차
    obs_front_shift_gain_(1.0),            // [수정] '앞면 측정' 보정 게인( size * 0.6 만큼 s를 뒤로 )
    pre_apex_0_(-1.0), 
    pre_apex_1_(-0.7), 
    pre_apex_2_(-0.5),
    post_apex_0_(1.0), 
    post_apex_1_(1.2), 
    post_apex_2_(1.5),
    last_ot_side_(""), 
    last_switch_time_(this->get_clock()->now()) {
    
//====================================================================================
// subscribers, publishers 정의 
//====================================================================================
    //--------------------------------subscribers-------------------------------------
    obstacles_sub_ = this->create_subscription<f110_msgs::msg::ObstacleArray>(
        "/obstacles", 10,
        std::bind(&SplinePlanner::obstacles_callback, this, std::placeholders::_1));
    
    ego_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/car_state/odom", 10,
        std::bind(&SplinePlanner::ego_odom_callback, this, std::placeholders::_1));

    global_waypoints_sub_ = this->create_subscription<f110_msgs::msg::WpntArray>(
        "/global_waypoints", 10,
        std::bind(&SplinePlanner::global_waypoints_callback, this, std::placeholders::_1));
    

    //-------------------------------publishers-------------------------------------
    local_trajectory_pub_ = this->create_publisher<f110_msgs::msg::WpntArray>(
        "/local_waypoints", 10);
    
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planner/avoidance/markers", 10);
    
    closest_obs_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/planner/avoidance/considered_OBS", 10);
    
    // 모든 장애물 시각화용 publisher 추가
    all_obstacles_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planner/all_obstacles", 10);
    
    reference_path_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planner/reference_path", 10);
    
    // frenet_debug_pub_ = this->create_publisher<f110_msgs::msg::FrenetDebugArray>(
    //     "/planner/frenet_debug", 10);
    
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
void SplinePlanner::obstacles_callback(const f110_msgs::msg::ObstacleArray::SharedPtr msg) {
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
void SplinePlanner::global_waypoints_callback(const f110_msgs::msg::WpntArray::SharedPtr msg) {
    global_waypoints_ = msg;

    //* 부호 정리하자 : map 경계 d_right, d_left가 들어올 때 d_right이 -가 아닌 +로 들어오니 이걸 들어오자 마자 -로 강제 시키자 
    for (auto& wp : global_waypoints_->wpnts) {
        double L = std::abs(wp.d_left);
        double R = std::abs(wp.d_right);
        wp.d_left = L; // +값
        wp.d_right = -R; // -값 
    }
    
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
                "Waypoint %zu: id=%d | s=%.3f, d=%.3f | x=%.3f, y=%.3f | d_right=%.3f, d_left=%.3f | psi=%.4f, kappa=%.6f, vx=%.3f, ax=%.3f",
                i,
                static_cast<int>(wp.id),
                wp.s_m, wp.d_m,
                wp.x_m, wp.y_m,
                wp.d_right, wp.d_left,
                wp.psi_rad, wp.kappa_radpm,
                wp.vx_mps, wp.ax_mps2);
        }
    }
}


void SplinePlanner::timer_callback() {
    process_spline_planning();
}

//===========================================================================================
//함수 (제어에서 사용할 수 있게 장애물이 있을 때만 /local_waypoints를 publish) 없을 때는 publish 하지 않음
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

    visualization_msgs::msg::MarkerArray markers_msg;
    bool publish_local = false;
    f110_msgs::msg::WpntArray trajectory_msg;


    // ========================================================================================
    // 장애물이 있으면 frenet 좌표계로 변환하고, 가까운 장애물 filtering, 가장 가까운 장애물 선택 후 회피 경로 생성
    // ========================================================================================
    if (obstacles_) {
        // 장애물 Frenet 좌표계로 변환
        auto frenet_obstacles = convert_obstacles_to_frenet(*obstacles_);
        
        // Frenet 디버깅 정보 publish
        // publish_frenet_debug_info(frenet_obstacles);
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

            if (!trajectory_msg.wpnts.empty()) {
                // ----------[수정] 회피 스플라인 뒤를 글로벌로 채워 80개 보장----------
                append_forward_points_decay(trajectory_msg, 80);
                publish_local = true;
            } else {
                RCLCPP_WARN(this->get_logger(), "회피 경로 생성 실패 - 장애물 회피 불가");
            }
            
            // ----------evasion_trajectory 시각화----------------
            markers_msg = create_trajectory_markers(trajectory_msg);
            
            // ----------가장 가까운 장애물 시각화----------------
            auto obs_marker = create_obstacle_marker(closest_obstacle);
            closest_obs_pub_->publish(obs_marker);
            
            RCLCPP_DEBUG(this->get_logger(), 
                "[Local Planning ON] publishing /local_waypoints with %zu points", 
                trajectory_msg.wpnts.size());
        } else {

            std::vector<ObstacleFrenet> empty_obstacles;
            // publish_frenet_debug_info(empty_obstacles);

            // [수정] 가까운 장애물이 없어도 항상 로컬 80개 발행(글로벌 forward 변환)
            trajectory_msg = this->build_forward_global_trajectory(80);
            publish_local = true;

            // 상태 마커
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
            
            //status_marker.text = "SPLINE PLANNER: NO CLOSE OBSTACLES (LOCAL=GLOBAL 80)";
            markers_msg.markers.push_back(status_marker);

            auto line = create_trajectory_markers(trajectory_msg);
            markers_msg.markers.insert(markers_msg.markers.end(), line.markers.begin(), line.markers.end());
        }
    } else {

        std::vector<ObstacleFrenet> empty_obstacles;
        // publish_frenet_debug_info(empty_obstacles);

        // [수정] 장애물 데이터가 없어도 글로벌 기반 80포인트 로컬 발행
        trajectory_msg = this->build_forward_global_trajectory(80);
        publish_local = true;
        
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
        //status_marker.text = "SPLINE PLANNER: WAITING FOR OBSTACLE DATA (LOCAL=GLOBAL 80)";
        markers_msg.markers.push_back(status_marker);

        auto line = create_trajectory_markers(trajectory_msg);
        markers_msg.markers.insert(markers_msg.markers.end(), line.markers.begin(), line.markers.end());
    }
    
    // Publish results
    if (publish_local){
        local_trajectory_pub_->publish(trajectory_msg);
    }
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
    const f110_msgs::msg::ObstacleArray& obstacles) {
    
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
        double s_raw = center_frenet.first;
        double d_raw = center_frenet.second;

        // [수정] ----- '앞면 측정'을 고려한 s 보정 -----
        //  센서가 장애물 앞면을 찍은 경우, 실제 중심/안전 경계가 더 앞쪽에 있음.
        //  경로가 뒤로 빨려들어 장애물을 긁지 않게 s를 size만큼 "뒤(-)"로 이동시켜 여유를 확보한다.
        //  게인 obs_front_shift_gain_ 으로 튜닝 가능(기본 1.0).
        double s_adj = s_raw - obs_front_shift_gain_ * frenet_obs.size;

        // s wrap
        if (gb_max_s_ > 0) {
            s_adj = std::fmod(s_adj + gb_max_s_, gb_max_s_);
        }

        // [수정] 보정된 s로 인덱스 계산(psi, 경계 판단 일관성 유지)
        int gb_idx = 0;
        if (wpnt_dist > 1e-9) {
            gb_idx = static_cast<int>(std::floor(s_adj / wpnt_dist));
            if (gb_idx < 0) gb_idx = 0;
            if (gb_idx >= gb_max_idx_) gb_idx = gb_max_idx_ - 1;
        }

        // [수정] 트랙 바깥 포인트는 초기에 제거 (벽/코너 잡음 제거)
        const auto& wpr = global_waypoints_->wpnts[gb_idx];
        double d_left  = wpr.d_left;
        double d_right = wpr.d_right;
        if ( (d_raw > d_left  + obs_outside_reject_margin_) ||
             (d_raw < d_right - obs_outside_reject_margin_) ) {
            // 트랙을 명백히 벗어난 장애물 → 무시
            continue;
        }

        // 해당 위치에서의 트랙 방향(psi)로 속도 성분 변환
        double track_psi = wpr.psi_rad;
        frenet_obs.vs = obs.vx * std::cos(track_psi) + obs.vy * std::sin(track_psi);
        frenet_obs.vd = -obs.vx * std::sin(track_psi) + obs.vy * std::cos(track_psi);

        // 최종 적용
        frenet_obs.s_center = s_adj;
        frenet_obs.d_center = d_raw;
        
        frenet_obstacles.push_back(frenet_obs);
        RCLCPP_INFO(this->get_logger(), 
            "장애물 %d: Frenet(s_raw=%.2f -> s_adj=%.2f, d=%.2f), 속도(vs=%.2f, vd=%.2f), size=%.2f",
            frenet_obs.id, s_raw, frenet_obs.s_center, frenet_obs.d_center, 
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
// 함수: calculate_more_space
// 장애물의 위치에 따라 left, right gap 계산하고, 더 많은 공간 선택 
// d_apex 계산
//===========================================================================================
std::pair<std::string, double>
SplinePlanner::calculate_more_space(const ObstacleFrenet& obstacle) {
    // 필수 체크
    if (gb_max_idx_ <= 1 || gb_max_s_ <= 0.0) {
        return {"right", 0.0};
    }

    // 공통 파라미터(일관화)
    const double ego_width  = 0.30;     // 차량 폭 (일관!)
    const double ego_half   = 0.5 * ego_width;
    const double sigma_d    = 0.03;
    const double k_sigma    = 2.0;      // 소형맵 튠: 불확실성 2σ
    const double k_kappa    = 0.02;
    const double k_v        = 0.01;
    const double graze_clearance = 0.05;
    const double wall_guard = 0.08;     // 벽 하드가드(조금 키움)

    // s→가까운 글로벌 인덱스
    const int denom = std::max(1, gb_max_idx_ - 1);
    const double wpnt_dist = gb_max_s_ / static_cast<double>(denom);
    int gb_idx = 0;
    if (wpnt_dist > 1e-9) {
        gb_idx = static_cast<int>(std::floor(obstacle.s_center / wpnt_dist));
        gb_idx = std::clamp(gb_idx, 0, gb_max_idx_-1);
    }
    const auto& wp = global_waypoints_->wpnts[gb_idx];

    // 트랙 코리도어
    const double track_width = wp.d_left - wp.d_right;
    const double eps = 1e-3;
    const double eff_margin = std::max(0.0, std::min(spline_bound_mindist_, 0.5*track_width - eps));
    const double d_min_bound = wp.d_right + eff_margin;
    const double d_max_bound = wp.d_left  - eff_margin;

    // 장애물 경계
    const double d_obs = obstacle.d_center;
    const double r_obs = obstacle.size;
    const double obs_left  = d_obs + r_obs;
    const double obs_right = d_obs - r_obs;

    // 좌/우 여유(양수 기대)
    const double left_gap  = d_max_bound - obs_left;
    const double right_gap = obs_right   - d_min_bound;

    // 최소 확보 여유(동특성 가중)
    const double buffer_dyn =
        graze_clearance +
        k_kappa * std::abs(wp.kappa_radpm) +
        k_v     * std::abs(cur_vs_) +
        k_sigma * sigma_d;

    // 기하 한계
    const double max_delta_left_geom  = d_max_bound - (d_obs + r_obs);
    const double max_delta_right_geom = (d_obs - r_obs) - d_min_bound;

    // 사용가능 여부
    bool feasL = (max_delta_left_geom  >= ego_half);
    bool feasR = (max_delta_right_geom >= ego_half);

    // 사이드 선택: 가능한 쪽 중 '더 큰 여유' 우선
    std::string side = "right";
    if      (feasL && !feasR) side = "left";
    else if (!feasL && feasR) side = "right";
    else if (feasL && feasR)  side = (left_gap >= right_gap) ? "left" : "right";
    else                      side = (left_gap >= right_gap) ? "left" : "right"; // 둘 다 불가→큰 쪽으로 포화

    // 목표 여유 (센터-엣지)
    double delta_goal =
        ego_half + buffer_dyn; // 차량 반폭 + 동적버퍼

    // 글로벌 캡: 가용 반폭의 일부만 사용(너무 크게 틀지 않도록)
    const double avail_half =
        0.5*track_width - eff_margin - r_obs; // 모서리~경계 최대 center-edge
    const double alpha_cap = 0.45;
    double cap_global = std::max(ego_half, alpha_cap * std::max(0.0, avail_half));

    // 사이드별 상한
    const double max_side = (side=="left") ? max_delta_left_geom : max_delta_right_geom;
    double upper_cap = std::min(cap_global, max_side - 1e-3);
    if (upper_cap < 0.0) upper_cap = 0.0;

    // 최종 delta
    const double delta_use = std::clamp(delta_goal, ego_half, upper_cap);

    // d_apex
    double d_apex = (side=="left")
                    ? (obs_left  + delta_use)
                    : (obs_right - delta_use);

    // 하드 클램프(+벽가드)
    d_apex = std::clamp(d_apex, d_min_bound + wall_guard, d_max_bound - wall_guard);

    // 최종 로그(단일 경로만 출력)
    RCLCPP_INFO(this->get_logger(),
        "[apex] side=%s gapL=%.3f gapR=%.3f tw=%.3f effM=%.3f "
        "delta=%.3f (goal=%.3f capG=%.3f maxSide=%.3f) d_apex=%.3f",
        side.c_str(), left_gap, right_gap, track_width, eff_margin,
        delta_use, delta_goal, cap_global, max_side, d_apex);

    return {side, d_apex};
}


//=========================================================================================
// 함수: evasion trajectory (/local waypoints) 생성
// ---------1)SplineInterpolator 사용하여 spline 생성
// ---------2)d값 clip
// ---------3)Cartesian 변환
//=========================================================================================
f110_msgs::msg::WpntArray SplinePlanner::generate_evasion_trajectory(
    const ObstacleFrenet& closest_obstacle) {

    f110_msgs::msg::WpntArray local_waypoints;
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

    // [추가] 급조향 방지: Δd/Δs 상한 + apex 주변 램프 분배
    const double slope_beta = std::clamp(0.45 + 0.02 * cur_vs_, 0.35, 0.70);      // m/m (0.25~0.45 권장)
    const double stretch_outside = 1.3; // 기존 outside 보정 유지
    const double min_L = 0.20;           // 앞/뒤 최소 유효거리
    const double stretch_inside = 1.15;

    // [추가] 먼저 s 위치들 계산(속도 스케일 + outside 보정)
    std::vector<double> tmp_s; tmp_s.reserve(spline_params.size());
    for (double dst : spline_params) {
        double dst_scaled = dst * std::clamp(1.0 + cur_vs_ / std::max(1e-3, gb_vmax_), 1.0, 1.5);
        double si = s_apex + ((outside == more_space) ? dst_scaled * stretch_outside
                                                      : dst_scaled * stretch_inside);
        tmp_s.push_back(si);
    }
    // 인덱스: 0,1,2,(3=apex),4,5,6
    const double L_in  = std::max(min_L, std::abs(tmp_s[3] - tmp_s[2]));
    const double L_out = std::max(min_L, std::abs(tmp_s[4] - tmp_s[3]));

    // [추가] calculate_more_space()에서 받은 d_apex에 기울기 제한 적용
    const double d_apex_sign = (d_apex >= 0.0) ? 1.0 : -1.0;
    const double d_allowed_by_slope = slope_beta * std::min(L_in, L_out);
    const double d_apex_limited = d_apex_sign * std::min(std::abs(d_apex), d_allowed_by_slope);

    // [추가] 램프 가중치(부드럽게 상승/하강)
    //const double w[7] = {0.25, 0.45, 0.80, 1.30, 1.00, 0.65, 0.45};
    //const double w[7] = {0.25, 0.45, 0.80, 1.10, 0.80, 0.45, 0.25};
    const double w[7] = {0.25, 0.45, 0.80, 1.0, 0.80, 0.45, 0.25};

    // [수정] s,d 점 생성: apex 이웃도 0이 아니라 w[i]*apex로 채워서 급변 억제
    evasion_s.clear(); evasion_d.clear();
    for (size_t i = 0; i < spline_params.size(); ++i) {
        double dst = spline_params[i];
        double dst_scaled = dst * std::clamp(1.0 + cur_vs_ / std::max(1e-3, gb_vmax_), 1.0, 1.5);
        double si = s_apex + ((outside == more_space) ? dst_scaled * stretch_outside
                                                      : dst_scaled * stretch_inside);
        double di = w[i] * d_apex_limited; // [중요] 이웃 점에도 d를 분배
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
    
    //trajectory_d 클램핑해서 트랙밖으로 안나가게 설정하기 
    for(size_t i=0; i<trajectory_d.size(); ++i){
        //s를 0~gb_max_s_로 래핑 
        double s_local = std::fmod(trajectory_s[i], gb_max_s_);
        if (s_local < 0) s_local += gb_max_s_;

        //각 s에서 가장 가까운 Global Waypoint 인덱스 
        int gb_idx_local = 0;
        if (wpnt_dist > 1e-9) {
            gb_idx_local = static_cast<int>(std::floor(s_local/wpnt_dist));
            gb_idx_local = std::clamp(gb_idx_local, 0, gb_max_idx_ - 1);
        }

        const auto& wp = global_waypoints_->wpnts[gb_idx_local];

        // 트랙 폭 기반 eff_margin(유효한 마진) 계산
        const double track_width = wp.d_left - wp.d_right; // 양수
        const double eps = 1e-3; // 아주 작은 값
        const double eff_margin = std::max(0.0, std::min(spline_bound_mindist_, 0.5*track_width - eps));

        // [추가] 벽 하드가드로 한 칸 더 안쪽으로
        const double wall_guard = 0.03; // 3 cm

        const double d_min = wp.d_right + eff_margin + wall_guard; // [수정]
        const double d_max = wp.d_left  - eff_margin - wall_guard; // [수정]

        trajectory_d[i] = std::clamp(trajectory_d[i], d_min, d_max);
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


        f110_msgs::msg::Wpnt waypoint;
        waypoint.id = static_cast<int>(local_waypoints.wpnts.size());
        waypoint.x_m = cartesian.first;
        waypoint.y_m = cartesian.second;
        waypoint.s_m = s;
        waypoint.d_m = trajectory_d[i];
        waypoint.vx_mps = velocity;
        waypoint.d_right = 0.0;
        waypoint.d_left  = 0.0;
        waypoint.psi_rad     = 0.0;      // [F][추가] 제어용 헤딩(rad)
        waypoint.kappa_radpm = 0.0;      // [F][추가] 제어용 곡률(rad/m)
        waypoint.ax_mps2     = 0.0;

        local_waypoints.wpnts.push_back(waypoint);
    }

    // 메타데이터 (메시지에 담지 않음: 내부 상태만 갱신)
    bool side_switched = (last_ot_side_ != more_space);
    if (side_switched) {
        last_switch_time_ = this->get_clock()->now();
    }
    last_ot_side_ = more_space;
    
    return local_waypoints;
}


//=====================================================================================
// 시각화 부분들 
//=====================================================================================
visualization_msgs::msg::MarkerArray SplinePlanner::create_trajectory_markers(
    const f110_msgs::msg::WpntArray& local_waypoints) {
    
    visualization_msgs::msg::MarkerArray markers;
    
    if (local_waypoints.wpnts.empty()) {
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
    for (const auto& waypoint : local_waypoints.wpnts) {
        geometry_msgs::msg::Point point;
        point.x = waypoint.x_m;
        point.y = waypoint.y_m;
        point.z = 0.02;
        line_marker.points.push_back(point);
    }
    
    markers.markers.push_back(line_marker);
    
    // Add individual waypoint markers (cylinders) - show every 3rd point to avoid clutter
    for (size_t i = 0; i < local_waypoints.wpnts.size(); i += 3) {
        const auto& waypoint = local_waypoints.wpnts[i];
        
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

//=====================================================================================
// [추가] 글로벌 웨이포인트로부터 "현재 cur_s_ 기준 앞으로 target_count개" 로컬 포맷 생성
//=====================================================================================
f110_msgs::msg::WpntArray SplinePlanner::build_forward_global_trajectory(int target_count) {
    f110_msgs::msg::WpntArray out;
    out.header.stamp = this->get_clock()->now();
    out.header.frame_id = "map";

    if (!global_waypoints_ || global_waypoints_->wpnts.empty() || gb_max_idx_ <= 0) {
        return out;
    }

    const int denom = std::max(1, gb_max_idx_ - 1);
    const double wpnt_dist = gb_max_s_ / static_cast<double>(denom);

    // cur_s_에 가장 가까운 글로벌 인덱스
    int start_idx = 0;
    if (wpnt_dist > 1e-9) {
        start_idx = static_cast<int>(std::floor(cur_s_ / wpnt_dist));
        start_idx = std::clamp(start_idx, 0, gb_max_idx_ - 1);
    }

    for (int k = 0; k < target_count; ++k) {
        int idx = (start_idx + k) % gb_max_idx_;
        const auto& g = global_waypoints_->wpnts[idx];

        f110_msgs::msg::Wpnt wp;
        wp.id = static_cast<int>(out.wpnts.size());
        wp.x_m = g.x_m;
        wp.y_m = g.y_m;
        wp.s_m = g.s_m;
        wp.d_m = g.d_m;
        wp.vx_mps = g.vx_mps;
        wp.d_right = g.d_right;
        wp.d_left  = g.d_left;
        wp.psi_rad = g.psi_rad;           // [F][추가]
        wp.kappa_radpm = g.kappa_radpm;   // [F][추가]
        wp.ax_mps2 = g.ax_mps2;

        out.wpnts.push_back(wp);
    }

    return out;
}

/*
//=====================================================================================
// 영향권 + 우선순위 기반 효율적 다중 장애물 회피 함수
// (필요 시 복구 가능: 현재는 미사용 블록)
//=====================================================================================
// ...
*/


//=====================================================================================
// Frenet 디버깅 정보 publish 함수
//=====================================================================================
// void SplinePlanner::publish_frenet_debug_info(const std::vector<ObstacleFrenet>& obstacles) {
//     // (삭제됨) Frenet debug 퍼블리시 사용 안 함
// }

void SplinePlanner::append_forward_points_decay(f110_msgs::msg::WpntArray& traj,
                                                int target_count,
                                                double /*hold_s*/,
                                                double /*decay_s*/) {
    if (!global_waypoints_ || global_waypoints_->wpnts.empty() || gb_max_idx_ <= 0) return;

    // 이미 충분하면 종료
    if ((int)traj.wpnts.size() >= target_count) return;

    // 이어붙일 시작 s
    double start_s = cur_s_;
    if (!traj.wpnts.empty()) {
        start_s = traj.wpnts.back().s_m;
    }

    const int denom = std::max(1, gb_max_idx_ - 1);
    const double wpnt_dist = gb_max_s_ / static_cast<double>(denom);

    // start_s 다음 글로벌 인덱스
    int start_idx = 0;
    if (wpnt_dist > 1e-9) {
        start_idx = static_cast<int>(std::floor(start_s / wpnt_dist)) + 1;
        // 래핑
        while (start_idx >= gb_max_idx_) start_idx -= gb_max_idx_;
        if (start_idx < 0) start_idx = 0;
    }

    while ((int)traj.wpnts.size() < target_count) {
        const auto& g = global_waypoints_->wpnts[start_idx];

        f110_msgs::msg::Wpnt wp;
        wp.id = static_cast<int>(traj.wpnts.size());
        wp.x_m = g.x_m;
        wp.y_m = g.y_m;
        wp.s_m = g.s_m;
        wp.d_m = g.d_m;        // [수정] 글로벌 d로 채움
        wp.vx_mps = g.vx_mps;
        wp.d_right = g.d_right;
        wp.d_left  = g.d_left;
        wp.psi_rad = g.psi_rad;           // [F][추가]
        wp.kappa_radpm = g.kappa_radpm;   // [F][추가]
        wp.ax_mps2 = g.ax_mps2;

        traj.wpnts.push_back(wp);

        start_idx = (start_idx + 1) % gb_max_idx_;
    }
}
