#ifndef SPLINE_PLANNER_HPP
#define SPLINE_PLANNER_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "f110_msgs/msg/wpnt_array.hpp"        // [수정] f110_msgs 네임스페이스로 통일
#include "f110_msgs/msg/wpnt.hpp"              // [수정]
#include "f110_msgs/msg/obstacle_array.hpp"    // [수정]
#include "spline_planner_dh_test/frenet_converter.hpp"
#include "spline_planner_dh_test/spline_interpolator.hpp"

#include <memory>
#include <vector>
#include <string>

/**
 * @brief 장애물을 Frenet 좌표계에서 다루기 위한 구조체
 */
struct ObstacleFrenet {
    int    id;
    double s_center, d_center;  // 중심점 Frenet 좌표
    double vs, vd;              // Frenet 속도 성분
    double yaw;                 // 장애물 자세
    double size;                // 반경 또는 반폭
    double priority = 0.0;      // 우선순위(미사용, 확장용)
};

/**
 * @brief Spline 기반 로컬 회피/추종 플래너
 *
 * 핵심 변경점(이 버전):
 *  - 글로벌(d=0) 경로로의 강제 Fallback 제거 → 장애물 옆으로 빨려들지 않음
 *  - 로컬 경로 생성 후, 부족한 포인트는 '로컬 d 유지→완만 감쇠' 방식으로 패딩하여 총 80개 보장
 *  - RViz 상태 텍스트 마커(“NO CLOSE OBSTACLES” 등) 생성 제거
 *  - [신규] 트랙 바운더리 바깥(벽 너머)의 장애물은 변환 단계에서 즉시 무시
 */
class SplinePlanner : public rclcpp::Node {
public:
    SplinePlanner();

private:
    // ==============================
    // ROS Interfaces (Subscribers)
    // ==============================
    rclcpp::Subscription<f110_msgs::msg::ObstacleArray>::SharedPtr obstacles_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       ego_odom_sub_;
    rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr     global_waypoints_sub_;

    // ==============================
    // ROS Interfaces (Publishers)
    // ==============================
    rclcpp::Publisher<f110_msgs::msg::WpntArray>::SharedPtr            local_trajectory_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr      closest_obs_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr all_obstacles_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr reference_path_pub_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    
    // ==============================
    // Data (latest buffers)
    // ==============================
    f110_msgs::msg::ObstacleArray::SharedPtr obstacles_;
    nav_msgs::msg::Odometry::SharedPtr       ego_odom_;
    f110_msgs::msg::WpntArray::SharedPtr     global_waypoints_;
    std::unique_ptr<FrenetConverter>         frenet_converter_;
    
    // 현재 상태 (Frenet)
    double cur_s_, cur_d_, cur_vs_, cur_vd_;
    double ego_yaw_;
    
    // ==============================
    // Parameters / track meta
    // ==============================
    double lookahead_;             // 전방 탐색 거리 (s)
    double evasion_dist_;          // (참조) 회피 기본거리 튜닝 용
    double obs_traj_thresh_;       // d 임계값(트랙 중심선 대비 장애물 고려 범위)
    double spline_bound_mindist_;  // 경계 마진 기본값
    double gb_max_s_;              // 전체 s 길이
    int    gb_max_idx_;            // 글로벌 웨이포인트 개수
    double gb_vmax_;               // 글로벌 최대 속도

    // [신규] 바운더리 바깥 장애물 무시 허용 오차 (m)
    //  d < d_right - margin  또는  d > d_left + margin 이면 리스트에서 제거
    double obs_outside_reject_margin_;

    // [수정] 센서가 장애물 ‘앞면’을 찍었을 때 s를 size 만큼 뒤로 당기는 보정 게인
    double obs_front_shift_gain_;  // 기본 1.0 (cpp에서 초기화)
    
    // 스플라인 샘플링 오프셋들 (apex 전/후 s 오프셋)
    double pre_apex_0_, pre_apex_1_, pre_apex_2_;
    double post_apex_0_, post_apex_1_, post_apex_2_;
    
    // 사이드 전환 메타
    std::string  last_ot_side_;
    rclcpp::Time last_switch_time_;

    // ==============================
    // Callbacks
    // ==============================
    void obstacles_callback(const f110_msgs::msg::ObstacleArray::SharedPtr msg);
    void ego_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void global_waypoints_callback(const f110_msgs::msg::WpntArray::SharedPtr msg);

    // ==============================
    // Main loop
    // ==============================
    void timer_callback();
    void process_spline_planning();
    
    // ==============================
    // Helpers
    // ==============================
    bool wait_for_messages();                 // 필수 메시지 대기
    void initialize_converter();              // Frenet 변환기 초기화

    // 좌표 변환/필터
    std::vector<ObstacleFrenet> convert_obstacles_to_frenet(const f110_msgs::msg::ObstacleArray& obstacles);
    std::vector<ObstacleFrenet> filter_close_obstacles(const std::vector<ObstacleFrenet>& obstacles);

    // 회피 side & d_apex 산출
    std::pair<std::string, double> calculate_more_space(const ObstacleFrenet& obstacle);

    // 로컬 회피 경로 생성 (+ 총 80개 포인트 보장)
    f110_msgs::msg::WpntArray generate_evasion_trajectory(const ObstacleFrenet& closest_obstacle);

    /**
     * @brief 로컬 경로 뒤를 '로컬 d 유지(hold_s)' 후 '감쇠(decay_s)'시키며 0으로 서서히 회귀.
     *        글로벌 d를 강제로 넣지 않고, 안전한 클램프만 적용.
     * @param traj         in/out 경로
     * @param target_count 목표 포인트 수(기본 80)
     * @param hold_s       마지막 d를 유지할 s 길이
     * @param decay_s      0으로 감쇠할 s 길이
     */
    void append_forward_points_decay(f110_msgs::msg::WpntArray& traj,
                                     int target_count,
                                     double hold_s = 1.0,
                                     double decay_s = 4.0);

    // [추가] 글로벌 웨이포인트를 현재 위치(cur_s_)부터 앞으로 target_count개 뽑아 로컬 포맷으로 변환
    f110_msgs::msg::WpntArray build_forward_global_trajectory(int target_count);

    // ==============================
    // Visualization
    // ==============================
    visualization_msgs::msg::MarkerArray create_trajectory_markers(const f110_msgs::msg::WpntArray& trajectory);
    visualization_msgs::msg::Marker      create_obstacle_marker(const ObstacleFrenet& obstacle);
    visualization_msgs::msg::MarkerArray create_all_obstacles_markers(const std::vector<ObstacleFrenet>& obstacles);
    void publish_reference_path();
    void publish_all_obstacles(const std::vector<ObstacleFrenet>& obstacles);
    
    // ==============================
    // Debug / Utils
    // ==============================
    double quaternion_to_yaw(double x, double y, double z, double w);
};

#endif // SPLINE_PLANNER_HPP
