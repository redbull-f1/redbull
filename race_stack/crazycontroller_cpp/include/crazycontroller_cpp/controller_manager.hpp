#pragma once

#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rcl_interfaces/msg/parameter_event.hpp>  // ✅ on_parameter_event 시그니처에 필요

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

#include <crazycontroller_cpp/map.hpp>
#include <crazycontroller_cpp/utils/global_parameter/parameter_event_handler.hpp>

namespace crazycontroller_cpp {

class CrazyController : public rclcpp::Node {
public:
  CrazyController();

  // Python의 wait_for_messages()를 main()에서 호출할 수 있게 public로 둡니다.
  void wait_for_messages();

private:
  // ===== 파라미터/모드 =====
  int rate_ = 40;
  std::string LUT_path_;
  std::string mode_;

  // ===== 퍼블리셔 =====
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  // ===== 서브스크립션 =====
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr sub_track_length_;
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr sub_local_waypoints_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_car_state_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_car_state_frenet_;

  // ===== 상태 =====
  std::optional<double> track_length_;
  Eigen::MatrixXd waypoint_array_in_map_;   // Nx>=8 (x,y,speed,ratio,s,kappa,psi,ax)
  std::optional<double> speed_now_;
  std::optional<Eigen::RowVector3d> position_in_map_;      // [x,y,yaw]
  std::optional<Eigen::Vector4d> position_in_map_frenet_;  // [s,d,vs,vd]
  int waypoint_safety_counter_ = 0;

  // 가속도 rolling buffer (Python: 길이 10)
  Eigen::VectorXd acc_now_;  // 크기 10, 평균용

  // ===== 컨트롤러 =====
  std::unique_ptr<MAP_Controller> map_controller_;

  // ===== 파라미터 이벤트 핸들러 =====
  std::unique_ptr<ParameterEventHandler> param_handler_;
  std::shared_ptr<ParameterEventHandler::ParameterEventCallbackHandle> callback_handle_;

  // ===== 타이머 =====
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr param_init_timer_;  // ✅ 0ms 지연 초기화용 타이머 (bad_weak_ptr 방지)

  // ===== 내부 메서드 =====
  void init_map_controller();

  // 파라미터 선언 유틸(동적 파라미터와 유사한 효과)
  void declare_l1_dynamic_parameters_from_yaml(const std::string& yaml_path);
  void on_parameter_event(const rcl_interfaces::msg::ParameterEvent & event);

  // 콜백들
  void track_length_cb(const f110_msgs::msg::WpntArray::SharedPtr msg);
  void local_waypoint_cb(const f110_msgs::msg::WpntArray::SharedPtr msg);
  void car_state_cb(const nav_msgs::msg::Odometry::SharedPtr msg);
  void car_state_frenet_cb(const nav_msgs::msg::Odometry::SharedPtr msg);

  // 메인 루프(타이머)
  void control_loop();

  // MAP 1사이클
  std::pair<double,double> map_cycle(); // speed, steer
};

} // namespace crazycontroller_cpp
