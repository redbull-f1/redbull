#include <crazycontroller_cpp/controller_manager.hpp>

#include <yaml-cpp/yaml.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <rcl_interfaces/msg/floating_point_range.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp/parameter.hpp>
#include <rclcpp/rclcpp.hpp>  // 로그 매크로 사용

#include <algorithm>
#include <cmath>
#include <stdexcept>

using ackermann_msgs::msg::AckermannDriveStamped;
using f110_msgs::msg::WpntArray;
using nav_msgs::msg::Odometry;
using rcl_interfaces::msg::FloatingPointRange;
using rcl_interfaces::msg::ParameterDescriptor;
using rcl_interfaces::msg::ParameterType;

namespace crazycontroller_cpp {

CrazyController::CrazyController()
: rclcpp::Node(
    "crazycontroller_manager",
    rclcpp::NodeOptions()
      .allow_undeclared_parameters(true)
      .automatically_declare_parameters_from_overrides(true))
{
  // rate (고정 40Hz)
  rate_ = 40;

  // 필수 파라미터: lookup_table_path, mode, l1_params_path
  LUT_path_ = this->get_parameter("lookup_table_path").as_string();
  mode_ = this->get_parameter("mode").as_string();
  RCLCPP_INFO(this->get_logger(), "Using lookup table: %s", LUT_path_.c_str());

  // 퍼블리셔
  std::string publish_topic = "/drive"; // python과 동일
  drive_pub_ = this->create_publisher<AckermannDriveStamped>(publish_topic, 10);

  // MAP 컨트롤러 초기화
  if (mode_ == "MAP") {
    RCLCPP_INFO(this->get_logger(), "Initializing MAP controller");
    init_map_controller();
  } else if (mode_ == "PP") {
    RCLCPP_INFO(this->get_logger(), "PP controller not implemented yet");
  } else {
    RCLCPP_ERROR(this->get_logger(), "Invalid mode: %s", mode_.c_str());
  }

  // 구독자
  sub_track_length_   = this->create_subscription<WpntArray>("/global_waypoints", 10,
                        std::bind(&CrazyController::track_length_cb, this, std::placeholders::_1));
  sub_local_waypoints_ = this->create_subscription<WpntArray>("/local_waypoints", 10,
                        std::bind(&CrazyController::local_waypoint_cb, this, std::placeholders::_1));
  sub_car_state_      = this->create_subscription<Odometry>("/car_state/odom", 10,
                        std::bind(&CrazyController::car_state_cb, this, std::placeholders::_1));
  sub_car_state_frenet_ = this->create_subscription<Odometry>("/car_state/frenet/odom", 10,
                        std::bind(&CrazyController::car_state_frenet_cb, this, std::placeholders::_1));

  // ✅ ParameterEventHandler는 생성자 밖(노드가 shared_ptr로 소유된 이후)에 초기화
  //    0ms 지연 타이머로 한 번만 실행되게 함
  param_init_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(0),
      [this]() {
        // 한 번만 실행하고 취소
        param_init_timer_->cancel();

        param_handler_ = std::make_unique<ParameterEventHandler>(
            this->shared_from_this(), rclcpp::ParametersQoS());
        callback_handle_ = param_handler_->add_parameter_event_callback(
            [this](const rcl_interfaces::msg::ParameterEvent & ev) {
              this->on_parameter_event(ev);
            });
        RCLCPP_INFO(this->get_logger(), "ParameterEventHandler initialized");
      });

  // 메인 제어 루프 타이머 (Python: self.create_timer(1 / rate, control_loop))
  const double period = 1.0 / static_cast<double>(rate_);
  timer_ = this->create_wall_timer(
      std::chrono::duration<double>(period),
      std::bind(&CrazyController::control_loop, this));

  RCLCPP_INFO(this->get_logger(), "CrazyController ready");
}

void CrazyController::wait_for_messages() {
  RCLCPP_INFO(this->get_logger(), "CrazyController Manager waiting for messages...");
  bool track_length_received = false;
  bool waypoint_array_received = false;
  bool car_state_received = false;

  while (rclcpp::ok() && (!track_length_received || !waypoint_array_received || !car_state_received)) {
    rclcpp::spin_some(this->shared_from_this());

    if (track_length_.has_value() && !track_length_received) {
      RCLCPP_INFO(this->get_logger(), "Received track length");
      track_length_received = true;
    }
    if (waypoint_array_in_map_.size() > 0 && !waypoint_array_received) {
      RCLCPP_INFO(this->get_logger(), "Received waypoint array");
      waypoint_array_received = true;
    }
    if (speed_now_.has_value() && position_in_map_.has_value() && position_in_map_frenet_.has_value() && !car_state_received) {
      RCLCPP_INFO(this->get_logger(), "Received car state messages");
      car_state_received = true;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(5));
  }
  RCLCPP_INFO(this->get_logger(), "All required messages received. Continuing...");
}

void CrazyController::declare_l1_dynamic_parameters_from_yaml(const std::string& yaml_path) {
  YAML::Node root = YAML::LoadFile(yaml_path);
  if (!root["controller"] || !root["controller"]["ros__parameters"]) {
    throw std::runtime_error("Invalid l1_params YAML: missing controller.ros__parameters");
  }
  const YAML::Node params = root["controller"]["ros__parameters"];

  auto fp = [](double a, double b, double step){ FloatingPointRange r; r.from_value=a; r.to_value=b; r.step=step; return r; };

  auto declare_double = [&](const std::string& name, double def, FloatingPointRange range) {
    ParameterDescriptor desc;
    desc.type = ParameterType::PARAMETER_DOUBLE;
    desc.floating_point_range = { range };
    (void)this->declare_parameter<double>(name, def, desc);
  };

  declare_double("t_clip_min",            params["t_clip_min"].as<double>(),            fp(0.0, 1.5, 0.01));
  declare_double("t_clip_max",            params["t_clip_max"].as<double>(),            fp(0.0, 10.0, 0.01));
  declare_double("m_l1",                  params["m_l1"].as<double>(),                  fp(0.0, 1.0, 0.001));
  declare_double("q_l1",                  params["q_l1"].as<double>(),                  fp(-1.0, 1.0, 0.001));
  declare_double("speed_lookahead",       params["speed_lookahead"].as<double>(),       fp(0.0, 1.0, 0.01));
  declare_double("lat_err_coeff",         params["lat_err_coeff"].as<double>(),         fp(0.0, 1.0, 0.01));
  declare_double("acc_scaler_for_steer",  params["acc_scaler_for_steer"].as<double>(),  fp(0.0, 1.5, 0.01));
  declare_double("dec_scaler_for_steer",  params["dec_scaler_for_steer"].as<double>(),  fp(0.0, 1.5, 0.01));
  declare_double("start_scale_speed",     params["start_scale_speed"].as<double>(),     fp(0.0, 10.0, 0.01));
  declare_double("end_scale_speed",       params["end_scale_speed"].as<double>(),       fp(0.0, 10.0, 0.01));
  declare_double("downscale_factor",      params["downscale_factor"].as<double>(),      fp(0.0, 0.5, 0.01));
  declare_double("speed_lookahead_for_steer", params["speed_lookahead_for_steer"].as<double>(), fp(0.0, 0.2, 0.01));
}

void CrazyController::init_map_controller() {
  const std::string l1_params_path = this->get_parameter("l1_params_path").as_string();
  declare_l1_dynamic_parameters_from_yaml(l1_params_path);

  // acc rolling buffer
  acc_now_ = Eigen::VectorXd::Zero(10);

  // MAP 컨트롤러 생성
  auto info = [this](const std::string& s){ RCLCPP_INFO(this->get_logger(), "%s", s.c_str()); };
  auto warn = [this](const std::string& s){ RCLCPP_WARN(this->get_logger(), "%s", s.c_str()); };

  map_controller_ = std::make_unique<MAP_Controller>(
      this->get_parameter("t_clip_min").as_double(),
      this->get_parameter("t_clip_max").as_double(),
      this->get_parameter("m_l1").as_double(),
      this->get_parameter("q_l1").as_double(),
      this->get_parameter("speed_lookahead").as_double(),
      this->get_parameter("lat_err_coeff").as_double(),
      this->get_parameter("acc_scaler_for_steer").as_double(),
      this->get_parameter("dec_scaler_for_steer").as_double(),
      this->get_parameter("start_scale_speed").as_double(),
      this->get_parameter("end_scale_speed").as_double(),
      this->get_parameter("downscale_factor").as_double(),
      this->get_parameter("speed_lookahead_for_steer").as_double(),
      static_cast<double>(rate_),
      LUT_path_,
      info, warn);
}

void CrazyController::on_parameter_event(const rcl_interfaces::msg::ParameterEvent & event) {
  if (event.node != "/crazycontroller_manager") return;
  if (!map_controller_) return;
  if (mode_ != "MAP") return;

  auto getp = [&](const char* name){ return this->get_parameter(name).as_double(); };

  map_controller_->set_t_clip_min(getp("t_clip_min"));
  map_controller_->set_t_clip_max(getp("t_clip_max"));
  map_controller_->set_m_l1(getp("m_l1"));
  map_controller_->set_q_l1(getp("q_l1"));
  map_controller_->set_speed_lookahead(getp("speed_lookahead"));
  map_controller_->set_lat_err_coeff(getp("lat_err_coeff"));
  map_controller_->set_acc_scaler_for_steer(getp("acc_scaler_for_steer"));
  map_controller_->set_dec_scaler_for_steer(getp("dec_scaler_for_steer"));
  map_controller_->set_start_scale_speed(getp("start_scale_speed"));
  map_controller_->set_end_scale_speed(getp("end_scale_speed"));
  map_controller_->set_downscale_factor(getp("downscale_factor"));
  map_controller_->set_speed_lookahead_for_steer(getp("speed_lookahead_for_steer"));

  RCLCPP_INFO(this->get_logger(), "Updated parameters");
}

void CrazyController::track_length_cb(const WpntArray::SharedPtr msg) {
  if (msg->wpnts.empty()) return;
  track_length_ = msg->wpnts.back().s_m;
}

void CrazyController::local_waypoint_cb(const WpntArray::SharedPtr msg) {
  const auto N = static_cast<int>(msg->wpnts.size());
  if (N <= 0) return;

  // [x, y, speed, ratio, s, kappa, psi, ax]
  waypoint_array_in_map_.resize(N, 8);
  for (int i = 0; i < N; ++i) {
    const auto & w = msg->wpnts[i];
    const double ratio = (w.d_left + w.d_right) != 0.0
                         ? std::min(w.d_left, w.d_right) / (w.d_right + w.d_left)
                         : 0.0;
    waypoint_array_in_map_(i,0) = w.x_m;
    waypoint_array_in_map_(i,1) = w.y_m;
    waypoint_array_in_map_(i,2) = w.vx_mps;
    waypoint_array_in_map_(i,3) = ratio;
    waypoint_array_in_map_(i,4) = w.s_m;
    waypoint_array_in_map_(i,5) = w.kappa_radpm;
    waypoint_array_in_map_(i,6) = w.psi_rad;
    waypoint_array_in_map_(i,7) = w.ax_mps2;
  }
  waypoint_safety_counter_ = 0;
}

void CrazyController::car_state_cb(const Odometry::SharedPtr msg) {
  speed_now_ = msg->twist.twist.linear.x;

  const auto & p = msg->pose.pose.position;
  const auto & q = msg->pose.pose.orientation;

  tf2::Quaternion quat(q.x, q.y, q.z, q.w);
  double roll, pitch, yaw;
  tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);

  Eigen::RowVector3d pose;
  pose << p.x, p.y, yaw;
  position_in_map_ = pose;
}

void CrazyController::car_state_frenet_cb(const Odometry::SharedPtr msg) {
  const double s  = msg->pose.pose.position.x;
  const double d  = msg->pose.pose.position.y;
  const double vs = msg->twist.twist.linear.x;
  const double vd = msg->twist.twist.linear.y;

  Eigen::Vector4d fr;
  fr << s, d, vs, vd;
  position_in_map_frenet_ = fr;
}

std::pair<double,double> CrazyController::map_cycle() {
  auto res = map_controller_->main_loop(
      position_in_map_.value(),
      waypoint_array_in_map_,
      speed_now_.value(),
      Eigen::Vector2d(position_in_map_frenet_.value()(0), position_in_map_frenet_.value()(1)),
      acc_now_,
      track_length_.value_or(0.0));

  waypoint_safety_counter_ += 1;
  if (waypoint_safety_counter_ >= rate_ * 5) {
    RCLCPP_WARN(this->get_logger(), "[crazycontroller_manager] No fresh local waypoints. STOPPING!!");
    res.speed = 0.0;
    res.steering_angle = 0.0;
  }
  return {res.speed, res.steering_angle};
}

void CrazyController::control_loop() {
  if (mode_ != "MAP" || !map_controller_) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Unsupported mode or controller not ready");
    return;
  }
  if (!speed_now_.has_value() || !position_in_map_.has_value() || !position_in_map_frenet_.has_value() || waypoint_array_in_map_.rows() == 0) {
    return;
  }

  auto [speed, steer] = map_cycle();
  RCLCPP_INFO(this->get_logger(), "[DEBUG] steering_angle to publish: %.6f", steer);

  AckermannDriveStamped ack;

  // Humble 호환 타임스탬프( to_msg() 미사용 )
  const rclcpp::Time now = this->get_clock()->now();
  const int64_t ns = now.nanoseconds();
  ack.header.stamp.sec = static_cast<int32_t>(ns / 1000000000LL);
  ack.header.stamp.nanosec = static_cast<uint32_t>(ns % 1000000000LL);

  ack.header.frame_id = "base_link";
  ack.drive.steering_angle = steer;
  ack.drive.speed = speed;
  drive_pub_->publish(ack);
}

} // namespace crazycontroller_cpp
