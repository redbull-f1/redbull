#include <crazycontroller_cpp/map.hpp>

#include <cmath>
#include <limits>

namespace crazycontroller_cpp {

// 편의 alias (필수는 아님)
using crazycontroller_cpp::utils::steering_lookup::LookupSteerAngle;

MAP_Controller::MAP_Controller(
    double t_clip_min,
    double t_clip_max,
    double m_l1,
    double q_l1,
    double speed_lookahead,
    double lat_err_coeff,
    double acc_scaler_for_steer,
    double dec_scaler_for_steer,
    double start_scale_speed,
    double end_scale_speed,
    double downscale_factor,
    double speed_lookahead_for_steer,
    double loop_rate,
    const std::string& LUT_path,
    Logger logger_info,
    Logger logger_warn)
  : t_clip_min_(t_clip_min),
    t_clip_max_(t_clip_max),
    m_l1_(m_l1),
    q_l1_(q_l1),
    speed_lookahead_(speed_lookahead),
    lat_err_coeff_(lat_err_coeff),
    acc_scaler_for_steer_(acc_scaler_for_steer),
    dec_scaler_for_steer_(dec_scaler_for_steer),
    start_scale_speed_(start_scale_speed),
    end_scale_speed_(end_scale_speed),
    downscale_factor_(downscale_factor),
    speed_lookahead_for_steer_(speed_lookahead_for_steer),
    loop_rate_(loop_rate),
    LUT_path_(LUT_path),
    acc_now_(10),                      // Python: np.zeros(10)
    logger_info_(std::move(logger_info)),
    logger_warn_(std::move(logger_warn)),
    steer_lookup_(LUT_path_, logger_info_)  // ⚡ LookupSteerAngle 생성
{
  acc_now_.setZero();
}

// ===================== main_loop =====================
MAPResult MAP_Controller::main_loop(
    const Eigen::RowVector3d& position_in_map,
    const Eigen::MatrixXd& waypoint_array_in_map,
    double speed_now,
    const Eigen::Vector2d& position_in_map_frenet,
    const Eigen::VectorXd& acc_now,
    double track_length)
{
  // 입력 저장
  position_in_map_ = position_in_map;
  waypoint_array_in_map_ = waypoint_array_in_map;
  speed_now_ = speed_now;
  position_in_map_frenet_ = position_in_map_frenet;
  acc_now_ = acc_now;
  track_length_ = track_length;

  const double yaw = position_in_map_(2);
  Eigen::Vector2d v(std::cos(yaw) * speed_now_, std::sin(yaw) * speed_now_);

  auto [lat_e_norm, lateral_error] = calc_lateral_error_norm();

  speed_command_ = calc_speed_command(v, lat_e_norm);

  double speed = 0.0, acceleration = 0.0, jerk = 0.0;
  if (speed_command_.has_value()) {
    speed = std::max(0.0, speed_command_.value());
  } else {
    speed = 0.0;
    acceleration = 0.0;
    jerk = 0.0;
    logger_warn_("[Controller] speed was none");
  }

  // L1 포인트/거리
  auto [L1_point, L1_distance] = calc_L1_point(lateral_error);

  if (!std::isfinite(L1_point.x()) || !std::isfinite(L1_point.y())) {
    throw std::runtime_error("L1_point is invalid");
  }

  // 조향
  double steering_angle = calc_steering_angle(L1_point, L1_distance, yaw, lat_e_norm, v);

  // 결과 패키징
  MAPResult out;
  out.speed = speed;
  out.acceleration = acceleration;
  out.jerk = jerk;
  out.steering_angle = steering_angle;
  out.L1_point = L1_point;
  out.L1_distance = L1_distance;
  out.idx_nearest_waypoint = idx_nearest_waypoint_.value_or(-1);
  return out;
}

// ===================== helpers =====================

double MAP_Controller::calc_steering_angle(const Eigen::Vector2d& L1_point,
                                           double L1_distance,
                                           double yaw,
                                           double lat_e_norm,
                                           const Eigen::Vector2d& v)
{
  // speed lookahead for steering
  const double adv_ts_st = speed_lookahead_for_steer_;
  Eigen::Vector2d la_position(
      position_in_map_(0) + v(0) * adv_ts_st,
      position_in_map_(1) + v(1) * adv_ts_st);

  // lookahead 인덱스
  const int idx_la_steer =
      nearest_waypoint(la_position, waypoint_array_in_map_.leftCols<2>());

  // 루프업용 속도
  const double speed_la_for_lu = waypoint_array_in_map_(idx_la_steer, 2);
  const double speed_for_lu = speed_adjust_lat_err(speed_la_for_lu, lat_e_norm);

  // eta 계산
  Eigen::Vector2d L1_vector(L1_point.x() - position_in_map_(0),
                            L1_point.y() - position_in_map_(1));
  double eta = 0.0;
  if (L1_vector.norm() == 0.0) {
    logger_warn_("[Controller] norm of L1 vector was 0, eta is set to 0");
    eta = 0.0;
  } else {
    Eigen::Vector2d nvec(-std::sin(yaw), std::cos(yaw)); // body 좌측 단위법선
    eta = std::asin(nvec.dot(L1_vector) / L1_vector.norm());
  }

  double lat_acc = 0.0;
  if (L1_distance == 0.0 || std::sin(eta) == 0.0) {
    logger_warn_("[Controller] L1 * np.sin(eta), lat_acc is set to 0");
    lat_acc = 0.0;
  } else {
    lat_acc = 2.0 * speed_for_lu * speed_for_lu / L1_distance * std::sin(eta);
  }

  // LU 기반 조향
  double steering_angle = steer_lookup_.lookup_steer_angle(lat_acc, speed_for_lu);

  // 속도 기반 스케일링 (comment 2 만 적용)
  steering_angle = speed_steer_scaling(steering_angle, speed_for_lu);

  // 현재 속도로 추가 스케일
  steering_angle *= clamp(1.0 + (speed_now_ / 10.0), 1.0, 1.25);

  // 스티어 변화율 제한
  const double threshold = 0.4;
  if (std::abs(steering_angle - curr_steering_angle_) > threshold) {
    logger_info_("[MAP Controller] steering angle clipped");
  }
  steering_angle = clamp(steering_angle,
                         curr_steering_angle_ - threshold,
                         curr_steering_angle_ + threshold);
  curr_steering_angle_ = steering_angle;
  return steering_angle;
}

std::pair<Eigen::Vector2d, double> MAP_Controller::calc_L1_point(double lateral_error) {
  idx_nearest_waypoint_ =
      nearest_waypoint(position_in_map_.head<2>(), waypoint_array_in_map_.leftCols<2>());

  if (!idx_nearest_waypoint_.has_value()) {
    idx_nearest_waypoint_ = 0;
  }

  // 남은 구간 평균 곡률
  if ((waypoint_array_in_map_.rows() - idx_nearest_waypoint_.value()) > 2) {
    curvature_waypoints_ =
        (waypoint_array_in_map_.block(idx_nearest_waypoint_.value(), 5,
                                      waypoint_array_in_map_.rows() - idx_nearest_waypoint_.value(), 1)
         .cwiseAbs().mean());
  }

  // L1 거리
  double L1_distance = q_l1_ + speed_now_ * m_l1_;

  // 하한 clip
  const double lower_bound = std::max(t_clip_min_, std::sqrt(2.0) * lateral_error);
  L1_distance = clamp(L1_distance, lower_bound, t_clip_max_);

  // 앞쪽 waypoint 위치
  Eigen::Vector2d L1_point =
      waypoint_at_distance_before_car(L1_distance,
                                      waypoint_array_in_map_.leftCols<2>(),
                                      idx_nearest_waypoint_.value());
  return {L1_point, L1_distance};
}

std::optional<double> MAP_Controller::calc_speed_command(const Eigen::Vector2d& v,
                                                         double lat_e_norm)
{
  const double adv_ts_sp = speed_lookahead_;
  Eigen::Vector2d la_position(
      position_in_map_(0) + v(0) * adv_ts_sp,
      position_in_map_(1) + v(1) * adv_ts_sp);

  const int idx_la_position =
      nearest_waypoint(la_position, waypoint_array_in_map_.leftCols<2>());

  double global_speed = waypoint_array_in_map_(idx_la_position, 2);
  global_speed = speed_adjust_lat_err(global_speed, lat_e_norm);
  return global_speed;
}

double MAP_Controller::distance(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
  return (p2 - p1).norm();
}

double MAP_Controller::acc_scaling(double steer) const {
  const double mean_acc = (acc_now_.size() > 0) ? acc_now_.mean() : 0.0;
  if (mean_acc >= 1.0) {
    return steer * acc_scaler_for_steer_;
  } else if (mean_acc <= -1.0) {
    return steer * dec_scaler_for_steer_;
  }
  return steer;
}

double MAP_Controller::speed_steer_scaling(double steer, double speed) const {
  const double speed_diff = std::max(0.1, end_scale_speed_ - start_scale_speed_);
  const double factor = 1.0 - clamp((speed - start_scale_speed_) / speed_diff, 0.0, 1.0) * downscale_factor_;
  return steer * factor;
}

std::pair<double,double> MAP_Controller::calc_lateral_error_norm() const {
  // frenet d 절대값
  const double lateral_error = std::abs(position_in_map_frenet_(1));

  const double max_lat_e = 0.5;
  const double min_lat_e = 0.0;
  const double lat_e_clip = clamp(lateral_error, min_lat_e, max_lat_e);
  const double lat_e_norm = 0.5 * ((lat_e_clip - min_lat_e) / (max_lat_e - min_lat_e));
  return {lat_e_norm, lateral_error};
}

double MAP_Controller::speed_adjust_lat_err(double global_speed, double lat_e_norm) const {
  double lat_e_coeff = lat_err_coeff_; // [0,1]
  lat_e_norm *= 2.0;

  // 평균 곡률 정규화
  const double curv = clamp(2.0 * ( (curvature_waypoints_) / 0.8 ) - 2.0, 0.0, 1.0);
  global_speed *= (1.0 - lat_e_coeff + lat_e_coeff * std::exp(-lat_e_norm * curv));
  return global_speed;
}

int MAP_Controller::nearest_waypoint(const Eigen::Vector2d& position,
                                     const Eigen::MatrixXd& waypoints_xy) const
{
  const Eigen::Index N = waypoints_xy.rows();
  if (N <= 0) return 0;
  Eigen::Vector2d pos = position;
  double best = std::numeric_limits<double>::max();
  int best_idx = 0;
  for (Eigen::Index i = 0; i < N; ++i) {
    double d = (pos - waypoints_xy.row(i).transpose()).cwiseAbs().norm();
    if (d < best) {
      best = d;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

Eigen::Vector2d MAP_Controller::waypoint_at_distance_before_car(double distance,
                                                                const Eigen::MatrixXd& waypoints_xy,
                                                                int idx_waypoint_behind_car) const
{
  if (!std::isfinite(distance)) distance = t_clip_min_;
  double d_distance = distance;
  const double waypoints_distance = 0.1; // 원본 고정값
  int d_index = static_cast<int>(d_distance / waypoints_distance + 0.5);

  const int idx = std::min<int>(static_cast<int>(waypoints_xy.rows()) - 1,
                                idx_waypoint_behind_car + d_index);
  return waypoints_xy.row(idx).transpose();
}

} // namespace crazycontroller_cpp
