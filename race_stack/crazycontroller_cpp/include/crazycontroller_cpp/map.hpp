#pragma once
#include <crazycontroller_cpp/utils/steering_lookup/lookup_steer_angle.hpp>

#include <Eigen/Dense>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace crazycontroller_cpp {

struct MAPResult {
  double speed;
  double acceleration;
  double jerk;
  double steering_angle;
  Eigen::Vector2d L1_point;    // (x, y)
  double L1_distance;
  int idx_nearest_waypoint;
};

class MAP_Controller {
public:
  using Logger = std::function<void(const std::string&)>;

  MAP_Controller(
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
      Logger logger_info = [](const std::string& s){ (void)s; },
      Logger logger_warn = [](const std::string& s){ (void)s; });

  // ===== 런타임 파라미터 업데이트용 setters =====
  void set_t_clip_min(double v){ t_clip_min_ = v; }
  void set_t_clip_max(double v){ t_clip_max_ = v; }
  void set_m_l1(double v){ m_l1_ = v; }
  void set_q_l1(double v){ q_l1_ = v; }
  void set_speed_lookahead(double v){ speed_lookahead_ = v; }
  void set_lat_err_coeff(double v){ lat_err_coeff_ = v; }
  void set_acc_scaler_for_steer(double v){ acc_scaler_for_steer_ = v; }
  void set_dec_scaler_for_steer(double v){ dec_scaler_for_steer_ = v; }
  void set_start_scale_speed(double v){ start_scale_speed_ = v; }
  void set_end_scale_speed(double v){ end_scale_speed_ = v; }
  void set_downscale_factor(double v){ downscale_factor_ = v; }
  void set_speed_lookahead_for_steer(double v){ speed_lookahead_for_steer_ = v; }

  // main_loop(
  //   position_in_map[1x3], waypoint_array_in_map[Nx>=7],
  //   speed_now, position_in_map_frenet[2x1], acc_now[kx1], track_length)
  MAPResult main_loop(
      const Eigen::RowVector3d& position_in_map,
      const Eigen::MatrixXd& waypoint_array_in_map,
      double speed_now,
      const Eigen::Vector2d& position_in_map_frenet,
      const Eigen::VectorXd& acc_now,
      double track_length);

private:
  // ===== helpers (Python 메서드 1:1 대응) =====
  double calc_steering_angle(const Eigen::Vector2d& L1_point,
                             double L1_distance,
                             double yaw,
                             double lat_e_norm,
                             const Eigen::Vector2d& v);

  std::pair<Eigen::Vector2d, double> calc_L1_point(double lateral_error);

  std::optional<double> calc_speed_command(const Eigen::Vector2d& v,
                                           double lat_e_norm);

  static double distance(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2);

  double acc_scaling(double steer) const;

  double speed_steer_scaling(double steer, double speed) const;

  std::pair<double,double> calc_lateral_error_norm() const;

  double speed_adjust_lat_err(double global_speed, double lat_e_norm) const;

  int nearest_waypoint(const Eigen::Vector2d& position,
                       const Eigen::MatrixXd& waypoints_xy) const;

  Eigen::Vector2d waypoint_at_distance_before_car(double distance,
                                                  const Eigen::MatrixXd& waypoints_xy,
                                                  int idx_waypoint_behind_car) const;

  // ===== members (Python 속성들) =====
  double t_clip_min_;
  double t_clip_max_;
  double m_l1_;
  double q_l1_;
  double speed_lookahead_;
  double lat_err_coeff_;
  double acc_scaler_for_steer_;
  double dec_scaler_for_steer_;
  double start_scale_speed_;
  double end_scale_speed_;
  double downscale_factor_;
  double speed_lookahead_for_steer_;

  double loop_rate_;
  std::string LUT_path_;

  std::vector<double> lateral_error_list_;
  double curr_steering_angle_ = 0.0;
  std::optional<int> idx_nearest_waypoint_;
  std::optional<double> track_length_;

  std::optional<double> speed_command_;
  double curvature_waypoints_ = 0.0; // mean(|curvature|)
  Eigen::VectorXd acc_now_;          // 평균 사용

  Logger logger_info_;
  Logger logger_warn_;

  // 입력 보관
  Eigen::RowVector3d position_in_map_;        // [x, y, yaw]
  Eigen::MatrixXd waypoint_array_in_map_;     // [:, :2]=xy, [:,2]=speed, [:,5]=curv, [:,6]=heading
  double speed_now_ = 0.0;
  Eigen::Vector2d position_in_map_frenet_;    // [s, d]에서 d만 사용

  // steer lookup
  utils::steering_lookup::LookupSteerAngle steer_lookup_;

  // clamp 유틸
  template <typename T>
  static T clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(v, hi));
  }
};

} // namespace crazycontroller_cpp
