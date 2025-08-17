#pragma once
#include <Eigen/Dense>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace crazycontroller_cpp {
namespace utils {
namespace steering_lookup {

// 파이썬: find_nearest(array, value) -> (array[idx], idx)
std::pair<double, std::size_t> find_nearest(const std::vector<double>& array, double value);

// 파이썬: find_closest_neighbors(array, value)
//  - NaN 앞까지만 사용
//  - value에 가장 가까운 인덱스(closest_idx) 기준으로 양 옆 이웃 중 더 가까운 것을 second로 선택
//  - 경계면 처리: 0 또는 마지막이면 같은 값/인덱스 두 번 반환
// 반환: (closest, closest_idx, second_closest, second_idx)
struct TwoNeighbors {
  double closest;
  std::size_t closest_idx;
  double second_closest;
  std::size_t second_idx;
};
TwoNeighbors find_closest_neighbors(const std::vector<double>& array, double value);

// 파이썬 클래스 LookupSteerAngle
class LookupSteerAngle {
public:
  using Logger = std::function<void(const std::string&)>;

  // lookup_table_path: CSV (',' 구분)
  //   - [0,0]은 사용 안 함
  //   - 0행 1열~ : 속도 벡터 (lu_vs)
  //   - 1행~ 0열 : 조향각 벡터 (lu_steers)
  //   - 1행~, 1열~ : 가속도 테이블
  explicit LookupSteerAngle(const std::string& lookup_table_path, Logger logger);

  // 파이썬: lookup_steer_angle(accel, vel) -> steer_angle * sign
  // sign은 accel > 0.0 이면 +1.0, 그렇지 않으면 -1.0 (0.0일 때도 -1.0로 동작함: 원본과 동일)
  double lookup_steer_angle(double accel, double vel) const;

  // 로드된 원본 테이블 접근(필요 시)
  const Eigen::MatrixXd& table() const { return lu_; }

private:
  static Eigen::MatrixXd load_csv_matrix(const std::string& path);

private:
  Eigen::MatrixXd lu_;   // 파이썬 self.lu
  Logger logger_;
};

} // namespace steering_lookup
} // namespace utils
} // namespace crazycontroller_cpp
