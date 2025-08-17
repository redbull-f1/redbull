#include <crazycontroller_cpp/utils/steering_lookup/lookup_steer_angle.hpp>

#include <algorithm>   // std::find_if
#include <cctype>      // std::isspace
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace crazycontroller_cpp {
namespace utils {
namespace steering_lookup {

// ---------- helpers ----------
static inline bool is_nan(double x) {
  return std::isnan(x);
}

// CSV -> Eigen::MatrixXd
Eigen::MatrixXd LookupSteerAngle::load_csv_matrix(const std::string& path) {
  std::ifstream fin(path);
  if (!fin.is_open()) {
    throw std::runtime_error("Failed to open CSV: " + path);
  }

  std::vector<std::vector<double>> rows;
  std::string line;
  std::size_t max_cols = 0;

  while (std::getline(fin, line)) {
    std::vector<double> cols;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      // 공백 제거
      if (!cell.empty() && (cell.front() == ' ' || cell.front() == '\t')) {
        cell.erase(cell.begin(), std::find_if(cell.begin(), cell.end(),
                   [](unsigned char ch){ return !std::isspace(ch); }));
      }
      if (!cell.empty() && (cell.back() == ' ' || cell.back() == '\t')) {
        cell.erase(std::find_if(cell.rbegin(), cell.rend(),
                   [](unsigned char ch){ return !std::isspace(ch); }).base(),
                   cell.end());
      }
      // 빈 셀 -> NaN
      if (cell.empty()) {
        cols.push_back(std::numeric_limits<double>::quiet_NaN());
      } else {
        try {
          cols.push_back(std::stod(cell));
        } catch (...) {
          // 숫자 변환 실패 시 NaN
          cols.push_back(std::numeric_limits<double>::quiet_NaN());
        }
      }
    }
    max_cols = std::max(max_cols, cols.size());
    rows.emplace_back(std::move(cols));
  }

  // ragged 보정: 부족한 칸은 NaN으로 채움
  const std::size_t r = rows.size();
  const std::size_t c = max_cols;
  Eigen::MatrixXd M(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
  for (std::size_t i = 0; i < r; ++i) {
    rows[i].resize(c, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t j = 0; j < c; ++j) {
      M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = rows[i][j];
    }
  }
  return M;
}

// ---------- free functions ----------
std::pair<double, std::size_t> find_nearest(const std::vector<double>& array, double value) {
  if (array.empty()) {
    throw std::runtime_error("find_nearest: array is empty");
  }
  double best_val = array[0];
  std::size_t best_idx = 0;
  double best_dist = std::abs(array[0] - value);

  for (std::size_t i = 1; i < array.size(); ++i) {
    double d = std::abs(array[i] - value);
    if (d < best_dist) {
      best_dist = d;
      best_idx = i;
      best_val = array[i];
    }
  }
  return {best_val, best_idx};
}

TwoNeighbors find_closest_neighbors(const std::vector<double>& array_in, double value) {
  if (array_in.empty()) {
    throw std::runtime_error("find_closest_neighbors: array is empty");
  }

  // 첫 NaN 이전까지만 사용
  std::vector<double> array;
  array.reserve(array_in.size());
  for (double v : array_in) {
    if (is_nan(v)) break;
    array.push_back(v);
  }
  if (array.empty()) {
    throw std::runtime_error("find_closest_neighbors: array has only NaNs");
  }

  auto [closest, closest_idx] = find_nearest(array, value);

  if (closest_idx == 0) {
    return {array[0], 0, array[0], 0};
  } else if (closest_idx == array.size() - 1) {
    std::size_t i = array.size() - 1;
    return {array[i], i, array[i], i};
  } else {
    // 두 이웃(closest_idx-1, closest_idx+1) 중 더 가까운 쪽 선택
    std::size_t left  = closest_idx - 1;
    std::size_t right = closest_idx + 1;

    double dl = std::abs(array[left]  - value);
    double dr = std::abs(array[right] - value);

    std::size_t second_idx = (dr < dl) ? right : left;
    double second_closest = array[second_idx];

    return {closest, closest_idx, second_closest, second_idx};
  }
}

// ---------- class ----------
LookupSteerAngle::LookupSteerAngle(const std::string& lookup_table_path, Logger logger)
  : lu_(load_csv_matrix(lookup_table_path)), logger_(std::move(logger)) {
  // Python 원본은 logger를 거의 사용하지 않음(주석 처리된 메시지). 그대로 유지.
}

double LookupSteerAngle::lookup_steer_angle(double accel, double vel) const {
  // sign 처리 (원본 로직 재현: accel > 0.0 ? +1 : -1) -> accel == 0.0 도 -1.0
  const double sign_accel = (accel > 0.0) ? 1.0 : -1.0;
  accel = std::abs(accel);

  // lu_vs: 첫 행(0), 첫 열 제외(1:)
  if (lu_.rows() < 2 || lu_.cols() < 2) {
    throw std::runtime_error("lookup table size invalid");
  }

  std::vector<double> lu_vs;
  lu_vs.reserve(static_cast<std::size_t>(lu_.cols() - 1));
  for (Eigen::Index j = 1; j < lu_.cols(); ++j) {
    lu_vs.push_back(lu_(0, j));
  }

  // lu_steers: 첫 열(0), 첫 행 제외(1:)
  std::vector<double> lu_steers;
  lu_steers.reserve(static_cast<std::size_t>(lu_.rows() - 1));
  for (Eigen::Index i = 1; i < lu_.rows(); ++i) {
    lu_steers.push_back(lu_(i, 0));
  }

  // 가장 가까운 속도 인덱스
  auto nearest = find_nearest(lu_vs, vel);
  std::size_t c_v_idx = nearest.second;

  // 해당 속도 열 (c_v_idx + 1), 행 1: (가속도들)
  std::vector<double> accel_col;
  accel_col.reserve(static_cast<std::size_t>(lu_.rows() - 1));
  for (Eigen::Index i = 1; i < lu_.rows(); ++i) {
    accel_col.push_back(lu_(i, static_cast<Eigen::Index>(c_v_idx + 1)));
  }

  // 두 이웃 가속도 찾기
  auto neigh = find_closest_neighbors(accel_col, accel);

  double steer_angle;
  if (neigh.closest_idx == neigh.second_idx) {
    steer_angle = lu_steers[neigh.closest_idx];
  } else {
    // 파이썬 np.interp(accel, [x0, x1], [y0, y1])
    const double x0 = neigh.closest;
    const double x1 = neigh.second_closest;
    const double y0 = lu_steers[neigh.closest_idx];
    const double y1 = lu_steers[neigh.second_idx];

    if (std::abs(x1 - x0) < 1e-12) {
      steer_angle = y0; // 분모 0 보호
    } else {
      const double t = (accel - x0) / (x1 - x0);
      steer_angle = y0 + (y1 - y0) * t;
    }
  }

  return steer_angle * sign_accel;
}

} // namespace steering_lookup
} // namespace utils
} // namespace crazycontroller_cpp
