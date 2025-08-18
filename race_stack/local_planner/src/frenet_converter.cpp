#include "spline_planner_dh_test/frenet_converter.hpp"

FrenetConverter::FrenetConverter(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 const std::vector<double>& psi,
                                 bool closed_loop)
: x_(x), y_(y), psi_(psi), closed_loop_(closed_loop) {
    compute_s_values();
}

void FrenetConverter::compute_s_values() {
    const size_t N = x_.size();
    s_.assign(N, 0.0);
    if (N < 2) { max_s_ = 0.0; return; }

    for (size_t i = 1; i < N; ++i) {
        s_[i] = s_[i-1] + distance(x_[i-1], y_[i-1], x_[i], y_[i]);
    }
    // 폐루프면 last->first 구간 포함
    if (closed_loop_) {
        close_seg_len_ = distance(x_.back(), y_.back(), x_.front(), y_.front());
        max_s_ = s_.back() + close_seg_len_;
    } else {
        close_seg_len_ = 0.0;
        max_s_ = s_.back();
    }
}

int FrenetConverter::find_closest_segment(double x, double y, double& t,
                                          double& ref_x, double& ref_y,
                                          double& ref_heading, double& out_sqdist) const {
    const size_t N = x_.size();
    auto best_i = 0;
    double best_t = 0.0;
    double best_sq = std::numeric_limits<double>::infinity();
    double best_rx=0.0, best_ry=0.0, best_heading=0.0;

    const auto check_seg = [&](size_t i, size_t j){
        double vx = x_[j] - x_[i];
        double vy = y_[j] - y_[i];
        double seg_len2 = vx*vx + vy*vy;
        if (seg_len2 < 1e-12) return; // degenerate

        // 수선 투영 계수 t
        double wx = x - x_[i];
        double wy = y - y_[i];
        double tt = (wx*vx + wy*vy) / seg_len2;
        tt = std::max(0.0, std::min(1.0, tt));

        double projx = x_[i] + tt * vx;
        double projy = y_[i] + tt * vy;
        double dx = x - projx;
        double dy = y - projy;
        double sq = dx*dx + dy*dy;

        if (sq < best_sq) {
            best_sq = sq;
            best_i = static_cast<int>(i);
            best_t = tt;
            best_rx = projx;
            best_ry = projy;
            best_heading = std::atan2(vy, vx); // 세그먼트 진행방향
        }
    };

    // 0..N-2 세그먼트
    for (size_t i = 0; i + 1 < N; ++i) check_seg(i, i+1);
    // 폐루프면 N-1 -> 0 세그먼트도
    if (closed_loop_ && N >= 2) check_seg(N-1, 0);

    t = best_t;
    ref_x = best_rx;
    ref_y = best_ry;
    ref_heading = best_heading;
    out_sqdist = best_sq;
    return best_i;
}

int FrenetConverter::locate_segment_by_s(double s, double& t) const {
    // s를 [0,max_s_)로 래핑
    if (max_s_ > 0.0) {
        while (s < 0.0) s += max_s_;
        while (s >= max_s_) s -= max_s_;
    }

    const size_t N = x_.size();
    if (!closed_loop_) {
        // 열린 경로: s_[i] <= s < s_[i+1] 를 찾음
        auto it = std::upper_bound(s_.begin(), s_.end(), s);
        size_t idx = (it == s_.begin()) ? 0 : (static_cast<size_t>(it - s_.begin()) - 1);
        if (idx >= N-1) { idx = N-2; t = 1.0; return static_cast<int>(idx); }
        double seg_len = s_[idx+1] - s_[idx];
        t = (seg_len > 1e-9) ? ( (s - s_[idx]) / seg_len ) : 0.0;
        t = std::clamp(t, 0.0, 1.0);
        return static_cast<int>(idx);
    } else {
        // 폐루프: s가 마지막 포인트 이후면 마지막->첫점 세그먼트
        if (s >= s_.back()) {
            double seg_len = close_seg_len_;
            t = (seg_len > 1e-9) ? ((s - s_.back()) / seg_len) : 0.0;
            t = std::clamp(t, 0.0, 1.0);
            return static_cast<int>(N - 1); // N-1 -> 0
        } else {
            auto it = std::upper_bound(s_.begin(), s_.end(), s);
            size_t idx = (it == s_.begin()) ? 0 : (static_cast<size_t>(it - s_.begin()) - 1);
            if (idx >= N-1) { idx = N-2; t = 1.0; return static_cast<int>(idx); }
            double seg_len = s_[idx+1] - s_[idx];
            t = (seg_len > 1e-9) ? ( (s - s_[idx]) / seg_len ) : 0.0;
            t = std::clamp(t, 0.0, 1.0);
            return static_cast<int>(idx);
        }
    }
}

std::pair<double, double> FrenetConverter::cartesian_to_frenet(double x, double y) const {
    double t, rx, ry, heading, sq;
    int i = find_closest_segment(x, y, t, rx, ry, heading, sq);

    // 세그먼트 기하
    size_t j = (i + 1 < x_.size()) ? (i + 1) : 0;
    double vx = x_[j] - x_[i];
    double vy = y_[j] - y_[i];
    double seg_len = std::sqrt(vx*vx + vy*vy);
    if (seg_len < 1e-12) return {0.0, 0.0};

    // 진행방향 단위벡터/법선
    double tx = vx / seg_len, ty = vy / seg_len;
    double nx = -ty, ny = tx; // 왼쪽 양수

    // ref point까지의 s
    double s = s_[i] + t * ((j>i) ? (s_[j] - s_[i]) : close_seg_len_);
    if (closed_loop_) {
        if (s >= max_s_) s -= max_s_;
        if (s < 0.0)     s += max_s_;
    } else {
        s = std::clamp(s, 0.0, max_s_);
    }

    // d = (p - p_ref)·n̂
    double dx = x - rx, dy = y - ry;
    double d = dx*nx + dy*ny;

    return {s, d};
}

std::pair<double, double> FrenetConverter::frenet_to_cartesian(double s, double d) const {
    double t;
    int i = locate_segment_by_s(s, t);

    size_t N = x_.size();
    size_t j = (i + 1 < static_cast<int>(N)) ? (i + 1) : 0;

    // 세그먼트 보간 ref point
    double ref_x, ref_y, vx, vy, seg_len;
    if (j != 0) {
        ref_x = x_[i] + t * (x_[j] - x_[i]);
        ref_y = y_[i] + t * (y_[j] - y_[i]);
        vx = x_[j] - x_[i];
        vy = y_[j] - y_[i];
    } else {
        // N-1 -> 0
        ref_x = x_[i] + t * (x_[0] - x_[i]);
        ref_y = y_[i] + t * (y_[0] - y_[i]);
        vx = x_[0] - x_[i];
        vy = y_[0] - y_[i];
    }
    seg_len = std::sqrt(vx*vx + vy*vy);
    if (seg_len < 1e-12) return {ref_x, ref_y};

    double tx = vx / seg_len, ty = vy / seg_len;
    double nx = -ty, ny = tx; // 왼쪽 양수

    double X = ref_x + d * nx;
    double Y = ref_y + d * ny;
    return {X, Y};
}
