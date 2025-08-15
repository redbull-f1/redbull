#pragma once
#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <algorithm>

class FrenetConverter {
public:
    // closed_loop: 트랙이 폐루프일 때 true (기본값 true)
    FrenetConverter(const std::vector<double>& x,
                    const std::vector<double>& y,
                    const std::vector<double>& psi,
                    bool closed_loop = true);

    // x,y -> (s,d)
    std::pair<double,double> cartesian_to_frenet(double x, double y) const;

    // (s,d) -> x,y
    std::pair<double,double> frenet_to_cartesian(double s, double d) const;

    double max_s() const { return max_s_; }

private:
    void compute_s_values();
    static inline double distance(double x1,double y1,double x2,double y2){
        double dx=x2-x1, dy=y2-y1; return std::sqrt(dx*dx+dy*dy);
    }

    // 점을 세그먼트 i->i+1(폐루프면 마지막->첫점 포함)에 수선 투영
    // 반환: (가장 가까운 세그먼트 index, t in [0,1], ref_x, ref_y, seg_heading, sq_dist)
    int find_closest_segment(double x, double y, double& t,
                             double& ref_x, double& ref_y,
                             double& ref_heading, double& out_sqdist) const;

    // s로 세그먼트/보간계수 찾기
    int locate_segment_by_s(double s, double& t) const;

private:
    std::vector<double> x_, y_, psi_;   // psi_는 안 써도 보관만
    std::vector<double> s_;             // 각 waypoint까지의 누적호 길이(마지막→첫점 구간은 s_에 없음)
    double max_s_ = 0.0;                // 전체 길이(폐루프면 last->first까지 포함)
    bool closed_loop_ = true;
    double close_seg_len_ = 0.0;        // 마지막→첫점 길이(폐루프일 때)
};
