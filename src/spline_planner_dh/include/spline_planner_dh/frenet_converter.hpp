#ifndef FRENET_CONVERTER_HPP
#define FRENET_CONVERTER_HPP

#include <vector>
#include <cmath>
#include <algorithm>

class FrenetConverter {
public:
    FrenetConverter(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& psi);
    
    // Convert from Cartesian to Frenet coordinates
    std::pair<double, double> cartesian_to_frenet(double x, double y) const;
    
    // Convert from Frenet to Cartesian coordinates
    std::pair<double, double> frenet_to_cartesian(double s, double d) const;
    
    // Get maximum s value
    double get_max_s() const { return max_s_; }

private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> psi_;
    std::vector<double> s_;
    double max_s_;
    
    void compute_s_values();
    int find_closest_point(double x, double y) const;
    double distance(double x1, double y1, double x2, double y2) const;
};

#endif // FRENET_CONVERTER_HPP
