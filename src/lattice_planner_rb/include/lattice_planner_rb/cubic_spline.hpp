#ifndef CUBIC_SPLINE_HPP
#define CUBIC_SPLINE_HPP

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace lattice_planner {

/**
 * @brief Cubic Spline Interpolation Class
 * 
 * This class implements cubic spline interpolation for smooth trajectory generation.
 * It computes the second derivatives and provides smooth interpolation between waypoints.
 */
class CubicSpline {
public:
    /**
     * @brief Constructor
     * @param x X coordinates (must be in ascending order)
     * @param y Y coordinates corresponding to x
     */
    CubicSpline(const std::vector<double>& x, const std::vector<double>& y);
    
    /**
     * @brief Default constructor
     */
    CubicSpline() = default;
    
    /**
     * @brief Set new data points for interpolation
     * @param x X coordinates (must be in ascending order)
     * @param y Y coordinates corresponding to x
     */
    void setPoints(const std::vector<double>& x, const std::vector<double>& y);
    
    /**
     * @brief Interpolate value at given x
     * @param x Query point
     * @return Interpolated y value
     */
    double interpolate(double x) const;
    
    /**
     * @brief Get first derivative at given x
     * @param x Query point
     * @return First derivative value
     */
    double derivative(double x) const;
    
    /**
     * @brief Get second derivative at given x
     * @param x Query point
     * @return Second derivative value
     */
    double secondDerivative(double x) const;
    
    /**
     * @brief Check if spline is initialized
     * @return True if spline has valid data
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Get the range of x values
     * @return Pair of (min_x, max_x)
     */
    std::pair<double, double> getRange() const;

private:
    std::vector<double> x_;  // X coordinates
    std::vector<double> y_;  // Y coordinates
    std::vector<double> a_;  // Coefficients a
    std::vector<double> b_;  // Coefficients b
    std::vector<double> c_;  // Coefficients c
    std::vector<double> d_;  // Coefficients d
    std::vector<double> h_;  // Step sizes
    bool initialized_;
    
    /**
     * @brief Compute spline coefficients
     */
    void computeCoefficients();
    
    /**
     * @brief Find the interval index for given x
     * @param x Query point
     * @return Index of the interval
     */
    size_t findIndex(double x) const;
    
    /**
     * @brief Solve tridiagonal system for second derivatives
     * @param n Size of the system
     * @param h Step sizes
     * @param alpha Right hand side values
     * @return Second derivatives
     */
    std::vector<double> solveTridiagonal(int n, const std::vector<double>& h, 
                                       const std::vector<double>& alpha) const;
};

/**
 * @brief 2D Cubic Spline Path Class
 * 
 * This class handles 2D path interpolation using cubic splines for both x and y coordinates
 * parameterized by arc length.
 */
class CubicSplinePath {
public:
    /**
     * @brief Constructor with waypoints
     * @param x X coordinates of waypoints
     * @param y Y coordinates of waypoints
     */
    CubicSplinePath(const std::vector<double>& x, const std::vector<double>& y);
    
    /**
     * @brief Default constructor
     */
    CubicSplinePath() = default;
    
    /**
     * @brief Set waypoints for the path
     * @param x X coordinates of waypoints
     * @param y Y coordinates of waypoints
     */
    void setWaypoints(const std::vector<double>& x, const std::vector<double>& y);
    
    /**
     * @brief Interpolate position at given arc length
     * @param s Arc length parameter
     * @return Pair of (x, y) coordinates
     */
    std::pair<double, double> interpolatePosition(double s) const;
    
    /**
     * @brief Get yaw angle at given arc length
     * @param s Arc length parameter
     * @return Yaw angle in radians
     */
    double interpolateYaw(double s) const;
    
    /**
     * @brief Get curvature at given arc length
     * @param s Arc length parameter
     * @return Curvature value
     */
    double interpolateCurvature(double s) const;
    
    /**
     * @brief Get total arc length of the path
     * @return Total arc length
     */
    double getTotalLength() const { return total_length_; }
    
    /**
     * @brief Check if path is initialized
     * @return True if path has valid data
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Generate uniformly spaced points along the path
     * @param resolution Distance between points
     * @return Vector of (s, x, y, yaw, curvature) tuples
     */
    std::vector<std::array<double, 5>> generateUniformPoints(double resolution = 0.1) const;

private:
    CubicSpline spline_x_;    // Spline for x coordinates
    CubicSpline spline_y_;    // Spline for y coordinates
    std::vector<double> s_;   // Arc length parameters
    double total_length_;     // Total arc length
    bool initialized_;
    
    /**
     * @brief Calculate cumulative arc lengths
     * @param x X coordinates
     * @param y Y coordinates
     * @return Vector of cumulative arc lengths
     */
    std::vector<double> calculateArcLengths(const std::vector<double>& x, 
                                          const std::vector<double>& y) const;
};

} // namespace lattice_planner

#endif // CUBIC_SPLINE_HPP
