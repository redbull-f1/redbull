#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>
#include <functional>

// Waypoint 구조체
struct Waypoint {
    double arc_distance;     // 시작점부터의 누적 거리
    cv::Point2f center;      // 중심점 (보간된 정확한 위치)
    cv::Point2f tangent;     // 접선 방향 (정규화됨)
    cv::Point2f normal;      // 법선 방향 (정규화됨)
    double left_width;       // 좌측 트랙 폭
    double right_width;      // 우측 트랙 폭
};

class ImageProcessor {
public:
    // Morphological opening으로 노이즈 제거
    static cv::Mat removeNoise(const cv::Mat& binary_image, int kernel_size);
    
    // Euclidean distance transform 적용
    static cv::Mat applyDistanceTransform(const cv::Mat& cleaned_image);
    
    // OpenCV의 thinning 알고리즘 (Zhang-Suen 또는 Guo-Hall)
    static cv::Mat applyThinning(const cv::Mat& binary_image, int thinning_type = cv::ximgproc::THINNING_ZHANGSUEN);

    // 스퍼(짧은 가지) 제거
    static cv::Mat pruneSpurs(const cv::Mat& skeleton_8u_255);

    // 중심선 Moving Average 스무딩
    static cv::Mat smoothCenterlineMovingAverage(const cv::Mat& skeleton, int window = 7);

    // Moving Average 커널 생성
    static std::vector<double> movingAverageKernel(int window);
    // 원형 필터 적용
    static std::vector<double> circularFilter(const std::vector<double>& data, const std::vector<double>& kernel);

    // Waypoint 생성
    static std::vector<cv::Point> orderCenterlineByDFS(const cv::Mat& centerline);
};

#endif // IMAGE_PROCESSOR_H
