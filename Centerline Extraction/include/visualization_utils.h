#ifndef VISUALIZATION_UTILS_H
#define VISUALIZATION_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace VisualizationUtils {
    // 원본 맵에 중심선(스무딩) 오버레이
    cv::Mat overlayCenterline(const cv::Mat& original, const cv::Mat& centerline, cv::Scalar color = cv::Scalar(0,0,255));
    
    // 이미지들을 output 폴더에 저장
    void saveImages(const std::vector<cv::Mat>& images, const std::vector<std::string>& filenames, const std::string& output_dir = "output");
}

#endif // VISUALIZATION_UTILS_H