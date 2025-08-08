#include "image_processor.h"
#include "config.h"

// Morphological opening filter
cv::Mat ImageProcessor::removeNoise(const cv::Mat& binary_image, int kernel_size) {
    cv::Mat cleaned_image;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(binary_image, cleaned_image, cv::MORPH_OPEN, kernel);
    return cleaned_image;
}

cv::Mat ImageProcessor::skeletonize(const cv::Mat& binary_map, double resolution) {
    cv::Mat skeleton;
    cv::ximgproc::thinning(binary_map, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
    return skeleton;
}