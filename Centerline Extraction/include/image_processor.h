#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

class ImageProcessor {
public:
    // Morphological opening으로 노이즈 제거
    static cv::Mat removeNoise(const cv::Mat& binary_image, int kernel_size);
    
    // Skeleton 추출
    static cv::Mat skeletonize(const cv::Mat& binary_image, double resolution);
};

#endif // IMAGE_PROCESSOR_H
