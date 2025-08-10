#include "image_processor.h"
#include "config.h"
#include <vector>

// Morphological opening filter
cv::Mat ImageProcessor::removeNoise(const cv::Mat& binary_image, int kernel_size) {
    cv::Mat cleaned_image;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(binary_image, cleaned_image, cv::MORPH_OPEN, kernel);
    return cleaned_image;
}

cv::Mat ImageProcessor::applyDistanceTransform(const cv::Mat& cleaned_image) {
    cv::Mat dist_transform;
    cv::distanceTransform(cleaned_image, dist_transform, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    return dist_transform; // 정규화하지 않고 반환
}

// OpenCV의 thinning 알고리즘 사용
cv::Mat ImageProcessor::applyThinning(const cv::Mat& binary_image, int thinning_type) {
    cv::Mat thinned_image;
    cv::ximgproc::thinning(binary_image, thinned_image, thinning_type);
    return thinned_image;
}

cv::Mat ImageProcessor::pruneSpurs(const cv::Mat& skeleton_8u_255) {
    CV_Assert(skeleton_8u_255.type() == CV_8U);

    // 0/255 -> 0/1 (uchar)
    cv::Mat skel01;
    cv::threshold(skeleton_8u_255, skel01, 0, 1, cv::THRESH_BINARY);

    auto computeDegree = [&](cv::Mat& degree32f) {
        // skel01(0/1, U8) -> F32
        cv::Mat s32f; 
        skel01.convertTo(s32f, CV_32F);

        // 3x3 ones (F32) 커널로 8-이웃 합산
        static const cv::Mat k32f = cv::Mat::ones(3, 3, CV_32F);
        cv::Mat neigh32f;
        cv::filter2D(s32f, neigh32f, CV_32F, k32f, cv::Point(-1,-1), 0.0, cv::BORDER_CONSTANT);

        // degree = 이웃수(자기 포함) - 자기자신
        degree32f = neigh32f - s32f;
    };

    while (true) {
        cv::Mat degree32f;
        computeDegree(degree32f);

        // 엔드포인트: degree == 1 && pixel == 1
        cv::Mat endpointsMask;                        // 0/255, U8
        cv::compare(degree32f, 1.0, endpointsMask, cv::CMP_EQ);

        cv::Mat skelMask255;                          // skel01을 0/255로
        skel01.convertTo(skelMask255, CV_8U, 255.0);
        cv::bitwise_and(endpointsMask, skelMask255, endpointsMask);

        if (cv::countNonZero(endpointsMask) == 0)
            break;

        // 0/255 -> 0/1로 변환해서 엔드포인트 제거
        cv::Mat endpoints01; 
        endpointsMask.convertTo(endpoints01, CV_8U, 1.0/255.0);
        skel01 -= endpoints01;                        // endpoints 위치만 1->0
    }

    cv::Mat pruned;
    skel01.convertTo(pruned, CV_8U, 255.0);
    return pruned;
}



// Moving Average 커널 생성 (window 크기)
std::vector<double> ImageProcessor::movingAverageKernel(int window) {
    // 중앙 평균 커널
    std::vector<double> kernel(window, 1.0 / window);
    return kernel;
}

std::vector<double> ImageProcessor::circularFilter(const std::vector<double>& data, const std::vector<double>& kernel) {
    int N = data.size(), W = kernel.size();
    std::vector<double> result(N, 0.0);
    int half = W / 2;
    for (int i = 0; i < N; ++i) {
        double sum = 0;
        for (int k = 0; k < W; ++k) {
            int idx = (i - half + k + N) % N; // 원형 인덱스
            sum += data[idx] * kernel[k];
        }
        result[i] = sum;
    }
    return result;
}


// Moving Average 기반 중심선 스무딩
cv::Mat ImageProcessor::smoothCenterlineMovingAverage(const cv::Mat& skeleton, int window) {
    CV_Assert(skeleton.type() == CV_8U);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(skeleton, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Mat smoothed = cv::Mat::zeros(skeleton.size(), CV_8U);
    for (const auto& contour : contours) {
        if (contour.size() < window) continue;

        // x, y 좌표 분리
        std::vector<double> xs, ys;
        for (const auto& pt : contour) {
            xs.push_back(pt.x);
            ys.push_back(pt.y);
        }

        // Moving Average 커널 생성 및 적용
        auto kernel = movingAverageKernel(window);
        auto xs_smooth = circularFilter(xs, kernel);
        auto ys_smooth = circularFilter(ys, kernel);

        // 스무딩된 좌표로 곡선 그리기
        std::vector<cv::Point> smooth_pts;
        for (size_t i = 0; i < xs_smooth.size(); ++i)
            smooth_pts.emplace_back((int)std::round(xs_smooth[i]), (int)std::round(ys_smooth[i]));
        if (smooth_pts.size() > 1)
            cv::polylines(smoothed, smooth_pts, true, cv::Scalar(255), 1, cv::LINE_AA);
    }
    return smoothed;
}

// DFS로 중심선 순서 정리
std::vector<cv::Point> ImageProcessor::orderCenterlineByDFS(const cv::Mat& centerline) {
    std::vector<cv::Point> ordered_points;
    cv::Mat visited = cv::Mat::zeros(centerline.size(), CV_8U);
    
    // 시작점 찾기 (첫 번째 비영 픽셀)
    cv::Point start(-1, -1);
    for (int y = 0; y < centerline.rows && start.x == -1; ++y) {
        for (int x = 0; x < centerline.cols; ++x) {
            if (centerline.at<uchar>(y, x) > 0) {
                start = cv::Point(x, y);
                break;
            }
        }
    }
    
    if (start.x == -1) return ordered_points;
    
    // DFS로 연결된 픽셀들 순차 방문
    std::function<void(cv::Point)> dfs = [&](cv::Point current) {
        if (visited.at<uchar>(current.y, current.x)) return;
        
        visited.at<uchar>(current.y, current.x) = 255;
        ordered_points.push_back(current);
        
        // 8방향 이웃 확인
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                cv::Point next(current.x + dx, current.y + dy);
                
                if (next.x >= 0 && next.x < centerline.cols && 
                    next.y >= 0 && next.y < centerline.rows &&
                    centerline.at<uchar>(next.y, next.x) > 0 &&
                    !visited.at<uchar>(next.y, next.x)) {
                    dfs(next);
                }
            }
        }
    };
    
    dfs(start);
    return ordered_points;
}


