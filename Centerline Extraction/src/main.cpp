#include "occupancy_grid_loader.h"
#include "image_processor.h"
#include "config.h"
#include "visualization_utils.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <sys/stat.h> // for mkdir

int main() {
    // 0) 맵 로드 + 이진화
    const std::string yaml_path = Config::MAP_PATH + Config::MAP_NAME + ".yaml";
    std::cout << "Loading YAML: " << yaml_path << std::endl;
    OccupancyGrid loaded_map = OccupancyGridLoader::loadMap(yaml_path);
    cv::Mat binary_map = loaded_map.binary_map;   // 전경=255, 배경=0 가정
    cv::Mat original_image = loaded_map.image;

    // 1) 노이즈 제거 
    cv::Mat cleaned_binary_map = ImageProcessor::removeNoise(binary_map, Config::MORPH_KERNEL_SIZE);
    if (cleaned_binary_map.type() != CV_8U) {
        cleaned_binary_map.convertTo(cleaned_binary_map, CV_8U, 255.0);
    }
    cv::threshold(cleaned_binary_map, cleaned_binary_map, 127, 255, cv::THRESH_BINARY);

    // 2) Euclidean Distance Transform
    cv::Mat dist_transform = ImageProcessor::applyDistanceTransform(cleaned_binary_map);
    const double Tpx = 2.0;
    cv::Mat centers_mask_u8;
    cv::compare(dist_transform, Tpx, centers_mask_u8, cv::CMP_GT);  // 결과: 0/255, CV_8U

    // 3) Skeletonize
    cv::Mat thinned = ImageProcessor::applyThinning(centers_mask_u8, Config::THINNING_TYPE);

    // 4) 스퍼(짧은 가지) 제거
    cv::Mat pruned = ImageProcessor::pruneSpurs(thinned);

    // 5) 중심선 Moving Average 스무딩
    cv::Mat smoothed = ImageProcessor::smoothCenterlineMovingAverage(pruned, 7);

    // 각 처리 단계별 이미지 저장
    cv::Mat dist_vis;
    cv::normalize(dist_transform, dist_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 중심선 오버레이 이미지 미리 생성
    cv::Mat overlay = VisualizationUtils::overlayCenterline(original_image, smoothed, cv::Scalar(0,0,255));

    std::vector<cv::Mat> images = {
        original_image,         // 원본 맵
        binary_map,             // 이진화 맵
        cleaned_binary_map,     // 노이즈 제거 맵
        dist_vis,               // 거리 변환 시각화
        thinned,                // 중심선 추출
        pruned,                 // 스퍼 제거
        smoothed,               // 중심선 스무딩
        overlay                 // 중심선 오버레이
    };

    std::vector<std::string> filenames = {
        "01_original_map.png",
        "02_binary_map.png",
        "03_cleaned_map.png",
        "04_distance_transform.png",
        "05_thinned.png",
        "06_pruned.png",
        "07_smoothed.png",
        "08_overlay_centerline.png"
    };

    // 동적으로 output 디렉터리 경로 생성
    std::string output_dir = Config::OUTPUT_BASE + Config::MAP_NAME;
    system(("mkdir -p " + output_dir).c_str());

    // 이미지 저장 (지정 경로)
    VisualizationUtils::saveImages(images, filenames, output_dir);

    // DFS로 센터라인 픽셀 좌표 순서대로 추출
    std::vector<cv::Point> ordered_centerline = ImageProcessor::orderCenterlineByDFS(smoothed);

    // Distance Transform을 cleaned map에 적용
    cv::Mat dist_transform_cleaned = ImageProcessor::applyDistanceTransform(cleaned_binary_map);

    // 센터라인 좌표를 순서대로 CSV 저장
    std::ofstream waypoint_file(output_dir + "/" + Config::CSV_FILENAME);
    waypoint_file << "# x_m,y_m,w_tr_right_m,w_tr_left_m\n";
    for (const auto& pt : ordered_centerline) {
        // 픽셀 -> 맵 좌표 변환
        cv::Point2d world_pos = OccupancyGridLoader::pixelToWorld(pt, loaded_map.metadata, loaded_map.image.rows);
        // Distance Transform 값 (센터라인 픽셀에서의 거리)
        double width_m = dist_transform_cleaned.at<float>(pt.y, pt.x) * loaded_map.metadata.resolution;
        waypoint_file << std::fixed << std::setprecision(4)
                      << world_pos.x << "," << world_pos.y << ","
                      << width_m << "," << width_m << "\n";
    }
    waypoint_file.close();

    return 0;
}
