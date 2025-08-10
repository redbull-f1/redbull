#include "occupancy_grid_loader.h"
#include "config.h"
#include "config.h"
#include <iostream>

OccupancyGrid OccupancyGridLoader::loadMap(const std::string& yaml_file_path) {
    OccupancyGrid grid;
    
    YAML::Node config = YAML::LoadFile(yaml_file_path);
    
    grid.metadata.image_path = config["image"].as<std::string>();
    grid.metadata.resolution = config["resolution"].as<double>();
    grid.metadata.negate = config["negate"].as<int>();
    grid.metadata.occupied_thresh = config["occupied_thresh"].as<double>();
    grid.metadata.free_thresh = config["free_thresh"].as<double>();
    grid.metadata.origin = config["origin"].as<std::vector<double>>();
    
    std::string yaml_dir = yaml_file_path.substr(0, yaml_file_path.find_last_of("/\\"));
    std::string full_image_path = yaml_dir + "/" + grid.metadata.image_path;
    
    grid.image = cv::imread(full_image_path, cv::IMREAD_GRAYSCALE);
    createBinaryMap(grid);
    
    return grid;
}

void OccupancyGridLoader::createBinaryMap(OccupancyGrid& grid) {
    grid.binary_map = cv::Mat::zeros(grid.image.size(), CV_8UC1);
    
    // Config에서 픽셀값 기준 임계값 사용
    int pixel_occupied_thresh = Config::OCCUPIED_THRESH;
    int pixel_free_thresh = Config::FREE_THRESH;
    
    std::cout << "픽셀값 기준 임계값 - Occupied(이하): " << pixel_occupied_thresh 
              << ", Free(이상): " << pixel_free_thresh << std::endl;
    
    for (int y = 0; y < grid.image.rows; y++) {
        for (int x = 0; x < grid.image.cols; x++) {
            uchar pixel_value = grid.image.at<uchar>(y, x);
            
            // negate 처리 (필요한 경우)
            uchar processed_value = pixel_value;
            if (grid.metadata.negate) {
                processed_value = 255 - pixel_value;
            }
            
            if (processed_value >= pixel_free_thresh) {
                grid.binary_map.at<uchar>(y, x) = 255; // 자유공간 -> 흰색
            } else if (processed_value <= pixel_occupied_thresh) {
                grid.binary_map.at<uchar>(y, x) = 0;   // 점유됨(벽) -> 검은색
            } else {
                // 중간값은 uncertain area - 보통 벽으로 처리
                grid.binary_map.at<uchar>(y, x) = 0;   // 벽으로 처리 -> 검은색
            }
        }
    }
}

cv::Point2d OccupancyGridLoader::pixelToWorld(const cv::Point2i& pixel, const MapMetadata& metadata, int image_rows) {
    cv::Point2d world;
    world.x = metadata.origin[0] + pixel.x * metadata.resolution;
    world.y = metadata.origin[1] + (image_rows - pixel.y - 1) * metadata.resolution;
    return world;
}

void OccupancyGridLoader::analyzeImageHistogram(const cv::Mat& image) {
    // 히스토그램 계산
    std::vector<cv::Mat> images = {image};
    cv::Mat hist;
    int histSize = 256;
    std::vector<int> channels = {0};
    std::vector<int> hist_sizes = {histSize};
    std::vector<float> ranges = {0, 256};
    
    cv::calcHist(images, channels, cv::Mat(), hist, hist_sizes, ranges);
    
    // 기본 정보 출력
    std::cout << "\n=== 이미지 밝기 히스토그램 분석 ===" << std::endl;
    std::cout << "이미지 크기: " << image.cols << "x" << image.rows << std::endl;
    
    // 전체 픽셀 수
    int total_pixels = image.cols * image.rows;
    std::cout << "전체 픽셀 수: " << total_pixels << std::endl;
    
    // 각 밝기값별 픽셀 수와 비율 (상위 3개만)
    std::vector<std::pair<int, int>> brightness_counts;
    
    for (int i = 0; i < histSize; i++) {
        int count = static_cast<int>(hist.at<float>(i));
        if (count > 0) {
            brightness_counts.push_back({i, count});
        }
    }
    
    // 픽셀 수로 정렬 (내림차순)
    std::sort(brightness_counts.begin(), brightness_counts.end(), 
              [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
                  return a.second > b.second;
              });
    
    std::cout << "\n상위 3개 밝기값:" << std::endl;
    for (int i = 0; i < std::min(3, static_cast<int>(brightness_counts.size())); i++) {
        int brightness = brightness_counts[i].first;
        int count = brightness_counts[i].second;
        double percentage = (count * 100.0) / total_pixels;
        std::cout << "밝기 " << brightness << ": " << count << "개 (" 
                  << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
    }
    
    std::cout << "================================\n" << std::endl;
}