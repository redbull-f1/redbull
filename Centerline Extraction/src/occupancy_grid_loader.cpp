#include "occupancy_grid_loader.h"
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
    
    // Config에서 임계값 설정 사용 여부 확인
    double occupied_thresh = grid.metadata.occupied_thresh;
    double free_thresh = grid.metadata.free_thresh;
    
    if (Config::USE_CUSTOM_THRESHOLD) {
        occupied_thresh = Config::CUSTOM_OCCUPIED_THRESH;
        free_thresh = Config::CUSTOM_FREE_THRESH;
        std::cout << "Using custom thresholds - Occupied: " << occupied_thresh 
                  << ", Free: " << free_thresh << std::endl;
    } else {
        std::cout << "Using YAML thresholds - Occupied: " << occupied_thresh 
                  << ", Free: " << free_thresh << std::endl;
    }
    
    for (int y = 0; y < grid.image.rows; y++) {
        for (int x = 0; x < grid.image.cols; x++) {
            uchar pixel_value = grid.image.at<uchar>(y, x);
            double prob = getOccupancyProbability(pixel_value, grid.metadata);
            
            if (prob >= occupied_thresh) {
                grid.binary_map.at<uchar>(y, x) = 255; // 점유됨 -> 벽 (흰색)
            } else {
                grid.binary_map.at<uchar>(y, x) = 0;   // 자유공간 -> 트랙 (검은색)
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

double OccupancyGridLoader::getOccupancyProbability(uchar pixel_value, const MapMetadata& metadata) {
    double normalized = pixel_value / 255.0;
    if (metadata.negate) {
        normalized = 1.0 - normalized;
    }
    return normalized;
}