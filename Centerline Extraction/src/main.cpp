#include "occupancy_grid_loader.h"
#include "image_processor.h"
#include "config.h"
#include <iostream>

int main() {
    // 맵 로드
    OccupancyGrid loaded_map = OccupancyGridLoader::loadMap(Config::MAP_PATH);
    cv::Mat binary_map = loaded_map.binary_map;
    cv::Mat original_image = loaded_map.image;
    double resolution = loaded_map.metadata.resolution;
    
    // Morphological opening으로 맵 외부 노이즈 제거
    cv::Mat cleaned_binary_map = ImageProcessor::removeNoise(binary_map, Config::MORPH_KERNEL_SIZE);
    
    // Skeleton 추출
    cv::Mat skeleton = ImageProcessor::skeletonize(cleaned_binary_map, resolution);

    // 시각화
    cv::imshow("Binary Map", binary_map);
    cv::imshow("Cleaned Binary Map", cleaned_binary_map);
    cv::imshow("Skeleton", skeleton);
    
    cv::waitKey(0);
    return 0;
}