#ifndef OCCUPANCY_GRID_LOADER_H
#define OCCUPANCY_GRID_LOADER_H

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include "config.h"
#include <filesystem>

struct MapMetadata {
    std::string image_path;
    double resolution;
    std::vector<double> origin;
    int negate;
    double occupied_thresh;
    double free_thresh;
};

struct OccupancyGrid {
    cv::Mat image;
    cv::Mat binary_map;
    MapMetadata metadata;
};

class OccupancyGridLoader {
public:
    static OccupancyGrid loadMap(const std::string& yaml_file_path);
    static void createBinaryMap(OccupancyGrid& grid);
    static cv::Point2d pixelToWorld(const cv::Point2i& pixel, const MapMetadata& metadata, int image_rows);
    static void analyzeImageHistogram(const cv::Mat& image);

private:
};

#endif // OCCUPANCY_GRID_LOADER_H