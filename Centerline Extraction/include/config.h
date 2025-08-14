#ifndef CONFIG_H
#define CONFIG_H
#include <string>

namespace Config {
    // Map settings
    const std::string MAP_PATH = "../input/";
    const std::string MAP_NAME = "redbull_0";

    // Occupancy thresholds
    const int OCCUPIED_THRESH = 220; // 점유된 것으로 간주할 픽셀값 임계값
    const int FREE_THRESH = 250;     // 자유공간으로 간주할 픽셀값 임계값

    // Image processing
    const int MORPH_KERNEL_SIZE = 20;
    const int THINNING_TYPE = 1; // 1: GUOHALL, 2: ZHANGSUEN
    const double DIST_TRANSFORM_THRESH = 2.0;

    // SG Smoothing
    const int SG_WINDOW = 7;
    const int SG_POLY_ORDER = 3;

    // Waypoint
    const double WAYPOINT_INTERVAL = 2.0;

    // Output
    const std::string OUTPUT_BASE = "../output/";
    const std::string CSV_FILENAME = "centerline.csv";
}

#endif // CONFIG_H
