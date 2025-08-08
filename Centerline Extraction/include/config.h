#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace Config {
    // Map settings
    const std::string MAP_PATH = "/home/lmw/Centerline Extraction/input/redbull_0.yaml";
    const std::string MAP_NAME = "redbull_0";
    
    // Image processing settings
    const int MORPH_KERNEL_SIZE = 20;
    
    // Binary threshold settings (override YAML values)
    const bool USE_CUSTOM_THRESHOLD = true;     // true면 아래 값 사용, false면 YAML 파일 값 사용
    const double CUSTOM_OCCUPIED_THRESH = 0.65; // 점유된 것으로 간주할 임계값 (0.0~1.0)
    const double CUSTOM_FREE_THRESH = 0.25;     // 자유공간으로 간주할 임계값 (0.0~1.0)
    
    // Output settings
    const std::string OUTPUT_DIR = "/home/lmw/Centerline Extraction/output/";
    const std::string CSV_FILENAME = "centerline_data.csv";
}

#endif // CONFIG_H
