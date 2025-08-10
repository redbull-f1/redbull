#include "visualization_utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib>

namespace VisualizationUtils {
    cv::Mat overlayCenterline(const cv::Mat& original, const cv::Mat& centerline, cv::Scalar color) {
        cv::Mat vis;
        if (original.channels() == 1)
            cv::cvtColor(original, vis, cv::COLOR_GRAY2BGR);
        else
            vis = original.clone();
            
        for (int y = 0; y < centerline.rows; ++y) {
            for (int x = 0; x < centerline.cols; ++x) {
                if (centerline.at<uchar>(y, x) > 0) {
                    vis.at<cv::Vec3b>(y, x)[0] = color[0];
                    vis.at<cv::Vec3b>(y, x)[1] = color[1];
                    vis.at<cv::Vec3b>(y, x)[2] = color[2];
                }
            }
        }
        return vis;
    }

    void saveImages(const std::vector<cv::Mat>& images, const std::vector<std::string>& filenames, const std::string& output_dir) {
        // output 디렉토리가 없으면 생성
        std::string cmd = "mkdir -p " + output_dir;
        system(cmd.c_str());
        
        for (size_t i = 0; i < images.size() && i < filenames.size(); ++i) {
            if (images[i].empty()) continue;
            
            std::string filepath = output_dir + "/" + filenames[i];
            cv::imwrite(filepath, images[i]);
            std::cout << "Saved: " << filepath << std::endl;
        }
    }
}