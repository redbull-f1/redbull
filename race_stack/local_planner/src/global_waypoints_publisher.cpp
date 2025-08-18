#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "f110_msgs/msg/wpnt_array.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ament_index_cpp/get_package_share_directory.hpp>

class GlobalWaypointsPublisher : public rclcpp::Node
{
public:
  GlobalWaypointsPublisher() : Node("global_waypoints_publisher")
  {
    declare_parameter<std::string>("csv_file", "/home/ailab0/total_ws/src/race_stack/maps/trajectory/redbull_1.csv");
    declare_parameter<std::string>("frame_id", "map");
    declare_parameter<double>("publish_rate", 2.0);

    csv_file_ = get_parameter("csv_file").as_string();
    frame_id_ = get_parameter("frame_id").as_string();
    double hz = get_parameter("publish_rate").as_double();

    pub_wpnts_ = create_publisher<f110_msgs::msg::WpntArray>("/global_waypoints", 10);
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/global_waypoints/visualization", 10);

    load_waypoints_from_csv();

    using namespace std::chrono_literals;
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(0.1, hz)),
      std::bind(&GlobalWaypointsPublisher::tick, this));

    RCLCPP_INFO(get_logger(), "GlobalWaypointsPublisher started: %zu points", msg_.wpnts.size());
  }

private:
  void load_waypoints_from_csv()
  {
    try {
      std::string full_path = csv_file_;
      std::ifstream file(full_path);
      if (!file.is_open()) {
        RCLCPP_ERROR(get_logger(), "Could not open CSV: %s", full_path.c_str());
        return;
      }
      std::string line;
      int id_counter = 0;
      msg_.wpnts.clear();

      while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        // CSV가 세미콜론(;) 구분자이면 그대로, 콤마(,)면 ','로 바꾸세요.
        while (std::getline(ss, token, ';')) tokens.push_back(token);

        if (tokens.size() >= 3) {
          f110_msgs::msg::Wpnt w;
          w.id = id_counter++;
          w.d_m = 0.0;
          w.ax_mps2 = 0.0;

          w.s_m = std::stod(tokens[0]);
          w.x_m = std::stod(tokens[1]);
          w.y_m = std::stod(tokens[2]);
          if (tokens.size() >= 4) w.d_right = std::stod(tokens[3]);
          if (tokens.size() >= 5) w.d_left  = std::stod(tokens[4]);
          if (tokens.size() >= 6) w.psi_rad = std::stod(tokens[5]);
          if (tokens.size() >= 7) w.kappa_radpm = std::stod(tokens[6]);
          if (tokens.size() >= 8) w.vx_mps = std::stod(tokens[7]);
          if (tokens.size() >= 9) w.ax_mps2 = std::stod(tokens[8]);

          msg_.wpnts.push_back(w);
        }
      }
      file.close();
      RCLCPP_INFO(get_logger(), "Loaded %zu waypoints from %s", msg_.wpnts.size(), csv_file_.c_str());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Load error: %s", e.what());
    }
  }

  visualization_msgs::msg::MarkerArray make_markers()
  {
    visualization_msgs::msg::MarkerArray arr;

    visualization_msgs::msg::Marker clear;
    clear.header.frame_id = frame_id_;
    clear.header.stamp = now();
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);

    if (msg_.wpnts.empty()) return arr;

    // raceline (LINE_STRIP)
    visualization_msgs::msg::Marker line;
    line.header.frame_id = frame_id_;
    line.header.stamp = now();
    line.ns = "raceline";
    line.id = 0;
    line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.scale.x = 0.1;
    line.color.r = 1.0;
    line.color.g = 0.0;
    line.color.b = 0.0;
    line.color.a = 1.0;
    line.pose.orientation.w = 1.0;

    for (const auto& w : msg_.wpnts) {
      geometry_msgs::msg::Point p;
      p.x = w.x_m; p.y = w.y_m; p.z = 0.0;
      line.points.push_back(p);
    }
    arr.markers.push_back(line);

    // points (POINTS)
    visualization_msgs::msg::Marker pts;
    pts.header.frame_id = frame_id_;
    pts.header.stamp = now();
    pts.ns = "waypoints";
    pts.id = 1;
    pts.type = visualization_msgs::msg::Marker::POINTS;
    pts.action = visualization_msgs::msg::Marker::ADD;
    pts.scale.x = 0.05; pts.scale.y = 0.05;
    pts.color.r = 0.5; pts.color.g = 1.0; pts.color.b = 0.5; pts.color.a = 1.0;
    pts.pose.orientation.w = 1.0;

    for (const auto& w : msg_.wpnts) {
      geometry_msgs::msg::Point p;
      p.x = w.x_m; p.y = w.y_m; p.z = 0.0;
      pts.points.push_back(p);
    }
    arr.markers.push_back(pts);

    return arr;
  }

  void tick()
  {
    if (msg_.wpnts.empty()) return;

    msg_.header.frame_id = frame_id_;
    msg_.header.stamp = now();
    pub_wpnts_->publish(msg_);

    auto markers = make_markers();
    pub_markers_->publish(markers);
  }

  // pubs
  rclcpp::Publisher<f110_msgs::msg::WpntArray>::SharedPtr pub_wpnts_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::TimerBase::SharedPtr timer_;

  // data
  f110_msgs::msg::WpntArray msg_;
  std::string csv_file_;
  std::string frame_id_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalWaypointsPublisher>());
  rclcpp::shutdown();
  return 0;
}
