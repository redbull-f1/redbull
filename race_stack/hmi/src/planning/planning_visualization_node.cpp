#include "HMI/planning/planning_visualization_node.hpp"

PlanningVisualizationNode::PlanningVisualizationNode(const std::string &node_name, const rclcpp::NodeOptions &options)
    : Node(node_name, options) {
    
    // QoS init
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
    
    // Parameters
    this->declare_parameter("planning_viz/loop_rate_hz", 10.0);
    this->declare_parameter("planning_viz/centerline_csv_file", "redbull_1_centerline.csv");

    ProcessParams();
    RCLCPP_INFO(this->get_logger(), "loop_rate_hz: %f", loop_rate_hz_);
    RCLCPP_INFO(this->get_logger(), "centerline_csv_file: %s", centerline_csv_file_.c_str());
    
    // Load centerline from CSV
    if (LoadCenterlineFromCSV(centerline_csv_file_)) {
        RCLCPP_INFO(this->get_logger(), "Successfully loaded %zu centerline points from CSV", centerline_data_.size());
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to load centerline from CSV file");
    }
    
    // Subscribers init
    s_global_waypoints_ = this->create_subscription<f110_msgs::msg::WpntArray>(
        "/global_waypoints", qos_profile, std::bind(&PlanningVisualizationNode::CallbackGlobalWaypoints, this, std::placeholders::_1));
    s_local_waypoints_ = this->create_subscription<f110_msgs::msg::WpntArray>(
        "/local_waypoints", qos_profile, std::bind(&PlanningVisualizationNode::CallbackLocalWaypoints, this, std::placeholders::_1));
    
    // Publisher init
    p_centerline_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planning/centerline_markers", qos_profile);
    p_global_waypoints_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planning/global_waypoints_markers", qos_profile);
    p_local_waypoints_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planning/local_waypoints_markers", qos_profile);
    
    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / loop_rate_hz_)),
        [this]() { this->Run(); });
}

PlanningVisualizationNode::~PlanningVisualizationNode() {}

void PlanningVisualizationNode::ProcessParams() {
    this->get_parameter("planning_viz/loop_rate_hz", loop_rate_hz_);
    this->get_parameter("planning_viz/centerline_csv_file", centerline_csv_file_);
}

void PlanningVisualizationNode::Run() {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 5000, "Planning Visualization Running ...");
    ProcessParams();

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Check waypoints data availability
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    if (b_is_global_waypoints_ == false) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 2000, "Waiting for global waypoints...");
    }
    
    if (b_is_local_waypoints_ == false) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 2000, "Waiting for local waypoints...");
    }

    // Publish centerline markers (always available)
    PublishCenterlineMarkers();
    
    // Publish waypoints markers only if data is available
    if (b_is_global_waypoints_) {
        PublishGlobalWaypointsMarkers();
    }
    
    if (b_is_local_waypoints_) {
        PublishLocalWaypointsMarkers();
    }
}

bool PlanningVisualizationNode::LoadCenterlineFromCSV(const std::string& csv_filename) {
    try {
        // Get the package share directory
        std::string package_share_directory = ament_index_cpp::get_package_share_directory("hmi");
        std::string csv_path = package_share_directory + "/data/" + csv_filename;
        
        RCLCPP_INFO(this->get_logger(), "Loading centerline from: %s", csv_path.c_str());
        
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open CSV file: %s", csv_path.c_str());
            return false;
        }
        
        centerline_data_.clear();
        std::string line;
        
        // Skip header lines that start with '#'
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            
            // Parse CSV line (x_m,y_m,w_tr_right_m,w_tr_left_m)
            while (std::getline(ss, cell, ',')) {
                try {
                    // Trim whitespace
                    cell.erase(0, cell.find_first_not_of(" \t"));
                    cell.erase(cell.find_last_not_of(" \t") + 1);
                    double value = std::stod(cell);
                    row.push_back(value);
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Error parsing CSV value: %s", cell.c_str());
                    return false;
                }
            }
            
            // Expect 4 columns: x_m, y_m, w_tr_right_m, w_tr_left_m
            if (row.size() == 4) {
                centerline_data_.push_back(row);
            } else {
                RCLCPP_WARN(this->get_logger(), "Skipping invalid CSV line with %zu columns", row.size());
            }
        }
        
        file.close();
        
        if (centerline_data_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No valid centerline points found in CSV file");
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception while loading CSV: %s", e.what());
        return false;
    }
}

void PlanningVisualizationNode::PublishCenterlineMarkers() {
    if (centerline_data_.empty()) {
        return;
    }
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // Create line strip marker for centerline
    visualization_msgs::msg::Marker centerline_marker;
    centerline_marker.header.frame_id = "map";
    centerline_marker.header.stamp = this->now();
    centerline_marker.ns = "centerline";
    centerline_marker.id = 0;
    centerline_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    centerline_marker.action = visualization_msgs::msg::Marker::ADD;
    
    centerline_marker.scale.x = 0.1;   // Moderate line width for centerline
    centerline_marker.color.r = 0.2;   // Dark blue color for centerline
    centerline_marker.color.g = 0.5;
    centerline_marker.color.b = 0.9;
    centerline_marker.color.a = 0.8;   // Semi-opaque for better visibility
    
    // Add centerline points to line strip
    for (const auto& row : centerline_data_) {
        geometry_msgs::msg::Point point;
        point.x = row[0];  // x_m
        point.y = row[1];  // y_m
        point.z = 0.0;
        centerline_marker.points.push_back(point);
    }
    
    marker_array.markers.push_back(centerline_marker);
    
    p_centerline_markers_->publish(marker_array);
    
    RCLCPP_DEBUG(this->get_logger(), "Published centerline markers with %zu points", centerline_data_.size());
}

void PlanningVisualizationNode::PublishGlobalWaypointsMarkers() {
    std::lock_guard<std::mutex> lock(mutex_global_waypoints_);
    
    if (!b_is_global_waypoints_ || global_waypoints_.wpnts.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "No global waypoints data available for visualization");
        return;
    }
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // Create line strip marker for global waypoints path
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = "map";
    line_marker.header.stamp = this->now();
    line_marker.ns = "global_waypoints";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    
    line_marker.scale.x = 0.12;  // Reasonable line width for global waypoints
    line_marker.color.r = 1.0;   // Red color 
    line_marker.color.g = 0.0;
    line_marker.color.b = 0.0;
    line_marker.color.a = 0.6;   // Semi-transparent for background appearance
    
    // Add waypoint positions to line strip (at ground level)
    for (const auto& waypoint : global_waypoints_.wpnts) {
        geometry_msgs::msg::Point point;
        point.x = waypoint.x_m;
        point.y = waypoint.y_m;
        point.z = 0.0;  // Ground level
        line_marker.points.push_back(point);
    }
    
    marker_array.markers.push_back(line_marker);
    
    p_global_waypoints_markers_->publish(marker_array);
    
    RCLCPP_DEBUG(this->get_logger(), "Published global waypoints markers with %zu points", global_waypoints_.wpnts.size());
}

void PlanningVisualizationNode::PublishLocalWaypointsMarkers() {
    std::lock_guard<std::mutex> lock(mutex_local_waypoints_);
    
    if (!b_is_local_waypoints_ || local_waypoints_.wpnts.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "No local waypoints data available for visualization");
        return;
    }
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // Create line strip marker for local waypoints path
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = "map";
    line_marker.header.stamp = this->now();
    line_marker.ns = "local_waypoints";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    
    line_marker.scale.x = 0.15;  // Thicker than global but not too thick
    line_marker.color.r = 0.0;   // Bright green color for prominence
    line_marker.color.g = 1.0;
    line_marker.color.b = 0.0;
    line_marker.color.a = 0.9;   // More opaque for prominence
    
    // Add local waypoint positions to line strip (slightly elevated)
    for (const auto& waypoint : local_waypoints_.wpnts) {
        geometry_msgs::msg::Point point;
        point.x = waypoint.x_m;
        point.y = waypoint.y_m;
        point.z = 0.05;  // Slightly elevated to show above global waypoints
        line_marker.points.push_back(point);
    }
    
    marker_array.markers.push_back(line_marker);
    
    p_local_waypoints_markers_->publish(marker_array);
    
    RCLCPP_DEBUG(this->get_logger(), "Published local waypoints markers with %zu points", local_waypoints_.wpnts.size());
}

int main(int argc, char **argv) {
    std::string node_name = "planning_visualization_node";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlanningVisualizationNode>(node_name));
    rclcpp::shutdown();
    return 0;
}
