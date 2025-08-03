#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import time

#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import time

'''
TinyCenterSpeed를 사용해서 LiDAR 스캔 데이터를 처리하고 동적 차량을 탐지하는 ROS2 노드입니다.
sub : /scan
pub : /objects (MarkerArray), /objects_data (Float32MultiArray)
'''

# Add the src directory to the path for model imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if os.path.exists(src_path):
    sys.path.insert(0, src_path)
else:
    # Try to find the package in the install directory
    try:
        import ament_index_python
        package_share_directory = ament_index_python.get_package_share_directory('redbull')
        sys.path.insert(0, package_share_directory)
    except:
        pass

try:
    from models.CenterSpeed import CenterSpeedDenseResidual, CenterSpeedModular
except ImportError as e:
    print(f"Warning: Could not import CenterSpeed models: {e}")
    # Try alternative import paths
    try:
        sys.path.append(os.path.dirname(__file__))
        from src.models.CenterSpeed import CenterSpeedDenseResidual, CenterSpeedModular
    except ImportError as e2:
        print(f"Warning: Alternative import also failed: {e2}")
        CenterSpeedDenseResidual = None
        CenterSpeedModular = None

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, Float32MultiArray

class DetectedObject:
    """Detected dynamic vehicle object"""
    def __init__(self, x, y, vx, vy, yaw):
        self.x = float(x)
        self.y = float(y) 
        self.vx = float(vx)
        self.vy = float(vy)
        self.yaw = float(yaw)
        self.size = 0.5  # Default size for visualization

class DynamicVehicleDetector(Node):
    """ROS2 node for detecting dynamic vehicles from LiDAR scan data using TinyCenterSpeed"""
    
    def __init__(self):
        super().__init__('dynamic_vehicle_detector')
        
        # Parameters
        self.declare_parameter('model_path', 'src/trained_models/TinyCenterSpeed.pt')
        self.declare_parameter('image_size', 64)
        self.declare_parameter('dense', True)
        self.declare_parameter('num_opponents', 5)
        self.declare_parameter('detection_threshold', 0.3)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.image_size = self.get_parameter('image_size').value
        self.dense = self.get_parameter('dense').value
        self.num_opponents = self.get_parameter('num_opponents').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        
        # CenterSpeed parameters
        self.pixelsize = 0.1  # size of a pixel in meters
        self.feature_size = 3  # number of features in the preprocessed data
        self.origin_offset = (self.image_size // 2) * self.pixelsize
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        self.load_model()
        
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )
        
        # Publishers
        self.objects_marker_pub = self.create_publisher(
            MarkerArray,
            '/objects',
            qos_profile
        )
        
        # Publish object data as Float32MultiArray [x, y, vx, vy, yaw] for each object
        self.objects_data_pub = self.create_publisher(
            Float32MultiArray,
            '/objects_data',
            qos_profile
        )
        
        # Initialize variables
        self.scan_data = None
        self.frame1 = None
        self.frame2 = None
        
        # Create timer for processing (40Hz to match CenterSpeed)
        self.timer = self.create_timer(1.0 / 40.0, self.process_scan)
        
        self.get_logger().info('TinyCenterSpeed Dynamic Vehicle Detector initialized!')
        self.get_logger().info(f'Using device: {self.device}')
        self.get_logger().info(f'Model path: {self.model_path}')
        
    def load_model(self):
        """Load the TinyCenterSpeed model"""
        try:
            # Check if models are available
            if CenterSpeedDenseResidual is None or CenterSpeedModular is None:
                self.get_logger().error("CenterSpeed models not available")
                self.net = None
                return
            
            # Try multiple paths for the model file
            model_paths = [
                os.path.join(os.path.dirname(__file__), self.model_path),  # Original path
                os.path.join(os.path.dirname(__file__), 'src', 'trained_models', 'TinyCenterSpeed.pt'),  # src relative
                self.model_path,  # Direct path
            ]
            
            # Try to find model in package share directory
            try:
                import ament_index_python
                package_share_directory = ament_index_python.get_package_share_directory('redbull')
                model_paths.append(os.path.join(package_share_directory, 'trained_models', 'TinyCenterSpeed.pt'))
            except:
                pass
            
            model_full_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_full_path = path
                    break
            
            if model_full_path is None:
                self.get_logger().error(f"Model file not found in any of these paths: {model_paths}")
                self.net = None
                return
                
            if self.dense:
                self.net = CenterSpeedDenseResidual(image_size=self.image_size)
            else:
                self.net = CenterSpeedModular(image_size=self.image_size)
                
            self.net.load_state_dict(torch.load(model_full_path, map_location=self.device, weights_only=True))
            self.net.eval()
            self.net.to(self.device)
            
            self.get_logger().info(f'TinyCenterSpeed model loaded successfully from: {model_full_path}')
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.net = None
    
    def scan_callback(self, msg):
        """Callback for LaserScan messages"""
        self.scan_data = msg
        
    def preprocess_scan(self, scan_msg):
        """Preprocess scan data using CenterSpeed format"""
        try:
            lidar_data = np.array(scan_msg.ranges, dtype=np.float32)
            
            # Handle intensities if available
            if hasattr(scan_msg, 'intensities') and len(scan_msg.intensities) > 0:
                intensities = np.array(scan_msg.intensities, dtype=np.float32)
                if intensities.max() > intensities.min():
                    intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
                else:
                    intensities = np.ones_like(lidar_data, dtype=np.float32)
            else:
                intensities = np.ones_like(lidar_data, dtype=np.float32)
            
            # Calculate angles
            angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(lidar_data))
            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)
            
            # Preprocess the lidar data
            preprocessed_scans = np.zeros((1, self.feature_size, self.image_size, self.image_size), dtype=np.float32)
            x = lidar_data * cos_angles
            y = lidar_data * sin_angles
            
            # Filter out points behind x = -0.5m (exclude rear area)
            forward_filter = x >= 0
            x = x[forward_filter]
            y = y[forward_filter]
            intensities = intensities[forward_filter]
            
            x_coord = ((x + self.origin_offset) / self.pixelsize).astype(int)
            y_coord = ((y + self.origin_offset) / self.pixelsize).astype(int)
            
            valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
            x_coord = x_coord[valid_indices]
            y_coord = y_coord[valid_indices]
            
            if len(x_coord) > 0:
                preprocessed_scans[:, 0, y_coord, x_coord] = 1  # set the pixel to occupied
                preprocessed_scans[:, 1, y_coord, x_coord] = np.maximum(
                    preprocessed_scans[:, 1, y_coord, x_coord], 
                    intensities[valid_indices]
                )  # store the maximum intensity value
                preprocessed_scans[:, 2, y_coord, x_coord] += 1  # count the number of points
            
            return preprocessed_scans
            
        except Exception as e:
            self.get_logger().error(f"Failed to preprocess scan: {e}")
            return None

    def index_to_cartesian(self, x_img, y_img):
        """Convert image coordinates to cartesian coordinates (CenterSpeed format)"""
        x = x_img * self.pixelsize - self.origin_offset
        y = y_img * self.pixelsize - self.origin_offset
        return x, y
    
    def find_k_peaks(self, image, k, radius=8, return_cartesian=True):
        """Find k highest peaks in the image (CenterSpeed format)"""
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        
        radius = radius / self.pixelsize
        image = image.copy()
        opp_coordinates = np.zeros((k, 2))
        valid_peaks = 0
        
        for i in range(k):
            max_idx = np.argmax(image.reshape(-1))
            if image.flat[max_idx] < self.detection_threshold:
                break
                
            max_coords = np.unravel_index(max_idx, image.shape)
            # Ensure max_coords is a tuple with exactly 2 elements
            if len(max_coords) == 2:
                opp_coordinates[i, 0] = max_coords[0]  # y coordinate
                opp_coordinates[i, 1] = max_coords[1]  # x coordinate
            else:
                # Handle case where image might be 1D or have different shape
                self.get_logger().warning(f"Unexpected image shape: {image.shape}, max_coords: {max_coords}")
                break
                
            valid_peaks += 1
            
            if i == k - 1:
                break
                
            # Set surrounding area to zero
            top = max(0, int(max_coords[0] - radius))
            bottom = min(image.shape[0], int(max_coords[0] + radius))
            left = max(0, int(max_coords[1] - radius))
            right = min(image.shape[1], int(max_coords[1] + radius))
            image[top:bottom, left:right] = 0
        
        if valid_peaks == 0:
            return [], []
        
        if return_cartesian:
            return self.index_to_cartesian(opp_coordinates[:valid_peaks, 1], opp_coordinates[:valid_peaks, 0])
        
        return opp_coordinates[:valid_peaks, 1], opp_coordinates[:valid_peaks, 0]

    def process_scan(self):
        """Main processing function using CenterSpeed approach"""
        if self.scan_data is None or self.net is None:
            return
            
        try:
            # Initialize frames if needed
            if self.frame1 is None:
                self.frame1 = self.preprocess_scan(self.scan_data)
                return
            if self.frame2 is None:
                self.frame2 = self.preprocess_scan(self.scan_data)
                return
            
            # Update frames (like CenterSpeed)
            self.frame1 = self.frame2
            self.frame2 = self.preprocess_scan(self.scan_data)
            
            if self.frame1 is None or self.frame2 is None:
                return
            
            # Create 6-channel input (concatenate two 3-channel frames)
            preprocessed_scans = np.concatenate([self.frame1, self.frame2], axis=1)
            input_tensor = torch.FloatTensor(preprocessed_scans).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.net(input_tensor)
                
                if isinstance(outputs, tuple):
                    output_hm = outputs[0]  # Detection heatmap
                    if len(outputs) > 1:
                        output_data = outputs[1]  # Velocity and orientation data
                    else:
                        output_data = torch.zeros(1, 3, device=self.device)  # Default values
                else:
                    output_hm = outputs
                    output_data = torch.zeros(1, 3, device=self.device)
            
            # Process outputs - handle CenterSpeed model output format
            if len(output_hm.shape) == 4:  # [batch, channels, height, width]
                output_hm = output_hm.squeeze(0)  # Remove batch dimension -> [channels, height, width]
            
            # CenterSpeed outputs multiple channels: [heatmap, vx, vy, theta]
            if len(output_hm.shape) == 3 and output_hm.shape[0] >= 4:  # [channels, height, width]
                # Take the first channel as detection heatmap
                detection_hm = output_hm[0]  # Detection heatmap
                # Store velocity maps for per-object velocity extraction
                self.velocity_maps = [
                    output_hm[1],  # vx map
                    output_hm[2],  # vy map
                    output_hm[3]   # theta map
                ]
                # Extract global velocity and orientation averages
                if output_hm.shape[0] >= 4:
                    output_data = torch.stack([
                        output_hm[1].mean(),  # Average vx
                        output_hm[2].mean(),  # Average vy  
                        output_hm[3].mean()   # Average theta
                    ])
                else:
                    output_data = torch.zeros(3, device=self.device)
                output_hm = detection_hm
            elif len(output_hm.shape) == 3:  # [channels, height, width] with fewer channels
                output_hm = output_hm[0]  # Take first channel as detection heatmap
                self.velocity_maps = None
                
            # Ensure output_hm is 2D
            if len(output_hm.shape) != 2:
                self.get_logger().error(f"Unexpected output_hm shape after processing: {output_hm.shape}")
                return
            
            # Process velocity/orientation data
            output_data = output_data.squeeze(0)
            if len(output_data.shape) == 0:  # scalar
                output_data = output_data.unsqueeze(0)
            
            # Find peaks using CenterSpeed method
            x, y = self.find_k_peaks(output_hm, self.num_opponents, radius=8) # 배열 형태의 x,y 여러 객체 동시 탐지 가능
            
            # Create detected objects
            detected_objects = []
            if len(x) > 0:
                for i, (px, py) in enumerate(zip(x, y)):
                    # For each detected object, extract velocity and theta from model output at that position
                    if hasattr(self, 'velocity_maps') and self.velocity_maps is not None:
                        # If we have velocity maps from the 4-channel output
                        img_x = int((px + self.origin_offset) / self.pixelsize)
                        img_y = int((py + self.origin_offset) / self.pixelsize)
                        
                        # Clamp to valid image bounds
                        img_x = max(0, min(self.image_size - 1, img_x))
                        img_y = max(0, min(self.image_size - 1, img_y))
                        
                        vx = float(self.velocity_maps[0][img_y, img_x])
                        vy = float(self.velocity_maps[1][img_y, img_x])
                        theta = float(self.velocity_maps[2][img_y, img_x])
                    else:
                        # Use global average from output_data
                        vx = float(output_data[0]) if len(output_data) > 0 else 0.0
                        vy = float(output_data[1]) if len(output_data) > 1 else 0.0
                        theta = float(output_data[2]) if len(output_data) > 2 else 0.0
                    
                    # Create object with model predictions
                    obj = DetectedObject(px, py, vx, vy, theta)
                    detected_objects.append(obj)
            
            # Publish objects
            self.publish_objects(detected_objects)
            self.publish_objects_data(detected_objects)
            
            if len(detected_objects) > 0:
                self.get_logger().info(f'Detected {len(detected_objects)} objects: x={detected_objects[0].x:.2f}, y={detected_objects[0].y:.2f}, vx={detected_objects[0].vx:.2f}, vy={detected_objects[0].vy:.2f}, yaw={detected_objects[0].yaw:.2f}')
            
        except Exception as e:
            self.get_logger().error(f"Error in process_scan: {e}")

    def publish_objects_data(self, objects):
        """Publish object data as Float32MultiArray"""
        data_msg = Float32MultiArray()
        
        # Format: [num_objects, x1, y1, vx1, vy1, yaw1, x2, y2, vx2, vy2, yaw2, ...]
        data = [float(len(objects))]
        
        for obj in objects:
            data.extend([obj.x, obj.y, obj.vx, obj.vy, obj.yaw])
        
        data_msg.data = data
        self.objects_data_pub.publish(data_msg)
    
    def publish_objects(self, objects):
        """Publish detected objects as markers for RViz"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Create markers for each object
        for i, obj in enumerate(objects):
            # Object position marker
            marker = Marker()
            marker.header.frame_id = "laser"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "dynamic_vehicles"
            marker.id = i * 2  # Even IDs for position
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = obj.x
            marker.pose.position.y = obj.y
            marker.pose.position.z = 0.0
            
            # Orientation (from yaw)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = np.sin(obj.yaw / 2.0)
            marker.pose.orientation.w = np.cos(obj.yaw / 2.0)
            
            # Scale
            marker.scale.x = obj.size
            marker.scale.y = obj.size
            marker.scale.z = 0.5
            
            # Color (red for dynamic vehicles)
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            marker_array.markers.append(marker)
            
            # Velocity arrow marker
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "laser"
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = "velocity_arrows"
            arrow_marker.id = i * 2 + 1  # Odd IDs for velocity
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            # Arrow start point
            start_point = Point()
            start_point.x = obj.x
            start_point.y = obj.y
            start_point.z = 0.5
            
            # Arrow end point (scaled velocity)
            velocity_scale = 1.0  # Scale factor for visualization
            end_point = Point()
            end_point.x = obj.x + obj.vx * velocity_scale
            end_point.y = obj.y + -obj.vy * velocity_scale
            end_point.z = 0.5
            
            arrow_marker.points = [start_point, end_point]
            
            # Arrow appearance
            arrow_marker.scale.x = 0.1  # Shaft diameter
            arrow_marker.scale.y = 0.2  # Head diameter
            arrow_marker.scale.z = 0.2  # Head length
            
            # Color (green for velocity)
            arrow_marker.color.a = 1.0
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 1.0
            arrow_marker.color.b = 0.0
            
            marker_array.markers.append(arrow_marker)
        
        # Publish marker array
        self.objects_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DynamicVehicleDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
