#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import cv2
import os
import yaml
import subprocess
from ament_index_python.packages import get_package_share_directory

class MappingSaverNode(Node):
    def __init__(self):
        super().__init__('mapping_saver_node')
        
        # Parameters
        self.declare_parameter('map_name', 'default_map')
        self.map_name = self.get_parameter('map_name').get_parameter_value().string_value
        
        # Create map directory
        stack_master_share = get_package_share_directory('stack_master')
        self.map_dir = os.path.join(stack_master_share, 'maps', self.map_name)
        
        # State variables
        self.map_data = None
        self.pose_data = None
        self.map_resolution = 0.05
        self.map_origin = None
        self.gui_initialized = False
        
        # Subscribers
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/car_state/pose',
            self.pose_callback,
            10
        )
        
        self.get_logger().info(f'Mapping Saver Node started. Map will be saved as: {self.map_name}')
        self.get_logger().info('Waiting for map and pose data...')

    def map_callback(self, msg):
        self.map_data = msg
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        
        if not self.gui_initialized and self.pose_data is not None:
            self.initialize_gui()
            self.gui_initialized = True
        
        if self.gui_initialized:
            self.update_map_display()

    def pose_callback(self, msg):
        self.pose_data = msg

    def initialize_gui(self):
        """Initialize the matplotlib GUI"""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.axfinish) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})
        self.fig.suptitle(f'Mapping Progress - {self.map_name}')
        self.btn_save = Button(self.axfinish, 'Save Map')
        self.btn_save.on_clicked(self.save_map_callback)
        
        self.get_logger().info('GUI initialized. Click "Save Map" button when mapping is complete.')

    def update_map_display(self):
        """Update the map display in real-time"""
        if self.map_data is None:
            return

        # Convert ROS occupancy grid to numpy array
        width = self.map_data.info.width
        height = self.map_data.info.height
        map_array = np.array(self.map_data.data).reshape((height, width))
        
        # Process the map (similar to global planner filtering)
        processed_map = self.filter_map(map_array)
        
        # Update the plot
        self.ax1.clear()
        self.ax1.imshow(processed_map, cmap='gray', origin='lower')
        self.ax1.set_title('Current Map')

        # # +) 실제 좌표로 축 레이블 변경
        # extent = [
        #     self.map_origin.x,  # x_min
        #     self.map_origin.x + width * self.map_resolution,  # x_max
        #     self.map_origin.y,  # y_min  
        #     self.map_origin.y + height * self.map_resolution  # y_max
        # ]
        # self.ax1.imshow(processed_map, cmap='gray', origin='lower', extent=extent)
        # self.ax1.set_xlabel('X (meters)')
        # self.ax1.set_ylabel('Y (meters)')

        # # +) 차량 현재 위치 표시 추가
        # if self.pose_data:
        #     vehicle_x = self.pose_data.pose.position.x
        #     vehicle_y = self.pose_data.pose.position.y
        #     self.ax1.plot(vehicle_x, vehicle_y, 'ro', markersize=10, label='Vehicle')

        plt.pause(0.1)  # Brief pause to update display

    def filter_map(self, original_map):
        """Filter the occupancy grid map"""
        # Convert unknown (-1) to occupied (100)
        original_map = np.where(original_map == -1, 100, original_map)
        
        # Binarize map
        occupancy_threshold = 50  # You can make this a parameter
        bw = np.where(original_map < occupancy_threshold, 255, 0)
        bw = np.uint8(bw)
        
        # Morphological filtering
        kernel_size = 3  # You can make this a parameter
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        filtered_map = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return filtered_map

    def save_map_callback(self, event):
        """Callback when save button is clicked"""
        if self.map_data is None:
            self.get_logger().warn('No map data available to save!')
            return
            
        self.get_logger().info('Saving map...')
        
        try:
            # Create map directory
            os.makedirs(self.map_dir, exist_ok=True)
            
            # Get filtered map
            width = self.map_data.info.width
            height = self.map_data.info.height
            map_array = np.array(self.map_data.data).reshape((height, width))
            filtered_map = self.filter_map(map_array)
            
            # Save PNG file
            img_path = os.path.join(self.map_dir, self.map_name + '.png')
            flipped_map = cv2.flip(filtered_map, 0)  # Flip to match ROS convention
            cv2.imwrite(img_path, flipped_map)
            
            # Save YAML file
            yaml_data = {
                'image': self.map_name + '.png',
                'resolution': self.map_resolution,
                'origin': [self.map_origin.x, self.map_origin.y, 0],
                'negate': 0,
                'occupied_thresh': 0.65,
                'free_thresh': 0.196
            }
            
            yaml_path = os.path.join(self.map_dir, self.map_name + '.yaml')
            with open(yaml_path, 'w') as file:
                yaml.dump(yaml_data, file, default_flow_style=False)
            
            # Save pbstream using finish_map.sh
            workspace_path = os.path.expanduser('~/f1tenth_ws')
            pbstream_path = os.path.join(self.map_dir, self.map_name + '.pbstream')
            finish_script_path = os.path.join(
                workspace_path, 'src', 'race_stack', 'stack_master', 'scripts', 'finish_map.sh'
            )
            
            # Make script executable and run it
            subprocess.run(['chmod', '+x', finish_script_path], check=True)
            result = subprocess.run([finish_script_path, pbstream_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.get_logger().info(f'Map successfully saved to {self.map_dir}')
                self.get_logger().info('Files created: .png, .yaml, and .pbstream')
                plt.close('all')  # Close the GUI
            else:
                self.get_logger().error(f'Error saving pbstream: {result.stderr}')
                
        except Exception as e:
            self.get_logger().error(f'Error saving map: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    mapping_saver = MappingSaverNode()
    
    try:
        rclpy.spin(mapping_saver)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')  # Ensure GUI is closed
        mapping_saver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
