import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.object_scan_pub = self.create_publisher(LaserScan, '/object_scan', 10)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        # 1m 초과하는 값은 inf로 처리
        filtered_ranges = np.where(ranges <= 1.0, ranges, float('inf'))

        object_scan = LaserScan()
        object_scan.header.frame_id = 'laser'
        object_scan.header.stamp = self.get_clock().now().to_msg()
        object_scan.angle_min = msg.angle_min
        object_scan.angle_max = msg.angle_max
        object_scan.angle_increment = msg.angle_increment
        object_scan.time_increment = msg.time_increment
        object_scan.scan_time = msg.scan_time
        object_scan.range_min = msg.range_min
        object_scan.range_max = msg.range_max
        object_scan.ranges = filtered_ranges.tolist()
        object_scan.intensities = [1.0]*len(filtered_ranges)
        self.object_scan_pub.publish(object_scan)

def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
