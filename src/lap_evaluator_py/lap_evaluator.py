#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
import numpy as np
import time


class LapEvaluator(Node):
    def __init__(self):
        super().__init__('lap_evaluator')
        # Frenet odom 구독
        self.subscription = self.create_subscription(
            Odometry,
            '/car_state/frenet/odom',
            self.odom_callback,
            10)
        # ego_racecar odom 구독 (차량 기준 속도)
        self.ego_odom_subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.ego_odom_callback,
            10)
        # /drive 토픽 구독
        self.drive_subscription = self.create_subscription(
            AckermannDriveStamped,
            '/drive',
            self.drive_callback,
            10)
        # Marker 퍼블리셔
        self.text_marker_pub = self.create_publisher(Marker, '/lap_evaluator/text_marker', 1)

        self.prev_s = None
        self.lap_start_time = None
        self.lap_count = 0
        self.d_errors = []
        self.lap_times = []
        self.latest_d = 0.0
        self.latest_ego_speed = 0.0
        self.latest_drive_speed = 0.0
        self.latest_drive_steer = 0.0
        self.get_logger().info('LapEvaluator node started.')


    def odom_callback(self, msg):
        s = msg.pose.pose.position.x
        d = msg.pose.pose.position.y
        vs = msg.twist.twist.linear.x
        now = self.get_clock().now().nanoseconds * 1e-9

        self.latest_d = d
        # self.latest_vs = vs  # 더 이상 사용하지 않음

        # Lap start
        if self.prev_s is None:
            self.prev_s = s
            self.lap_start_time = now
            self.d_errors = [d]
            self.publish_text_marker()
            return

        # Detect lap completion (s wraps around)
        if self.prev_s > 5.0 and s < 2.0:  # Thresholds may need tuning
            self.lap_count += 1
            lap_time = now - self.lap_start_time
            self.lap_times.append(lap_time)
            max_d = np.max(np.abs(self.d_errors))
            rmse_d = np.sqrt(np.mean(np.square(self.d_errors)))
            self.get_logger().info(f"Lap {self.lap_count}: Time = {lap_time:.2f}s, Max d = {max_d:.3f}, RMSE d = {rmse_d:.3f}")
            # Reset for next lap
            self.lap_start_time = now
            self.d_errors = [d]
        else:
            self.d_errors.append(d)
        self.prev_s = s
        self.publish_text_marker()


    def ego_odom_callback(self, msg):
        # 차량 기준 x축 속도
        self.latest_ego_speed = msg.twist.twist.linear.x
        self.publish_text_marker()

    def drive_callback(self, msg):
        self.latest_drive_speed = msg.drive.speed
        self.latest_drive_steer = msg.drive.steering_angle
        self.publish_text_marker()

    def publish_text_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lap_evaluator"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        # icra_2.yaml origin: [-50.0, -2.3, 0.0]
        marker.pose.position.x = -45.5  # 왼쪽에서 약간 오른쪽
        marker.pose.position.y = 13.0  # 위쪽에서 약간 아래
        marker.pose.position.z = 1.5    # 띄우는 높이
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.40  # 더 작고 보기 좋은 텍스트 크기
        # 색상: 밝은 노랑
        marker.color.r = 1.0
        marker.color.g = 0.95
        marker.color.b = 0.2
        marker.color.a = 0.95
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = (
            f"Lap: {self.lap_count}\n"
            f"Tracking error (d): {self.latest_d:.3f} m\n"
            f"Ego speed: {self.latest_ego_speed:.3f} m/s\n"
            f"/drive speed: {self.latest_drive_speed:.3f} m/s\n"
            f"/drive steering: {self.latest_drive_steer:.3f} rad"
        )
        self.text_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = LapEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
