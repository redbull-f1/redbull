#!/usr/bin/env python3
# -- coding: utf-8 --
#
# Copyright (c) 2013 PAL Robotics SL.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
# ...

import os
import pygame
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header

class VelocityRamp:
    """
    목표값(target)에 accel만큼씩 부드럽게 접근시키는 램프
    """
    def __init__(self, accel: float, hz: float):
        self.accel = accel
        self.dt    = 1.0 / hz

    def step(self, current: float, target: float) -> float:
        if current < target:
            return min(current + self.accel * self.dt, target)
        elif current > target:
            return max(current - self.accel * self.dt, target)
        return current

class PygameTeleop(Node):
    def __init__(self):
        super().__init__('pygame_key_teleop')

        # Publisher for AckermannDriveStamped on 'key_vel'
        self._pub = self.create_publisher(
            AckermannDriveStamped,
            'key_vel',
            qos_profile_system_default
        )

        # Parameters (launch 파일이나 CLI 로 override 가능)
        self._hz             = self.declare_parameter('hz',                30.0).value # 30.0으로 변경 (주기)
        self._max_forward    = self.declare_parameter('forward_rate',      0.9).value  # 0.8에서 5.0으로 변경 (전진 최대 속도)
        self._max_backward   = self.declare_parameter('backward_rate',     0.9).value # 0.5에서 2.0으로 변경 (후진 최대 속도)
        self._max_steer      = self.declare_parameter('rotation_rate',     0.36).value  # 1.0에서 1.5로 변경 (회전 최대 속도)
        self._lin_accel      = self.declare_parameter('linear_acceleration',  1.0).value # 1.0에서 0.5로 변경 (선형 가속도)
        self._ang_accel      = self.declare_parameter('angular_acceleration', 1.0).value # 원래 2.0

        # 상태 변수
        self._current_speed  = 0.0
        self._current_steer  = 0.0
        self._target_speed   = 0.0
        self._target_steer   = 0.0

        # 램프 객체
        self._speed_ramp = VelocityRamp(self._lin_accel, self._hz)
        self._steer_ramp = VelocityRamp(self._ang_accel, self._hz)

    def publish_cmd(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'key_teleop'
        msg.drive.speed          = self._current_speed
        msg.drive.steering_angle = self._current_steer
        self._pub.publish(msg)

def main():
    # ROS2 초기화
    rclpy.init()
    node = PygameTeleop()

    # pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Ackermann Teleop")
    font  = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    hz = node._hz

    running = True
    while running and rclpy.ok():
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 키 상태 읽어서 목표값 설정
        keys = pygame.key.get_pressed()

        # 전·후진
        if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            node._target_speed =  node._max_forward
        elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
            node._target_speed = -node._max_backward
        else:
            node._target_speed = 0.0

        # 좌·우 조향
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            node._target_steer =  node._max_steer
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            node._target_steer = -node._max_steer
        else:
            node._target_steer = 0.0

        # 램프 적용: 부드러운 가속/제동 및 조향
        node._current_speed = node._speed_ramp.step(
            node._current_speed, node._target_speed)
        node._current_steer = node._steer_ramp.step(
            node._current_steer, node._target_steer)

        # ROS 발행
        node.publish_cmd()

        # 화면 렌더링
        screen.fill((30, 30, 30))
        txt1 = font.render(f"Speed: {node._current_speed:.2f} m/s", True, (255, 255, 255))
        txt2 = font.render(f"Steer: {node._current_steer:.2f} rad", True, (255, 255, 255))
        screen.blit(txt1, (20, 60))
        screen.blit(txt2, (20, 100))
        pygame.display.flip()

        # 루프 타이밍
        clock.tick(hz)

    # 종료 처리
    pygame.quit()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
