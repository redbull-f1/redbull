import os
import yaml
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.client import Client
from rcl_interfaces.srv import GetParameters

from ament_index_python import get_package_share_directory
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from visualization_msgs.msg import Marker, MarkerArray
# from frenet_conversion.frenet_converter import FrenetConverter

from crazycontroller.map import MAP_Controller
# from crazycontroller.pp import PP_Controller  # TODO: implement later
from rcl_interfaces.msg import ParameterValue, ParameterType, ParameterDescriptor, FloatingPointRange
from transforms3d import euler
from .utils.global_parameter.parameter_event_handler import ParameterEventHandler

from typing import List, Dict, Union


class CrazyController(Node):
    def __init__(self):
        super().__init__('crazycontroller_manager',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.type_arr = ["not_set", "bool_value", "integer_value", "double_value", "string_value",
                         "byte_array_value", "bool_array_value", "integer_array_value",
                         "double_array_value", "string_array_value"]

        # self.map_path = self.get_parameter('map_path').value
        # self.racecar_version = self.get_parameter('racecar_version').value
        # self.sim = self.get_parameter('sim').value

        # variables
        self.rate = 40

        self.LUT_path = self.get_parameter('lookup_table_path').value  # full path to lookup table csv
        self.get_logger().info(f"Using lookup table: {self.LUT_path}")
        self.mode = self.get_parameter('mode').value

        # Publishers
        self.publish_topic = '/drive'  # simulation + real 다된대요 안되면 rethink
        # self.publish_topic = '/vesc/high_level/ackermann_cmd_mux/input/nav_1'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.publish_topic, 10)
        self.steering_pub = self.create_publisher(Marker, 'steering', 10)
        self.lookahead_pub = self.create_publisher(Marker, 'lookahead_point', 10)
        self.waypoint_pub = self.create_publisher(MarkerArray, 'my_waypoints', 10)
        self.l1_pub = self.create_publisher(Point, 'l1_distance', 10)

        # State variables
        self.track_length = None
        self.waypoint_array_in_map = None
        self.speed_now = None
        self.position_in_map = None
        self.position_in_map_frenet = None
        self.waypoint_safety_counter = 0

        # buffers for improved computation
        self.waypoint_array_buf = MarkerArray()
        self.markers_buf = [Marker() for _ in range(1000)]

        # controller choice
        if self.mode == "MAP":
            self.get_logger().info("Initializing MAP controller")
            self.init_map_controller()
        elif self.mode == "PP":
            self.get_logger().info("PP controller not implemented yet")
            # TODO: implement PP controller
            # self.init_pp_controller()
            return
        else:
            self.get_logger().error(f"Invalid mode: {self.mode}")
            return

        # Subscribers
        self.create_subscription(WpntArray, '/global_waypoints', self.track_length_cb, 10)
        self.create_subscription(WpntArray, '/local_waypoints', self.local_waypoint_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom', self.odom_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom', self.car_state_cb, 10)
        self.create_subscription(Odometry, '/car_state/frenet/odom', self.car_state_frenet_cb, 10)

        # Block until relevant data is here
        self.wait_for_messages()

        # Init converter
        # self.converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2]) 주석처리

        # 코드 병합 과정에서 주석 처리 - 0811
        # Dynamic parameters (simplified - removed trailing parameters)
        self.param_handler = ParameterEventHandler(self)
        self.callback_handle = self.param_handler.add_parameter_event_callback(
            callback=self.l1_param_cb,
        )
        param_dicts = [
            {'name': 't_clip_min',
             'default': self.l1_params["t_clip_min"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
            {'name': 't_clip_max',
             'default': self.l1_params["t_clip_max"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
            {'name': 'm_l1',
             'default': self.l1_params["m_l1"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.001)])},
            {'name': 'q_l1',
             'default': self.l1_params["q_l1"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=-1.0, to_value=1.0, step=0.001)])},
            {'name': 'speed_lookahead',
             'default': self.l1_params["speed_lookahead"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
            {'name': 'lat_err_coeff',
             'default': self.l1_params["lat_err_coeff"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
            {'name': 'acc_scaler_for_steer',
             'default': self.l1_params["acc_scaler_for_steer"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
            {'name': 'dec_scaler_for_steer',
             'default': self.l1_params["dec_scaler_for_steer"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
            {'name': 'start_scale_speed',
             'default': self.l1_params["start_scale_speed"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
            {'name': 'end_scale_speed',
             'default': self.l1_params["end_scale_speed"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
            {'name': 'downscale_factor',
             'default': self.l1_params["downscale_factor"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.5, step=0.01)])},
            {'name': 'speed_lookahead_for_steer',
             'default': self.l1_params["speed_lookahead_for_steer"],
             'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                               floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.2, step=0.01)])}
        ]
        params = self.declare_dyn_parameters(param_dicts)
        self.set_parameters(params)

        # main loop
        self.create_timer(1 / self.rate, self.control_loop)

        self.get_logger().info("CrazyController ready")

    def wait_for_messages(self):
        self.get_logger().info('CrazyController Manager waiting for messages...')
        track_length_received = False
        waypoint_array_received = False
        car_state_received = False

        while not track_length_received or not waypoint_array_received or not car_state_received:
            rclpy.spin_once(self)
            if self.track_length is not None and not track_length_received:
                self.get_logger().info('Received track length')
                track_length_received = True
            if self.waypoint_array_in_map is not None and not waypoint_array_received:
                self.get_logger().info('Received waypoint array')
                waypoint_array_received = True
            if self.speed_now is not None and self.position_in_map is not None and self.position_in_map_frenet is not None and not car_state_received:
                self.get_logger().info('Received car state messages')
                car_state_received = True

        self.get_logger().info('All required messages received. Continuing...')

    def declare_dyn_parameters(self, param_dicts):
        params = []
        for param_dict in param_dicts:
            param = self.declare_parameter(param_dict['name'], param_dict['default'], param_dict['descriptor'])
            params.append(param)
        return params

    def init_map_controller(self):
        # get l1 parameters from direct path
        l1_params_path = self.get_parameter('l1_params_path').value
        with open(l1_params_path, 'r') as f:
            self.l1_params = yaml.safe_load(f)
            self.l1_params = self.l1_params['controller']['ros__parameters']

        # IMU subscription for acceleration data (only for MAP, PP mode)
        # self.create_subscription(Imu, '/vesc/sensors/imu/raw', self.imu_cb, 10)
        self.acc_now = np.zeros(10)  # rolling buffer for acceleration

        # Initialize MAP controller with simplified parameters (removed trailing parameters)
        self.map_controller = MAP_Controller(
            self.l1_params["t_clip_min"],
            self.l1_params["t_clip_max"],
            self.l1_params["m_l1"],
            self.l1_params["q_l1"],
            self.l1_params["speed_lookahead"],
            self.l1_params["lat_err_coeff"],
            self.l1_params["acc_scaler_for_steer"],
            self.l1_params["dec_scaler_for_steer"],
            self.l1_params["start_scale_speed"],
            self.l1_params["end_scale_speed"],
            self.l1_params["downscale_factor"],
            self.l1_params["speed_lookahead_for_steer"],
            self.rate,
            self.LUT_path)

    def get_remote_parameter(self, remote_node_name, param_name):
        cli = self.create_client(GetParameters, remote_node_name + '/get_parameters')
        while not cli.wait_for_service(timeout_sec=1):
            self.get_logger().info('service not available, waiting again...')
        req = GetParameters.Request()
        req.names = [param_name]
        future = cli.call_async(req)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    res = future.result()
                    return getattr(res.values[0], self.type_arr[res.values[0].type])
                except Exception as e:
                    self.get_logger().warn('Service call failed %r' % (e,))
                break

    #############
    # CALLBACKS #
    #############
    def l1_param_cb(self, parameter_event):
        """
        Notices the change in the parameters and alters the controller params accordingly
        """
        if parameter_event.node != "/crazycontroller_manager":
            return

        if self.mode == "MAP":
            self.map_controller.t_clip_min = self.get_parameter('t_clip_min').value
            self.map_controller.t_clip_max = self.get_parameter('t_clip_max').value
            self.map_controller.m_l1 = self.get_parameter('m_l1').value
            self.map_controller.q_l1 = self.get_parameter('q_l1').value
            self.map_controller.speed_lookahead = self.get_parameter('speed_lookahead').value
            self.map_controller.lat_err_coeff = self.get_parameter('lat_err_coeff').value
            self.map_controller.acc_scaler_for_steer = self.get_parameter('acc_scaler_for_steer').value
            self.map_controller.dec_scaler_for_steer = self.get_parameter('dec_scaler_for_steer').value
            self.map_controller.start_scale_speed = self.get_parameter('start_scale_speed').value
            self.map_controller.end_scale_speed = self.get_parameter('end_scale_speed').value
            self.map_controller.downscale_factor = self.get_parameter('downscale_factor').value
            self.map_controller.speed_lookahead_for_steer = self.get_parameter('speed_lookahead_for_steer').value

        self.get_logger().info("Updated parameters")

    def track_length_cb(self, data: WpntArray):
        self.track_length = data.wpnts[-1].s_m
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.psi_rad] for wpnt in data.wpnts])

    def odom_cb(self, data: Odometry):
        self.speed_now = data.twist.twist.linear.x

    def car_state_cb(self, data: Odometry):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        rot = Rotation.from_quat([data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                                   data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        rot_euler = rot.as_euler('xyz', degrees=False)
        theta = rot_euler[2]
        self.position_in_map = np.array([x, y, theta])[np.newaxis]

    def local_waypoint_cb(self, data: WpntArray):
        self.waypoint_list_in_map = []
        for waypoint in data.wpnts:
            waypoint_in_map = [waypoint.x_m, waypoint.y_m]
            speed = waypoint.vx_mps
            if waypoint.d_right + waypoint.d_left != 0:
                self.waypoint_list_in_map.append([
                    waypoint_in_map[0],
                    waypoint_in_map[1],
                    speed,
                    min(waypoint.d_left, waypoint.d_right) / (waypoint.d_right + waypoint.d_left),
                    waypoint.s_m,
                    waypoint.kappa_radpm,
                    waypoint.psi_rad,
                    waypoint.ax_mps2
                ])
            else:
                self.waypoint_list_in_map.append([
                    waypoint_in_map[0],
                    waypoint_in_map[1],
                    speed,
                    0,
                    waypoint.s_m,
                    waypoint.kappa_radpm,
                    waypoint.psi_rad,
                    waypoint.ax_mps2
                ])
        self.waypoint_array_in_map = np.array(self.waypoint_list_in_map)
        self.waypoint_safety_counter = 0

    # def imu_cb(self, data):
    #     # save acceleration in a rolling buffer
    #     self.acc_now[1:] = self.acc_now[:-1]
    #     self.acc_now[0] = -data.linear_acceleration.y  # vesc is rotated 90 deg, so (-acc_y) == (long_acc)

    def car_state_frenet_cb(self, data: Odometry):
        s = data.pose.pose.position.x
        d = data.pose.pose.position.y
        vs = data.twist.twist.linear.x
        vd = data.twist.twist.linear.y
        self.position_in_map_frenet = np.array([s, d, vs, vd])

    def map_cycle(self):
        # Simplified call to MAP controller - removed state and opponent parameters
        speed, acceleration, jerk, steering_angle, L1_point, L1_distance, idx_nearest_waypoint = self.map_controller.main_loop(
            self.position_in_map,
            self.waypoint_array_in_map,
            self.speed_now,
            self.position_in_map_frenet,
            self.acc_now,
            self.track_length)

        # Visualization
        self.set_lookahead_marker(L1_point, 100)
        self.visualize_steering(steering_angle)
        self.l1_pub.publish(Point(x=float(idx_nearest_waypoint), y=L1_distance))

        # Safety check for waypoint timeout
        self.waypoint_safety_counter += 1
        if self.waypoint_safety_counter >= self.rate * 5:  # 5 seconds timeout
            self.get_logger().warning("[crazycontroller_manager] No fresh local waypoints. STOPPING!!")
            speed = 0
            steering_angle = 0

        return speed, steering_angle

    #############
    # MAIN LOOP #
    #############
    def control_loop(self):
        if self.mode == "MAP":
            speed, steer = self.map_cycle()
        else:
            self.get_logger().error(f"Unsupported mode: {self.mode}")
            speed = 0
            steer = 0

        # Publish drive command
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.header.frame_id = 'base_link'
        ack_msg.drive.steering_angle = steer
        ack_msg.drive.speed = speed
        self.drive_pub.publish(ack_msg)

    ############################################ VISUALIZATION ############################################
    def visualize_steering(self, theta):
        quaternions = euler.euler2quat(0, 0, theta)

        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "car_state/base_link"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 0
        lookahead_marker.id = 50
        lookahead_marker.scale.x = 0.6
        lookahead_marker.scale.y = 0.05
        lookahead_marker.scale.z = 0.01
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = 0.0
        lookahead_marker.pose.position.y = 0.0
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.pose.orientation.x = quaternions[0]
        lookahead_marker.pose.orientation.y = quaternions[1]
        lookahead_marker.pose.orientation.z = quaternions[2]
        lookahead_marker.pose.orientation.w = quaternions[3]
        self.steering_pub.publish(lookahead_marker)

    def set_waypoint_markers(self, waypoints):
        wpnt_id = 0

        for waypoint in waypoints:
            waypoint_marker = self.markers_buf[wpnt_id]
            waypoint_marker.header.frame_id = "map"
            waypoint_marker.header.stamp = self.get_clock().now().to_msg()
            waypoint_marker.type = 2
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            waypoint_marker.color.r = 0.0
            waypoint_marker.color.g = 0.0
            waypoint_marker.color.b = 1.0
            waypoint_marker.color.a = 1.0
            waypoint_marker.pose.position.x = waypoint[0]
            waypoint_marker.pose.position.y = waypoint[1]
            waypoint_marker.pose.position.z = 0.0
            waypoint_marker.pose.orientation.x = 0.0
            waypoint_marker.pose.orientation.y = 0.0
            waypoint_marker.pose.orientation.z = 0.0
            waypoint_marker.pose.orientation.w = 1.0
            waypoint_marker.id = wpnt_id + 1
            wpnt_id += 1
        self.waypoint_array_buf.markers = self.markers_buf[:wpnt_id]
        self.waypoint_pub.publish(self.waypoint_array_buf)

    def set_lookahead_marker(self, lookahead_point, id):
        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "map"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 2
        lookahead_marker.id = id
        lookahead_marker.scale.x = 0.15
        lookahead_marker.scale.y = 0.15
        lookahead_marker.scale.z = 0.15
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = lookahead_point[0]
        lookahead_marker.pose.position.y = lookahead_point[1]
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.pose.orientation.x = 0.0
        lookahead_marker.pose.orientation.y = 0.0
        lookahead_marker.pose.orientation.z = 0.0
        lookahead_marker.pose.orientation.w = 1.0
        self.lookahead_pub.publish(lookahead_marker)


def main():
    rclpy.init()
    node = CrazyController()
    rclpy.spin(node)
    rclpy.shutdown()
