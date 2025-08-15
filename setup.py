from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'redbull'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(include=['redbull', 'redbull.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py') if os.path.exists('launch') else []),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') if os.path.exists('config') else []),
        (os.path.join('share', package_name, 'trained_models'), glob('train/trained_models/*.pt') if os.path.exists('train/trained_models') else []),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.21.0',
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
    ],
    zip_safe=True,
    maintainer='harry',
    maintainer_email='your_email@example.com',
    description='Dynamic vehicle detection using TinyCenterSpeed and LiDAR preprocessing package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 반드시 redbull. 접두사 + 모듈에 main() 있어야 함
            'car_tracker = redbull.detector.car_tracker:main',
            'dynamic_vehicle_detector = redbull.detector.dynamic_vehicle_detector_ros2:main',
            'parse_bag_csv = redbull.src.parse_bag_csv:main',
            'parse_bag_ros2 = redbull.src.parse_bag_ros2:main',
            'sim_detector_obstacles = redbull.detector.sim_detector_ObstacleArray:main',
        ],
    },
)
