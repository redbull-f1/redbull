from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'redbull'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    py_modules=['car_tracker'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files if they exist
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py') if os.path.exists('launch') else []),
        # Include config files if they exist
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') if os.path.exists('config') else []),
        # Include model files if they exist
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
            'car_tracker = car_tracker:main',
            'dynamic_vehicle_detector = detector.dynamic_vehicle_detector_ros2:main',
            'parse_bag_csv = parse_bag_csv:main',
            'parse_bag_ros2 = parse_bag_ros2:main',
        ],
    },
)

