from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'redbull'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('src/models/*.py')),
        (os.path.join('share', package_name, 'trained_models'), glob('src/trained_models/*')),
        (os.path.join('lib', package_name), ['car_tracker.py', 'dynamic_vehicle_detector_ros2.py']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scikit-learn',
        'torch',
        'torchvision'
    ],
    zip_safe=True,
    maintainer='harry',
    maintainer_email='your_email@example.com',
    description='LiDAR preprocessing package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'car_tracker = car_tracker:main',
            'dynamic_vehicle_detector = dynamic_vehicle_detector_ros2:main'
        ],
    },
)

