from setuptools import setup
package_name = 'redbull'
setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, f'{package_name}.detector', f'{package_name}.src', f'{package_name}.src.models'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='harry',
    maintainer_email='your_email@example.com',
    description='Dynamic vehicle detection using TinyCenterSpeed and LiDAR preprocessing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sim_detector = redbull.detector.sim_detector_ObstacleArray:main',
            'real_detector = redbull.detector.real_detector_ObstacleArray:main',
            'dyn_detector  = redbull.detector.dynamic_vehicle_detector_ros2:main',
        ],
    },
)
