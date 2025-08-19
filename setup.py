from setuptools import find_packages, setup

package_name = 'lap_evaluator_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'numpy', 'nav_msgs', 'ackermann_msgs', 'visualization_msgs'],
    zip_safe=True,
    maintainer='lmw',
    maintainer_email='minwon0314@hanyang.ac.kr',
    description='Lap evaluator node for F1TENTH',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lap_evaluator = lap_evaluator_py.lap_evaluator:main',
        ],
    },
)
