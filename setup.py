from setuptools import setup

package_name = 'redbull'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='harry',
    maintainer_email='your_email@example.com',
    description='LiDAR preprocessing package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lidar_preprocessing = redbull.lidar_preprocessing:main'
        ],
    },
)

