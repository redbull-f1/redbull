from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'crazycontroller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ForzaETH',
    maintainer_email='nicolas.baumann@pbl.ee.ethz.ch',
    description='Simplified controller for time trial racing',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crazycontroller = crazycontroller.controller_manager:main',
        ],
    },
)
