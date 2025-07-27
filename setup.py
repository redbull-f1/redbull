from setuptools import setup

package_name = 'redbull'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=['car_tracker'],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scikit-learn'
    ],
    zip_safe=True,
    maintainer='harry',
    maintainer_email='your_email@example.com',
    description='LiDAR preprocessing package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'car_tracker = car_tracker:main'
        ],
    },
)

