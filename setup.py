import os
from glob import glob
from setuptools import setup

package_name = 'autonomous_bot_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='radid',
    maintainer_email='radid@example.com',
    description='Autonomous sonar obstacle-avoidance bot (Gazebo + RViz)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'obstacle_randomizer = autonomous_bot_sim.obstacle_randomizer:main',
            'autonomous_controller = autonomous_bot_sim.autonomous_controller:main',
            'start_button = autonomous_bot_sim.start_button:main',
        ],
    },
)
