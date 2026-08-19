from setuptools import find_packages, setup
from glob import glob

package_name = 'kinematic_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Clinton Enwerem',
    maintainer_email='me@clintonenwerem.com',
    description='Standalone collision-free kinematic planner (RRT*, Informed RRT*).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = kinematic_planner.scripts.planner_node:main',
            'informed_rrt_star_node = kinematic_planner.scripts.informed_rrt_star_node:main',
            'obstacle_publisher = kinematic_planner.scripts.obstacle_publisher:main',
            'robot_geom_publisher = kinematic_planner.scripts.robot_geom_publisher:main',
        ],
    },
)
