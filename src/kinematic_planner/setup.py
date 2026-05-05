from setuptools import find_packages, setup

package_name = 'kinematic_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Clinton Enwerem',
    maintainer_email='robotdevx@gmail.com',
    description='Standalone collision-free kinematic planner (RRT*, Informed RRT*). No MoveIt.',
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
