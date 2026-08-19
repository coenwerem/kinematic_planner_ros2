import rclpy
import pytest
from shape_msgs.msg import SolidPrimitive

from kinematic_planner.scripts.obstacle_publisher import ObstaclePublisher


@pytest.fixture(autouse=True)
def _rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_publisher(obstacle_type, obstacle_sizes, num_obstacles, obstacle_positions):
    return ObstaclePublisher(
        parameter_overrides=[
            rclpy.parameter.Parameter("obstacle_type", value=obstacle_type),
            rclpy.parameter.Parameter("num_obstacles", value=num_obstacles),
            rclpy.parameter.Parameter("obstacle_sizes", value=obstacle_sizes),
            rclpy.parameter.Parameter("obstacle_positions", value=obstacle_positions),
        ]
    )


def test_sphere_obstacle_type_publishes_sphere_primitives():
    node = _make_publisher("sphere", [0.2], 1, [0.0, 0.5, 0.5])
    msg = node.build_obstacles_message()
    assert len(msg.scene_obstacles) == 1
    assert msg.scene_obstacles[0].type == SolidPrimitive.SPHERE
    assert msg.scene_obstacles[0].dimensions[SolidPrimitive.SPHERE_RADIUS] == 0.2
    node.destroy_node()


def test_cylinder_obstacle_type_publishes_cylinder_primitives():
    node = _make_publisher("cylinder", [0.3, 0.05], 1, [0.0, 0.5, 0.5])
    msg = node.build_obstacles_message()
    assert len(msg.scene_obstacles) == 1
    assert msg.scene_obstacles[0].type == SolidPrimitive.CYLINDER
    assert msg.scene_obstacles[0].dimensions[SolidPrimitive.CYLINDER_HEIGHT] == 0.3
    assert msg.scene_obstacles[0].dimensions[SolidPrimitive.CYLINDER_RADIUS] == 0.05
    node.destroy_node()


def test_sphere_obstacle_type_does_not_crash_marker_publishing():
    node = _make_publisher("sphere", [0.2], 1, [0.0, 0.5, 0.5])
    node.build_obstacles_message()
    node.publish_obstacle_markers()
    node.destroy_node()
