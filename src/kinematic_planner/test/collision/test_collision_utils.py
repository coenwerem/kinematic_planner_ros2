from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from robot_3r_interfaces.msg import SceneObstacles
from kinematic_planner.collision.collision_utils import obstacle_to_fclobj


def _pose_at_origin():
    pose = PoseStamped()
    pose.pose.orientation.w = 1.0
    return pose


def test_sphere_obstacle_does_not_raise_attribute_error():
    obstacles = SceneObstacles()
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [0.5]
    obstacles.scene_obstacles = [sphere]
    obstacles.obstacle_poses = [_pose_at_origin()]
    obstacles.obstacle_ids = [1]

    fcl_objs = obstacle_to_fclobj(obstacles)
    assert len(fcl_objs) == 1


def test_cylinder_obstacle_does_not_raise_attribute_error():
    obstacles = SceneObstacles()
    cylinder = SolidPrimitive()
    cylinder.type = SolidPrimitive.CYLINDER
    cylinder.dimensions = [0.3, 1.2]  # [radius, height], per SolidPrimitive.msg field order
    obstacles.scene_obstacles = [cylinder]
    obstacles.obstacle_poses = [_pose_at_origin()]
    obstacles.obstacle_ids = [1]

    fcl_objs = obstacle_to_fclobj(obstacles)
    assert len(fcl_objs) == 1
