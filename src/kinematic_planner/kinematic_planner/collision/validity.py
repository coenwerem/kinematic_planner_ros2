"""URDF-driven state validity checking, independent of ROS.

`build_collision_fn` returns a predicate reporting whether a configuration is
collision-free, given a robot model parsed from URDF and a set of obstacles. The
predicate performs forward kinematics, robot-obstacle queries, and self-collision
queries through FCL.

Nothing in this module imports rclpy. Any consumer needing collision checking
without a running ROS graph, such as a learning pipeline filtering sampled
configurations or validating a dataset, can import it directly:

    from kinematic_planner.collision.validity import build_collision_fn, build_rtb_model

Mesh geometry referenced by `package://` URIs still resolves through the ament
index, so plain-Python use outside a sourced workspace requires either absolute
mesh paths in the URDF or primitive collision geometry.
"""

import fcl
import numpy as np
import spatialmath as sm

from kinematic_planner.collision.collision_utils import obstacle_to_fclobj
from kinematic_planner.collision.robot_collision_model import link_shapes_to_fcl_objects
from kinematic_planner.collision.self_collision import check_self_collision
from kinematic_planner.planning.tree import TreeNode

try:
    from roboticstoolbox.robot.ERobot import ERobot
    _RTB_AVAILABLE = True
except ImportError:
    _RTB_AVAILABLE = False


def build_rtb_model(urdf_path: str):
    """Build a Robotics Toolbox ERobot from a URDF file path."""
    class _RobotModel(ERobot):
        def __init__(self, path):
            links, name, urdf_string, urdf_fp = super().URDF_read(path)
            super().__init__(links, name=name.upper(), manufacturer="Custom",
                             urdf_string=urdf_string, urdf_filepath=urdf_fp)
    return _RobotModel(urdf_path)


def build_collision_fn(robot_config, link_shapes, obstacle_geom, rtb_model,
                        collision_checker: str, min_obs_dist: float,
                        check_collision: bool, get_logger=lambda: None):
    """Build a collision_fn closure for kinematic_planner.planning.

    check_collision=False returns a closure reporting collision-free
    unconditionally, before any FCL/link_shapes code runs. RRTPlannerBase
    (kinematic_planner.planning.tree) never reads check_collision itself;
    build_collision_fn is the single call site consuming check_collision.

    link_shapes (kinematic_planner.collision.robot_collision_model output)
    replaces the RigidBodyGeom/SolidPrimitive message path: every
    <collision> element per link is checked, each carrying its own local
    origin transform, instead of only the first element at the raw link
    frame. Self-collision runs alongside robot-obstacle checking through
    the same per-waypoint FK, using RobotConfig.get_collision_pairs() for
    the pairs to check.

    obstacle_geom is fixed for the lifetime of the returned closure (a
    fresh collision_fn is built per planning attempt), so its FCL objects
    are converted once here rather than on every candidate check -- with
    thousands of checks per plan, re-parsing the same static obstacle
    geometry every call was the dominant collision_fn cost.
    """
    if not check_collision:
        return lambda _node: True

    urdf_link_names = set(link_shapes.keys())
    rtb_link_names = {link.name for link in rtb_model.links}
    link_names = list(urdf_link_names & rtb_link_names)
    unreachable_links = urdf_link_names - rtb_link_names
    if unreachable_links:
        logger = get_logger()
        if logger is not None:
            logger.error(
                "The following URDF collision links are not reachable from the "
                f"Robotics Toolbox model and are skipped in collision checking: "
                f"{sorted(unreachable_links)}"
            )
    self_collision_pairs = robot_config.get_collision_pairs()

    try:
        obs_fcl_objects = list(obstacle_to_fclobj(obstacles=obstacle_geom))
    except Exception as e:
        logger = get_logger()
        if logger is not None:
            logger.error(f"Error converting obstacle geometry: {e}")
        return lambda _node: False

    def collision_fn(candidate_node: TreeNode) -> bool:
        for q in candidate_node.path_q:
            rtb_model.q = q
            link_fcl_objects = {}
            try:
                for link_name in link_names:
                    T_fk = rtb_model.fkine(q, end=link_name, include_base=True).A
                    link_fcl_objects[link_name] = link_shapes_to_fcl_objects(
                        link_shapes[link_name], T_fk,
                    )
            except Exception as e:
                logger = get_logger()
                if logger is not None:
                    logger.error(f"Error computing FK/geometry for collision check: {e}")
                return False

            for link_name, robot_objs in link_fcl_objects.items():
                if link_name == robot_config.base_link_name:
                    # The base link's pose is fixed relative to obstacles regardless
                    # of joint configuration, so a proximity violation here can never
                    # be resolved by planning; only self-collision applies to it.
                    continue
                for rob_obj in robot_objs:
                    for obs_obj in obs_fcl_objects:
                        try:
                            if collision_checker == "bvol":
                                creq = fcl.CollisionRequest()
                                creq.enable_contact = True
                                cres = fcl.CollisionResult()
                                ret = fcl.collide(rob_obj, obs_obj, creq, cres)
                                if cres.is_collision or ret > 0:
                                    return False
                            elif collision_checker == "proximity":
                                dreq = fcl.DistanceRequest(enable_signed_distance=True)
                                dres = fcl.DistanceResult()
                                fcl.distance(rob_obj, obs_obj, dreq, dres)
                                if dres.min_distance < min_obs_dist:
                                    return False
                        except Exception as e:
                            logger = get_logger()
                            if logger is not None:
                                logger.error(f"Error checking {link_name} against an obstacle: {e}")
                            return False

            try:
                if check_self_collision(link_fcl_objects, self_collision_pairs):
                    return False
            except Exception as e:
                logger = get_logger()
                if logger is not None:
                    logger.error(f"Error checking self-collision: {e}")
                return False

        return True

    return collision_fn


# Retained under the original private name so existing imports continue to work.
_build_rtb_model = build_rtb_model
