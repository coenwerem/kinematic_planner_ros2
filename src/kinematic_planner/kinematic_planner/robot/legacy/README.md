# Legacy URDF Parser

`urdf_parser.py` is a standalone URDF parser with custom forward kinematics,
Jacobian, and inverse kinematics code, written as course material for
ENEE467, Robotics Projects Laboratory, University of Maryland.

The runtime planning path in `kinematic_planner` (`planner_node.py`,
`informed_rrt_star_node.py`, `robot_config.py`, `robot_collision_model.py`)
does not import `urdf_parser.py`. Forward kinematics in the planning path comes
from Robotics Toolbox `ERobot`. Collision geometry construction comes from
`kinematic_planner.collision.robot_collision_model`.

Kept here as educational reference material, with a working box/sphere/
cylinder path and a documented, disabled mesh-loading fallback. Not part of
the supported planning pipeline, and not covered by the package's test
suite.
