# kinematic_planner_ros2

  ┌───────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
  │               File                │                                       What changed                                        │  
  ├───────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ robot/robot_config.py             │ New. Parses joint names, limits, link names from raw URDF using stdlib xml.etree.        │
  │                                   │ Replaces all pymoveit2.robots.robot_3r + ament_index + SRDF calls.    │
  ├───────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤  
  │ robot/urdf_parser.py              │ Copied verbatim — it was already pure Python (custom FK/IK/Jacobian).    │
  ├───────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤  
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ robot/robot_config.py             │ New. Parses joint names, limits, link names from raw URDF using stdlib xml.etree. Replaces all pymoveit2.robots.robot_3r + ament_index + SRDF calls.           │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ robot/urdf_parser.py              │ Copied verbatim — it was already pure Python (custom FK/IK/Jacobian).                                                                         │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ collision/collision_utils.py      │ Adapted — se3_to_pose_stamped() moved here so no import from pymoveit2.                                                                         │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/planner_node.py           │ Adapted from sampling_based_planner.py — all 9 module-level robot globals replaced byRobotConfig.from_urdf() in __init__. RTB FK retained (not a MoveIt dep). │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/informed_rrt_star_node.py │ Adapted from informed_rrts_planner.py — same decoupling.                                                                         │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/obstacle_publisher.py     │ Adapted — platform_height, ring_radius, base_link_name, world_frame are all ROS2 parameters now (no 3R assumptions).                                           │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/robot_geom_publisher.py   │ Adapted — reads everything from robot_description parameter.
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ collision/collision_utils.py      │ Adapted — se3_to_pose_stamped() moved here so no import from pymoveit2.            │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/planner_node.py           │ Adapted from sampling_based_planner.py — all 9 module-level robot globals replaced │
  │                                   │  by RobotConfig.from_urdf() in __init__. RTB FK retained (not a MoveIt dep).       │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/informed_rrt_star_node.py │ Adapted from informed_rrts_planner.py — same decoupling.                           │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/obstacle_publisher.py     │ Adapted — platform_height, ring_radius, base_link_name, world_frame are all ROS2   │
  │                                   │ parameters now (no 3R assumptions).                                                │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/robot_geom_publisher.py   │ Adapted — reads everything from robot_description parameter.                       │
  ├───────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ launch/planner.launch.py          │ New — loads xacro URDF and starts all three nodes. No MoveIt package path needed.  │
  └───────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

# INstallation
  To build and run:
  cd /home/drce/OSS_projects/kinematic_planner_ros2
  colcon build --packages-select robot_3r_interfaces robot_3r_description kinematic_planner
  source install/setup.bash
  ros2 launch launch/planner.launch.py goal_config:="[1.5093, 0.6072, 1.4052]"