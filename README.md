# kinematic_planner_ros2

A suite of ROS 2 packages implementing standalone collision-free **RRT\*** and **Informed RRT\*** path planning for robot manipulators, built from the ground-up to serve as an educational complement of the more feature-rich and standard motion planning framework, [MoveIt](https://github.com/moveit/moveit2).  Collision checking uses the
[Flexible Collision Library (FCL)](https://github.com/humanoid-path-planner/hpp-fcl) directly.
Forward kinematics use the [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python), and the robot is described entirely by a URDF.

The included example robot is a 3-DOF serial manipulator (3R arm).
The planner supports any N-DOF robot whose links use **convex collision primitives**
(box, cylinder, sphere) in their URDF `<collision>` elements.

---

## Components

| Capability | Implementation |
|---|---|
| Sampling-based planning | Custom RRT\* and Informed RRT\* in `scripts/planner_node.py` and `scripts/informed_rrt_star_node.py` |
| Collision checking | FCL bounding-volume and signed-distance modes in `collision/collision_utils.py` |
| Forward kinematics | Robotics Toolbox `ERobot` loaded from URDF |
| Generic robot interface | `robot/robot_config.py` — parses joint limits, link names, and kinematic topology from a raw URDF string using stdlib `xml.etree`; no external config files needed |
| Robot-agnostic obstacle scene | `scripts/obstacle_publisher.py` — all geometry parameters are ROS 2 parameters, nothing hardcoded |

---

## Workspace Layout

```
kinematic_planner_ros2/
├── launch/
│   └── planner.launch.py            # single entry point; wires all three nodes
└── src/
    ├── kinematic_planner/           # main package — planner, collision, robot model
    │   └── kinematic_planner/
    │       ├── robot/
    │       │   ├── robot_config.py  # RobotConfig dataclass; from_urdf() classmethod
    │       │   └── legacy/urdf_parser.py  # educational FK/Jacobian/IK reference, not used at runtime
    │       ├── collision/
    │       │   └── collision_utils.py  # FCL helpers
    │       └── scripts/
    │           ├── planner_node.py          # RRT* ROS 2 node
    │           ├── informed_rrt_star_node.py# Informed RRT* ROS 2 node
    │           ├── obstacle_publisher.py    # publishes scene obstacles
    │           └── robot_geom_publisher.py  # publishes robot link geometry from URDF
    ├── robot_3r_description/        # URDF/xacro for the example 3R robot
    └── kinematic_planner_interfaces/ # custom ROS 2 message definitions
```

---

## Prerequisites

### System

- **Ubuntu 22.04** (Jammy) — tested environment
- **ROS 2 Humble** (desktop install recommended)
- **xacro** — for processing the robot description:
  ```bash
  sudo apt install ros-humble-xacro
  ```

### Python packages

Install into your ROS 2 Python environment:

```bash
pip install \
  roboticstoolbox-python \
  spatialmath-python \
  transforms3d \
  trimesh \
  numpy
```

Install the FCL Python bindings:

```bash
pip install python-fcl
```

> **Note on `python-fcl`:** if `pip install python-fcl` fails, install the system
> FCL library first:
> ```bash
> sudo apt install libfcl-dev
> pip install python-fcl
> ```

---

## Build

```bash
cd kinematic_planner_ros2

# source your ROS 2 installation
source /opt/ros/humble/setup.bash

# build all three packages
colcon build --packages-select kinematic_planner_interfaces robot_3r_description kinematic_planner

# source the workspace overlay
source install/setup.bash
```

---

## Run

### RRT\* planner (default)

```bash
ros2 launch launch/planner.launch.py
```

Plan to a specific goal configuration:

```bash
ros2 launch launch/planner.launch.py goal_config:="[1.5093, 0.6072, 1.4052]"
```

Enable the dense obstacle ring (10 obstacles arranged in a circle around the robot):

```bash
ros2 launch launch/planner.launch.py is_dense:=true
```

### Informed RRT\* planner

Informed RRT\* runs the same RRT\* algorithm until a first solution is found, then
focuses all subsequent sampling inside the smallest ellipsoid in C-space that can contain
any path of equal or lower cost — converging to the optimum faster than plain RRT\*.

```bash
ros2 run kinematic_planner informed_rrt_star_node --ros-args \
  -p goal_config:="[1.5093, 0.6072, 1.4052]" \
  -p rrts_search_until_max_iter:=true \
  -p robot_description:="$(xacro src/robot_3r_description/urdf/robot_3r.urdf.xacro)"
```

### Verify a path was found

```bash
ros2 topic echo /smpb_planner/jsp_path --once
```

You should see a `JointSpacePath` message with waypoints from the start configuration
to the goal.

---

## Launch arguments

| Argument | Default | Description |
|---|---|---|
| `goal_config` | `[1.5093, 0.6072, 1.4052]` | Goal joint configuration in radians |
| `is_dense` | `false` | `true` → 10-obstacle ring; `false` → 2 sparse obstacles |
| `check_collision` | `true` | Enable FCL collision checking |
| `collision_checker` | `proximity` | `proximity` (signed distance) or `bvol` (bounding volume) |
| `rrts_max_iter` | `300` | Maximum RRT\* iterations |
| `verbose` | `false` | Print per-iteration planning logs |
| `world_frame` | `world` | Fixed frame name |
| `base_link_name` | `base_link` | Robot base link name |
| `dense_platform_height` | `0.755` | First-link height used for dense obstacle placement (metres) |
| `dense_ring_radius` | `0.45` | Radius of the dense obstacle ring (metres) |

---

## Planner node parameters

All parameters can be overridden at runtime with `--ros-args -p <name>:=<value>`.

| Parameter | Default | Description |
|---|---|---|
| `robot_description` | — | URDF XML string (set by launch file) |
| `goal_config` | joint-limit midpoints | Goal joint angles in radians |
| `planning_algorithm` | `rrt_star` | Planning algorithm (`rrt_star` only for now) |
| `rrts_expand_dist` | `0.3` | Max tree growth per iteration (radians) |
| `rrts_path_resolution` | `0.1` | Interpolation step size (radians) |
| `rrts_max_iter` | `300` | Maximum iterations |
| `rrts_connect_circle_dist` | `20` | Neighbour search radius multiplier |
| `rrts_search_until_max_iter` | `false` | Keep improving after first solution |
| `use_goal_biased_sampling` | `false` | Bias sampling toward goal |
| `rrts_goal_sample_rate` | `0.3` | Goal bias probability (when enabled) |
| `collision_checker` | `proximity` | `proximity` or `bvol` |
| `min_obs_dist` | `0.1` | Minimum safe clearance from obstacles (metres) |
| `random_seed` | `42` | Seed for reproducible results |
| `disabled_collision_pairs` | `[""]` | Self-collision pairs to skip, as `"link_a:link_b"` strings |

---

## Using a different robot

1. Add your URDF or xacro to the workspace (or point to an existing package).
2. Edit `launch/planner.launch.py` — change `urdf_file` to your robot's xacro path.
3. Set `disabled_collision_pairs` to the adjacent link pairs you want to exclude from
   self-collision checking (equivalent to `<disable_collisions>` entries in an SRDF):
   ```bash
   ros2 launch launch/planner.launch.py \
     disabled_collision_pairs:="['base_link:link1','link1:link2','link2:link3']"
   ```
4. Set `dense_platform_height` to match your robot's first link height if using
   the dense obstacle mode.

The robot must use **convex collision primitives** (`<box>`, `<cylinder>`, or `<sphere>`)
in its URDF `<collision>` elements.  Mesh-based collision geometry is not supported.

---

## Algorithm references

- **RRT\***: Karaman & Frazzoli, "Sampling-based algorithms for optimal motion planning," *IJRR* 2011.
- **Informed RRT\***: Gammell, Srinivasa & Barfoot, "Informed RRT\*: Optimal Sampling-based Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic," *IROS* 2014.
- Implementation structure adapted from [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics).

---

## License

MIT
