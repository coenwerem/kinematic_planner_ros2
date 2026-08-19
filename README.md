# kinematic_planner_ros2

![RRT* threading the 7-DOF xArm7 through a narrow-passage obstacle cluster, rendered in MuJoCo](media/xarm7_demo.gif)

A suite of ROS 2 packages implementing **RRT\*** and **Informed RRT\*** path planning for robot manipulators, built from scratch with no MoveIt dependency: collision checking runs directly against the
[Flexible Collision Library (FCL)](https://github.com/humanoid-path-planner/hpp-fcl), forward kinematics use the [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python), and the robot is described entirely by a URDF, so any URDF-described manipulator plugs in without SRDF files or a MoveIt config package.

The planner supports any N-DOF robot whose links use **convex collision primitives**
(box, cylinder, sphere) in their URDF `<collision>` elements — proven above on
the 7-DOF [xArm7](#realistic-robot-example-xarm7) and, as a smaller reference
example kept in the repo, a 3-DOF serial manipulator (3R arm).

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

## Realistic robot example: xArm7

The hero GIF at the top runs the exact same planning stack (`RobotConfig`,
`build_collision_fn`, `RRTStar`) against `xarm7_description`'s 7-DOF xArm7
URDF, no code changes required — proof that the "any URDF-described
manipulator" claim above holds for more than the bundled 3R arm. The scene
is a narrow-passage obstacle cluster sized to the arm's reach, standing on
a small mounting table.

MuJoCo (not RViz or the matplotlib renderer used for the 3R demo) is the
sim/render backend here: `tools/render_xarm7_demo.py` builds a MuJoCo model
directly from the xArm7 URDF (MuJoCo's own URDF importer, with the visual
meshes kept), adds the table and obstacles as extra MuJoCo bodies at the
exact same positions the real FCL-based `collision_fn` checks against, then
drives the model's joint positions frame by frame along the interpolated
RRT\* path and records MuJoCo's offscreen renderer. The recording is
kinematic playback, not a dynamics simulation: the script sets joint
positions directly rather than driving them through actuator control or
physics integration, so the recording validates neither torque, contact
forces, nor timing.

The script prints, rather than silently assumes, the properties the demo
is meant to prove: the naive straight-line interpolation between start
and goal is genuinely in collision (proving the obstacles forced a real
detour), every waypoint RRT\* returned is itself collision-free, and the
path starts and ends exactly at the requested configurations.

Reproduce the recording (needs `mujoco`; see [Prerequisites](#prerequisites)):

```bash
colcon build --packages-select xarm7_description kinematic_planner_interfaces robot_3r_description kinematic_planner
source install/setup.bash
python3 tools/render_xarm7_demo.py
```

`src/xarm7_description/urdf/xarm7.urdf` and its meshes are copied from the
MIT-licensed [`frogger`](https://github.com/albertli24/frogger) project's
`models/xarm7/` directory; see `src/xarm7_description/NOTICE.md` for exact
provenance and the one line added to the URDF, a MuJoCo-only compiler
directive that has no effect on ROS or FCL. The arm's own collision
geometry is convex primitives (cylinders and spheres), not meshes — mesh
collision support exists in `collision/robot_collision_model.py` and is
covered by `test/collision/test_robot_collision_model.py`, but the xArm7
URDF's own collision geometry does not exercise the mesh path.

---

## Workspace Layout

```
kinematic_planner_ros2/
└── src/
    ├── kinematic_planner/           # main package — planner, collision, robot model
    │   ├── launch/
    │   │   └── planner.launch.py    # single entry point; brings up robot_state_publisher,
    │   │                             # joint_state_publisher, robot_geom_publisher,
    │   │                             # obstacle_publisher, and the selected planner node
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
    ├── xarm7_description/           # xArm7 URDF + meshes (realistic robot example)
    └── kinematic_planner_interfaces/ # custom ROS 2 message definitions
tools/
├── render_demo.py           # 3R RRT* demo: matplotlib, no RViz
└── render_xarm7_demo.py     # xArm7 RRT* demo: MuJoCo sim/render backend
```

---

## Prerequisites

### System

- **Ubuntu 24.04** (Noble) — tested environment
- **ROS 2 Jazzy** (desktop install recommended)
- **xacro** — for processing the robot description:
  ```bash
  sudo apt install ros-jazzy-xacro
  ```
- **robot_state_publisher** and **joint_state_publisher** — the launch file
  runs both nodes to publish `/robot_description` and a start `/joint_states`:
  ```bash
  sudo apt install ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher
  ```

### Python packages

Install into your ROS 2 Python environment:

```bash
pip install \
  "roboticstoolbox-python>=1.3.1" \
  "spatialgeometry>=1.3.0" \
  spatialmath-python \
  transforms3d \
  trimesh \
  mujoco \
  imageio \
  "numpy>=2.0"
```

`mujoco` and `imageio` are only needed to render the demo GIFs/MP4s
(`tools/render_demo.py`, `tools/render_xarm7_demo.py`); the planner and
tests do not import either.

> **Note on `numpy` 2.x:** `roboticstoolbox-python` releases before 1.2 and
> `spatialgeometry` releases before 1.3.0 ship compiled extensions built
> against the NumPy 1.x ABI and fail to import under NumPy 2.x
> (`ImportError: numpy.core.multiarray failed to import`). The versions
> pinned above are the first to publish NumPy 2-compatible wheels.

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
source /opt/ros/jazzy/setup.bash

# build all three packages
colcon build --packages-select kinematic_planner_interfaces robot_3r_description kinematic_planner

# source the workspace overlay
source install/setup.bash
```

---

## Run: the 3R reference example

The 3R reference example below is the smaller 3-DOF arm bundled with the
repo as a quick, dependency-light demo. For the 7-DOF xArm7 MuJoCo demo
shown at the top, see [Realistic robot example: xArm7](#realistic-robot-example-xarm7).

To watch the 3R arm plan live in RViz, showing the final path as markers
(not an animated sweep, unlike the MuJoCo demo above):

```bash
ros2 launch kinematic_planner planner.launch.py &
rviz2 -d src/robot_3r_description/rviz/view_3r_demo.rviz
```

### RRT\* planner (default)

```bash
ros2 launch kinematic_planner planner.launch.py
```

Plan to a specific goal configuration:

```bash
ros2 launch kinematic_planner planner.launch.py goal_config:="[1.5, -0.3, 0.6]"
```

Enable the dense obstacle ring (10 obstacles standing on the platform around the robot);
the default `goal_config` is collision-free against both scenes:

```bash
ros2 launch kinematic_planner planner.launch.py is_dense:=true
```

### Informed RRT\* planner

Informed RRT\* runs the same RRT\* algorithm until a first solution is found, then
focuses all subsequent sampling inside the smallest ellipsoid in C-space that can contain
any path of equal or lower cost — converging to the optimum faster than plain RRT\*.

```bash
ros2 launch kinematic_planner planner.launch.py algorithm:=informed_rrt_star
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
| `goal_config` | `[0.8, -0.5, 0.5]` | Goal joint configuration in radians |
| `algorithm` | `rrt_star` | `rrt_star` or `informed_rrt_star` — which planner node to launch |
| `is_dense` | `false` | `true` → 10-obstacle ring; `false` → 2 sparse obstacles |
| `check_collision` | `true` | Enable FCL collision checking |
| `collision_checker` | `proximity` | `proximity` (signed distance) or `bvol` (bounding volume) |
| `rrts_max_iter` | `2000` | Maximum RRT\* iterations |
| `platform_height` | `0.755` | Height of the robot's mounting platform; obstacles rest on top of the platform |
| `verbose` | `false` | Print per-iteration planning logs |
| `world_frame` | `world` | Fixed frame name |
| `base_link_name` | `base_link` | Robot base link name |
| `dense_ring_radius` | `0.45` | Radius of the dense obstacle ring (metres) |

---

## Planner node parameters

All parameters can be overridden at runtime with `--ros-args -p <name>:=<value>`.

| Parameter | Default | Description |
|---|---|---|
| `robot_description` | — | URDF XML string (set by launch file) |
| `goal_config` | joint-limit midpoints | Goal joint angles in radians |
| `planning_algorithm` | `rrt_star` | Log label only; the launch file's `algorithm` argument picks which node runs, not `planning_algorithm` |
| `rrts_expand_dist` | `0.3` | Max tree growth per iteration (radians) |
| `rrts_path_resolution` | `0.1` | Interpolation step size (radians) |
| `rrts_max_iter` | `2000` | Maximum iterations |
| `rrts_connect_circle_dist` | `20` | Neighbour search radius multiplier |
| `rrts_search_until_max_iter` | `false` | Keep improving after first solution |
| `use_goal_biased_sampling` | `false` | Bias sampling toward goal |
| `rrts_goal_sample_rate` | `0.3` | Goal bias probability (when enabled) |
| `collision_checker` | `proximity` | `proximity` or `bvol` |
| `min_obs_dist` | `0.1` | Minimum safe clearance from obstacles (metres) |
| `random_seed` | `42` | Seed for reproducible results |
| `disabled_collision_pairs` | `[""]` | Additional self-collision pairs to skip beyond the joint-adjacent pairs `RobotConfig.from_urdf` already auto-excludes, as `"link_a:link_b"` strings |

---

## Using a different robot

1. Add your URDF or xacro to the workspace (or point to an existing package).
2. Edit `src/kinematic_planner/launch/planner.launch.py` — change `urdf_file` to your robot's xacro path.
3. `RobotConfig.from_urdf` auto-excludes every joint-adjacent link pair from self-collision
   checking, computed from the URDF's own joint parent/child structure, so no manual listing
   of adjacent pairs is required. Use `disabled_collision_pairs` only for further exclusions
   on top of the automatic adjacency exclusion, e.g. non-adjacent link pairs your robot's
   geometry makes intentionally non-colliding (equivalent to extra `<disable_collisions>`
   entries in an SRDF):
   ```bash
   ros2 launch kinematic_planner planner.launch.py \
     disabled_collision_pairs:="['base_link:link3']"
   ```
4. Set `platform_height` to your robot's mounting platform height, so the
   built-in obstacle scenes rest on top of the platform instead of floating through the platform.

The robot must use **convex collision primitives** (`<box>`, `<cylinder>`, or `<sphere>`)
in its URDF `<collision>` elements.  Mesh-based collision geometry is not supported.

---

## Algorithm references

- **RRT\***: Karaman & Frazzoli, "Sampling-based algorithms for optimal motion planning," *IJRR* 2011.
- **Informed RRT\***: Gammell, Srinivasa & Barfoot, "Informed RRT\*: Optimal Sampling-based Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic," *IROS* 2014.
- Implementation structure adapted from [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics).

---

## Citation

If `kinematic_planner_ros2`'s planning infrastructure supported your work,
please cite the software directly (see `CITATION.cff`), and consider
citing the related paper below:

```bibtex
@inproceedings{enwerem2026variational,
  title={Variational Neural Belief Parameterizations for Robust Dexterous Grasping under Multimodal Uncertainty},
  author={Enwerem, Clinton and Kalyanaraman, Shreya and Baras, John S. and Belta, Calin},
  booktitle={Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026},
  eprint={2604.25897},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  note={Accepted for publication}
}
```

---

## License

MIT
