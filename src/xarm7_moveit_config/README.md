# xarm7_moveit_config

MoveIt 2 configuration for the 7-DOF xArm7, used by the planner comparison in
`tools/benchmark_moveit_planners.py`. It registers four planning pipelines
against one robot model so their results can be measured side by side. The
comparison, its problem setup, and the recorded results live in the
[top-level README](../../README.md#against-the-moveit-2-planning-ecosystem).

This package configures MoveIt's own planners. It does not expose the RRT* and
Informed RRT* implementations in `kinematic_planner` through a MoveIt planner
plugin.

## Pipelines

| Pipeline | Plugin | Planners configured |
|---|---|---|
| `ompl` | `ompl_interface/OMPLPlanner` | RRTConnect, RRT*, PRM |
| `chomp` | `chomp_interface/CHOMPPlanner` | covariant gradient optimization |
| `stomp` | `stomp_moveit/StompPlanner` | stochastic trajectory optimization |
| `pilz_industrial_motion_planner` | `pilz_industrial_motion_planner/CommandPlanner` | PTP, LIN, CIRC |

The default pipeline is `ompl` with RRTConnect.

## Configuration files

| File | Purpose |
|---|---|
| `config/xarm7.srdf` | `xarm7` planning group over the `link_base` to `link7` chain, a `home` state, and adjacent-link collision disables derived from the URDF tree |
| `config/kinematics.yaml` | KDL kinematics solver for the group |
| `config/joint_limits.yaml` | velocity limits mirroring the URDF, plus per-joint acceleration limits |
| `config/ompl_planning.yaml` | OMPL planner configs, projection evaluator, and `longest_valid_segment_fraction` |
| `config/chomp_planning.yaml` | CHOMP parameters including `ridge_factor` and failure recovery |
| `config/stomp_planning.yaml` | STOMP rollout count, iteration schedule, and cost weights |
| `config/pilz_industrial_motion_planner_planning.yaml` | Pilz command planner and sequence capabilities |
| `config/pilz_cartesian_limits.yaml` | Cartesian velocity and acceleration bounds required by Pilz |

Non-adjacent link pairs stay enabled for collision checking, so the planners
verify self-collision across the whole arm. Pilz refuses to plan without
per-joint acceleration limits, which is why `joint_limits.yaml` sets them
explicitly and uses one value across all seven joints, giving no planner an
advantage from asymmetric limits.

## Launch

```bash
ros2 launch xarm7_moveit_config move_group.launch.py
```

The launch file brings up `move_group` with all four pipelines loaded, alongside
`robot_state_publisher` and `joint_state_publisher`. No controllers are
configured, because the benchmark only plans and never executes a trajectory.
