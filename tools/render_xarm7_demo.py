#!/usr/bin/env python3
"""Render an animated RRT* planning demo for the 7-DOF xArm7 arm, using
MuJoCo as the sim/render backend.

Runs the real ROS-independent planning/collision stack (RobotConfig,
build_link_collision_shapes, build_collision_fn, RRTStar) against a narrow-
passage obstacle scene sized for xArm7's reach, then renders the resulting
path by driving a MuJoCo model's joint positions directly (kinematic
playback only, no dynamics simulation) and recording MuJoCo's offscreen
renderer frame by frame.

The MuJoCo model is MuJoCo's own auto-conversion of xarm7_description's
URDF (mujoco.mj_saveLastXML), with a ground plane, camera, lighting, a
mounting table, and the obstacles added programmatically -- the SAME
obstacle positions this script hands to the real FCL-based collision_fn,
so what you see is what the planner actually checked against.

Requires the workspace built and sourced (colcon build, source
install/setup.bash).

Usage:
    python3 tools/render_xarm7_demo.py
"""
import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

for _name in [n for n in sys.modules if n == "mpl_toolkits" or n.startswith("mpl_toolkits.")]:
    del sys.modules[_name]

import imageio.v2 as imageio
import mujoco
import numpy as np
from ament_index_python.packages import get_package_share_directory

from kinematic_planner.collision.robot_collision_model import build_link_collision_shapes
from kinematic_planner.planning.interpolate import interpolate_waypoints
from kinematic_planner.planning.rrt_star import RRTStar
from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.scripts.planner_node import _build_rtb_model, build_collision_fn
from kinematic_planner_interfaces.msg import SceneObstacles
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped

START = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MIN_OBS_DIST = 0.04
RRTS_MAX_ITER = 2000
RANDOM_SEED = 42
STEPS_PER_SEGMENT = 4
PLAYBACK_FPS = 30
START_HOLD_FRAMES = 12
HOLD_FRAMES = 24
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")

TABLE_SIZE = (1.1, 1.1, 0.4)
TABLE_POS = (0.0, 0.0, -0.2)

# Three scenes, same table: "sparse" is 3 short obstacles (the easy case),
# "tall" is fewer obstacles than an earlier 7-pillar attempt (found too
# hard to read visually) but each one taller, so the arm has to duck
# under/around rather than mostly clear over the top, and "dense" is 6
# short obstacles (more clutter, same height as "sparse") with a goal
# chosen for a short path despite the added clutter.
SCENES = {
    "sparse": {
        "obstacles": [
            {"pos": (0.35, 0.15, 0.2), "size": (0.08, 0.08, 0.4)},
            {"pos": (0.35, -0.15, 0.2), "size": (0.08, 0.08, 0.4)},
            {"pos": (0.50, 0.0, 0.15), "size": (0.08, 0.08, 0.3)},
        ],
        "goal": [0.2, -0.5, 0.0, 1.0, 0.0, 0.9, 0.0],
        "out_name": "xarm7",
    },
    "tall": {
        "obstacles": [
            {"pos": (0.35, 0.15, 0.3), "size": (0.08, 0.08, 0.6)},
            {"pos": (0.35, -0.15, 0.3), "size": (0.08, 0.08, 0.6)},
            {"pos": (0.50, 0.0, 0.25), "size": (0.08, 0.08, 0.5)},
            {"pos": (0.20, 0.32, 0.3), "size": (0.08, 0.08, 0.6)},
        ],
        "goal": [0.084, -0.74, -0.122, 0.011, 0.654, -0.842, -0.64],
        "out_name": "xarm7_tall",
    },
    "dense": {
        "obstacles": [
            {"pos": (0.35, 0.15, 0.2), "size": (0.08, 0.08, 0.4)},
            {"pos": (0.35, -0.15, 0.2), "size": (0.08, 0.08, 0.4)},
            {"pos": (0.50, 0.0, 0.15), "size": (0.08, 0.08, 0.3)},
            {"pos": (0.45, 0.28, 0.2), "size": (0.08, 0.08, 0.4)},
            {"pos": (0.45, -0.28, 0.2), "size": (0.08, 0.08, 0.4)},
            {"pos": (0.20, 0.30, 0.2), "size": (0.08, 0.08, 0.3)},
        ],
        "goal": [-0.002, -0.61, -0.215, 0.059, -0.085, -0.734, -0.118],
        "out_name": "xarm7_dense",
    },
}


def load_urdf_string() -> str:
    urdf_path = os.path.join(get_package_share_directory("xarm7_description"), "urdf", "xarm7.urdf")
    return open(urdf_path).read()


def obstacle_scene(obstacles) -> SceneObstacles:
    all_boxes = [{"pos": TABLE_POS, "size": TABLE_SIZE}] + obstacles
    scene = SceneObstacles()
    boxes, poses, ids = [], [], []
    for i, obs in enumerate(all_boxes):
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(obs["size"])
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = obs["pos"]
        pose.pose.orientation.w = 1.0
        boxes.append(box)
        poses.append(pose)
        ids.append(i)
    scene.scene_obstacles = boxes
    scene.obstacle_poses = poses
    scene.obstacle_ids = ids
    return scene


def build_mujoco_model(urdf_str: str, workdir: str, obstacles) -> mujoco.MjModel:
    urdf_path = os.path.join(workdir, "xarm7.urdf")
    # mesh paths in the urdf (and, via meshdir, in the saved MJCF below) are
    # relative to its own directory
    real_urdf_dir = os.path.join(get_package_share_directory("xarm7_description"), "urdf")
    os.symlink(os.path.join(real_urdf_dir, "meshes"), os.path.join(workdir, "meshes"))
    with open(urdf_path, "w") as f:
        f.write(urdf_str)
    raw_model = mujoco.MjModel.from_xml_path(urdf_path)
    mjcf_path = os.path.join(workdir, "xarm7.xml")
    mujoco.mj_saveLastXML(mjcf_path, raw_model)
    tree = ET.parse(mjcf_path)

    root = tree.getroot()
    worldbody = root.find("worldbody")

    visual = ET.SubElement(root, "visual")
    visual.append(ET.Element("headlight", {"ambient": "0.4 0.4 0.4"}))
    visual.append(ET.Element("global", {"offwidth": "960", "offheight": "720"}))

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "type": "2d", "builtin": "checker", "name": "grid",
        "rgb1": "0.14 0.16 0.2", "rgb2": "0.09 0.1 0.13", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid", "texrepeat": "4 4", "reflectance": "0.1"})

    ET.SubElement(worldbody, "light", {"pos": "0.5 -0.5 1.5", "dir": "-0.3 0.3 -1", "diffuse": "0.9 0.9 0.85"})
    ET.SubElement(worldbody, "light", {"pos": "-0.6 0.6 1.2", "dir": "0.4 -0.4 -1", "diffuse": "0.4 0.4 0.45"})
    ET.SubElement(worldbody, "geom", {
        "type": "plane", "size": "2 2 0.05", "pos": "0 0 -0.6", "material": "grid", "group": "1",
    })
    ET.SubElement(worldbody, "geom", {
        "type": "box", "size": " ".join(f"{s / 2:.4f}" for s in TABLE_SIZE),
        "pos": " ".join(str(p) for p in TABLE_POS), "rgba": "0.55 0.55 0.58 1", "group": "1",
    })
    for obs in obstacles:
        ET.SubElement(worldbody, "geom", {
            "type": "box", "size": " ".join(f"{s / 2:.4f}" for s in obs["size"]),
            "pos": " ".join(str(p) for p in obs["pos"]), "rgba": "0.88 0.53 0.16 1", "group": "1",
        })

    augmented_path = os.path.join(workdir, "xarm7_scene.xml")
    tree.write(augmented_path)
    return mujoco.MjModel.from_xml_path(augmented_path)


def verify_demo_claims(path, goal, robot_config, collision_fn):
    """Print evidence for the claims this demo makes, instead of asserting
    them silently: that the naive straight-line interpolation between start
    and goal is genuinely invalid (so RRT* did real work), that every
    waypoint the planner returned is itself collision-free, and that the
    path starts and ends exactly where requested."""
    straight_line = interpolate_waypoints([np.array(path[0]), np.array(path[-1])], steps_per_segment=20)
    straight_line_free = all(
        collision_fn(_single_config_node(q)) for q in straight_line
    )
    print(f"Straight-line start->goal collision-free: {straight_line_free} "
          f"(expected False -- proves the obstacles force a genuine detour)")

    all_waypoints_free = all(collision_fn(_single_config_node(np.array(q))) for q in path)
    print(f"Every returned waypoint collision-free: {all_waypoints_free}")

    print(f"Start matches requested start: {np.allclose(path[0], START)}")
    print(f"Goal matches requested goal: {np.allclose(path[-1], goal)}")


def _single_config_node(q):
    from kinematic_planner.planning.tree import TreeNode
    node = TreeNode(np.asarray(q))
    node.path_q = [node.q]
    return node


def find_goal_with_solution(robot_config, collision_fn, goal, rng_seed):
    """START is fixed; goal is a hand-picked reachable, collision-free
    configuration on the far side of the obstacle cluster from START,
    verified below to have an RRT* solution."""
    rng = np.random.default_rng(rng_seed)
    planner = RRTStar(
        start=START, goal=goal, joint_limits=robot_config.joint_limits,
        expand_dist=0.3, path_resolution=0.1, max_iter=RRTS_MAX_ITER,
        connect_circle_dist=25, goal_sample_rate=0.2, collision_fn=collision_fn,
        use_goal_biased_sampling=True, goal_noise_sigma=0.4, rng=rng,
    )
    return planner, planner.plan()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(SCENES), default="sparse")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()
    scene = SCENES[args.scene]
    obstacles, goal = scene["obstacles"], scene["goal"]
    out_dir = args.out_dir
    out_name = args.out_name or scene["out_name"]
    os.makedirs(out_dir, exist_ok=True)

    urdf_str = load_urdf_string()
    robot_config = RobotConfig.from_urdf(urdf_str)
    link_shapes = build_link_collision_shapes(ET.fromstring(urdf_str))

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_str)
        rtb_path = f.name
    rtb_model = _build_rtb_model(rtb_path)
    os.remove(rtb_path)

    collision_fn = build_collision_fn(
        robot_config=robot_config, link_shapes=link_shapes, obstacle_geom=obstacle_scene(obstacles),
        rtb_model=rtb_model, collision_checker="proximity", min_obs_dist=MIN_OBS_DIST,
        check_collision=True,
    )

    planner, path = find_goal_with_solution(robot_config, collision_fn, goal, RANDOM_SEED)
    if path is None:
        print("No path found -- aborting render.", file=sys.stderr)
        sys.exit(1)
    print(f"Path found: {len(path)} waypoints, cost {planner.compute_path_cost(path):.3f} rad")
    verify_demo_claims(path, goal, robot_config, collision_fn)

    frames_q = interpolate_waypoints([np.array(q) for q in path], STEPS_PER_SEGMENT)
    frames_q = [frames_q[0]] * (START_HOLD_FRAMES - 1) + frames_q
    frames_q = frames_q + [frames_q[-1]] * HOLD_FRAMES

    import tempfile
    with tempfile.TemporaryDirectory() as workdir:
        model = build_mujoco_model(urdf_str, workdir, obstacles)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.lookat = [0.2, 0.0, 0.45]
    cam.distance = 1.6
    cam.azimuth = 130
    cam.elevation = -12
    render_opt = mujoco.MjvOption()
    render_opt.geomgroup[0] = 0  # hide collision primitives
    render_opt.geomgroup[1] = 1  # show visual meshes, table, obstacles, floor

    frame_dir = os.path.join(out_dir, f"_frames_{out_name}")
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []
    for i, q in enumerate(frames_q):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=render_opt)
        pixels = renderer.render()
        path_png = os.path.join(frame_dir, f"frame_{i:04d}.png")
        imageio.imwrite(path_png, pixels)
        frame_paths.append(path_png)
    renderer.close()

    mp4_path = os.path.join(out_dir, f"{out_name}_demo.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(PLAYBACK_FPS), "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        mp4_path,
    ], check=True)
    print(f"wrote {mp4_path}")

    palette_path = os.path.join(out_dir, f"_{out_name}_palette.png")
    gif_path = os.path.join(out_dir, f"{out_name}_demo.gif")
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path, "-vf", "fps=20,scale=560:-1:flags=lanczos,palettegen",
        palette_path,
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path, "-i", palette_path,
        "-filter_complex", "fps=20,scale=560:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none",
        gif_path,
    ], check=True)
    os.remove(palette_path)
    print(f"wrote {gif_path}")

    poster_path = os.path.join(out_dir, f"{out_name}_poster.png")
    imageio.imwrite(poster_path, imageio.imread(frame_paths[len(frames_q) - HOLD_FRAMES - 1]))
    print(f"wrote {poster_path}")

    for p in frame_paths:
        os.remove(p)
    os.rmdir(frame_dir)


if __name__ == "__main__":
    main()
