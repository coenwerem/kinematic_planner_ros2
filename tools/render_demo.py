#!/usr/bin/env python3
"""Render an animated RRT* planning demo for the 3R arm without RViz.

Runs the real ROS-independent planning/collision stack (RRTStar,
build_collision_fn, RobotConfig, the URDF-driven collision model) against
the same scene planner.launch.py uses by default, then renders the
resulting path as a matplotlib 3D animation: linear interpolation between
waypoints (kinematic playback only, no dynamics or timing model), one
frame per interpolation step, assembled into a GIF, MP4, and poster PNG.

Requires the workspace built and sourced (colcon build, source
install/setup.bash) so `kinematic_planner` and `robot_3r_description`
are importable/findable.

Usage:
    python3 tools/render_demo.py
    python3 tools/render_demo.py --elev 30 --azim -120 --out-dir media/_variants --out-name variant_a
"""
import argparse
import os
import subprocess
import sys

# See kinematic_planner/__init__.py: an apt-installed python3-matplotlib
# nspkg.pth shim can bind sys.modules["mpl_toolkits"] to a stale directory
# before this script ever runs, breaking mpl_toolkits.mplot3d.
for _name in [n for n in sys.modules if n == "mpl_toolkits" or n.startswith("mpl_toolkits.")]:
    del sys.modules[_name]

import matplotlib
matplotlib.use("Agg")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from ament_index_python.packages import get_package_share_directory
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kinematic_planner.collision.robot_collision_model import (
    build_link_collision_shapes,
    link_shapes_to_fcl_objects,
)
from kinematic_planner.planning.interpolate import interpolate_waypoints
from kinematic_planner.planning.rrt_star import RRTStar
from kinematic_planner.robot.robot_config import RobotConfig
from kinematic_planner.scripts.obstacle_publisher import default_obstacle_scene
from kinematic_planner.scripts.planner_node import _build_rtb_model, build_collision_fn
import xml.etree.ElementTree as ET

from kinematic_planner_interfaces.msg import SceneObstacles
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped

START = [0.0, 0.0, 0.0]
GOAL = [-1.5, 0.5, -0.9]
PLATFORM_HEIGHT = 0.755
IS_DENSE = True
MIN_OBS_DIST = 0.1
RRTS_MAX_ITER = 2000
RANDOM_SEED = 42
STEPS_PER_SEGMENT = 12
PLAYBACK_FPS = 15
START_HOLD_FRAMES = 12
HOLD_FRAMES = 20
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")


def obstacle_positions_and_sizes():
    positions, sizes = default_obstacle_scene(is_dense=IS_DENSE, platform_height=PLATFORM_HEIGHT)
    n = len(positions) // 3
    return (
        [tuple(positions[3 * i:3 * i + 3]) for i in range(n)],
        [tuple(sizes[3 * i:3 * i + 3]) for i in range(n)],
    )


def obstacle_scene() -> SceneObstacles:
    positions, sizes = obstacle_positions_and_sizes()
    scene = SceneObstacles()
    boxes, poses, ids = [], [], []
    for i, (pos, size) in enumerate(zip(positions, sizes)):
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos
        pose.pose.orientation.w = 1.0
        boxes.append(box)
        poses.append(pose)
        ids.append(i)
    scene.scene_obstacles = boxes
    scene.obstacle_poses = poses
    scene.obstacle_ids = ids
    return scene


def load_urdf_string() -> str:
    urdf_xacro = os.path.join(
        get_package_share_directory("robot_3r_description"), "urdf", "robot_3r.urdf.xacro"
    )
    return subprocess.check_output(["xacro", urdf_xacro], text=True)


DEFAULT_LINK_COLOR = "#2f6fb0"


def parse_link_colors(urdf_root) -> dict:
    """Map link name -> the link's own <visual><material> color from the
    URDF, so the render matches the robot's real per-link materials
    (e.g. base_link is grey, not the same blue as the moving arm links)
    instead of flattening every link to one made-up color."""
    named_colors = {}
    for material_el in urdf_root.findall("material"):
        name = material_el.get("name")
        color_el = material_el.find("color")
        if name and color_el is not None:
            r, g, b, _a = (float(v) for v in color_el.get("rgba").split())
            named_colors[name] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

    link_colors = {}
    for link_el in urdf_root.findall("link"):
        link_name = link_el.get("name")
        visual_el = link_el.find("visual")
        material_el = visual_el.find("material") if visual_el is not None else None
        material_name = material_el.get("name") if material_el is not None else None
        hex_color = named_colors.get(material_name, DEFAULT_LINK_COLOR)
        link_colors[link_name] = _lighten_if_too_dark(hex_color)
    return link_colors


def _lighten_if_too_dark(hex_color, min_channel_max=90, brightness_boost=70):
    """The URDF's real materials include near-black/dark-grey links (link0,
    base_link) that would be nearly invisible against a dark render
    background; add a fixed brightness boost to those rather than a
    proportional blend toward white, since link0 (black) and base_link
    (dark grey) need to stay visually distinct from each other (link0 is
    a pedestal sitting on top of base_link), and a proportional blend
    compresses two already-close dark colors even closer together.
    Colors already visible against the background are left untouched."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    if max(r, g, b) >= min_channel_max:
        return hex_color
    r, g, b = (min(255, c + brightness_boost) for c in (r, g, b))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def box_faces(center, size, rotation=np.eye(3)):
    dx, dy, dz = np.array(size) / 2.0
    corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
    ])
    corners = corners @ rotation.T + np.array(center)
    faces_idx = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4],
    ]
    return [corners[f] for f in faces_idx]


def link_boxes_world(rtb_model, link_shapes, link_names, q):
    """(link_name, center, size, rotation) for every box collision shape,
    evaluated at joint configuration q."""
    boxes = []
    for link_name in link_names:
        T = rtb_model.fkine(q, end=link_name, include_base=True).A
        for shape in link_shapes[link_name]:
            if not hasattr(shape.fcl_geometry, "side"):
                continue
            world_T = T @ shape.local_origin
            boxes.append((link_name, world_T[:3, 3], shape.fcl_geometry.side, world_T[:3, :3]))
    return boxes


def ee_positions(rtb_model, frames_q):
    """End-effector (tool_link) world position at every interpolated
    frame, matching the real ee_path marker planner_node.py publishes
    (rtb_model.fkine(q) with no end= argument picks the kinematic
    chain's terminal link)."""
    return [rtb_model.fkine(q, include_base=True).A[:3, 3] for q in frames_q]


def scene_bounds(rtb_model, link_shapes, link_names, frames_q, obstacles, ee_path):
    """Axis-aligned bounds covering every link box, the end-effector path,
    and the (static) obstacles across the whole animation, so the camera
    framing never clips or jumps between frames."""
    corners = []
    for q in frames_q:
        for _, center, size, rotation in link_boxes_world(rtb_model, link_shapes, link_names, q):
            for face in box_faces(center, size, rotation):
                corners.extend(face)
    for pos, size in obstacles:
        for face in box_faces(pos, size):
            corners.extend(face)
    corners.extend(ee_path)
    corners = np.array(corners)
    return corners.min(axis=0), corners.max(axis=0)


VIEW_ELEV, VIEW_AZIM = 38, -50


def _camera_direction(elev=VIEW_ELEV, azim=VIEW_AZIM):
    elev, azim = np.radians(elev), np.radians(azim)
    return np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])


def draw_scene(ax, rtb_model, link_shapes, link_names, link_colors, q, obstacles, ee_path_so_far, bounds,
               elev=VIEW_ELEV, azim=VIEW_AZIM):
    ax.clear()
    # matplotlib's Poly3DCollection z-ordering is a rough painter's algorithm
    # (per-collection centroid, not a real depth buffer) and gets small
    # objects sitting flush on a large flat face -- like link0 resting on
    # base_link's top -- backwards. Draw explicitly farthest-from-camera
    # first so nearer boxes always occlude farther ones correctly.
    view_dir = _camera_direction(elev, azim)
    boxes = list(link_boxes_world(rtb_model, link_shapes, link_names, q))
    boxes = [(name, c, s, r, "link") for name, c, s, r in boxes]
    boxes += [(None, np.array(pos), size, np.eye(3), "obstacle") for pos, size in obstacles]
    boxes.sort(key=lambda b: np.dot(b[1], view_dir))
    for link_name, center, size, rotation, kind in boxes:
        faces = box_faces(center, size, rotation)
        if kind == "link":
            ax.add_collection3d(Poly3DCollection(
                faces, facecolor=link_colors[link_name], edgecolor="#10131a", linewidths=0.6, alpha=0.95,
            ))
        else:
            ax.add_collection3d(Poly3DCollection(
                faces, facecolor="#e0872a", edgecolor="#7a4712", linewidths=0.6, alpha=0.9,
            ))
    if len(ee_path_so_far) > 1:
        pts = np.array(ee_path_so_far)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#3ddc4a", linewidth=2.0)
    lo, hi = bounds
    pad = 0.08 * np.max(hi - lo)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(0.0, hi[2] + pad)
    extent = hi - lo
    ax.set_box_aspect((extent[0] + 2 * pad, extent[1] + 2 * pad, hi[2] + pad))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elev", type=float, default=VIEW_ELEV)
    parser.add_argument("--azim", type=float, default=VIEW_AZIM)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--out-name", default="rrt_star_3r", help="base name; produces <name>_demo.{gif,mp4} and <name>_poster.png")
    args = parser.parse_args()
    elev, azim = args.elev, args.azim
    out_dir = args.out_dir
    out_name = args.out_name

    os.makedirs(out_dir, exist_ok=True)
    urdf_str = load_urdf_string()

    robot_config = RobotConfig.from_urdf(urdf_str, base_link_name="base_link")
    link_shapes = build_link_collision_shapes(ET.fromstring(urdf_str))
    link_colors = parse_link_colors(ET.fromstring(urdf_str))
    rtb_model = _build_rtb_model_from_string(urdf_str)

    positions, sizes = obstacle_positions_and_sizes()
    obstacles = list(zip(positions, sizes))
    collision_fn = build_collision_fn(
        robot_config=robot_config,
        link_shapes=link_shapes,
        obstacle_geom=obstacle_scene(),
        rtb_model=rtb_model,
        collision_checker="proximity",
        min_obs_dist=MIN_OBS_DIST,
        check_collision=True,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    planner = RRTStar(
        start=START, goal=GOAL, joint_limits=robot_config.joint_limits,
        expand_dist=0.3, path_resolution=0.1, max_iter=RRTS_MAX_ITER,
        connect_circle_dist=20, goal_sample_rate=0.3, collision_fn=collision_fn,
        rng=rng,
    )
    path = planner.plan()
    if path is None:
        print("No path found -- aborting render.", file=sys.stderr)
        sys.exit(1)
    print(f"Path found: {len(path)} waypoints, cost {planner.compute_path_cost(path):.3f} rad")

    frames_q = interpolate_waypoints([np.array(q) for q in path], STEPS_PER_SEGMENT)
    # hold on the start pose for a beat before the sweep begins
    frames_q = [frames_q[0]] * (START_HOLD_FRAMES - 1) + frames_q
    link_names = [n for n in link_shapes if n != robot_config.base_link_name] + [robot_config.base_link_name]
    ee_path = ee_positions(rtb_model, frames_q)
    bounds = scene_bounds(rtb_model, link_shapes, link_names, frames_q, obstacles, ee_path)

    fig = plt.figure(figsize=(8, 6), dpi=110)
    fig.patch.set_facecolor("#1c1f26")
    ax = fig.add_axes([-0.1, -0.02, 1.2, 1.15], projection="3d")
    # matplotlib's automatic z-order computation gets small objects sitting
    # flush on large flat faces backwards (see draw_scene); disabling it
    # makes the explicit camera-distance sort in draw_scene actually apply.
    ax.computed_zorder = False
    ax.set_facecolor("#1c1f26")
    frame_paths = []
    frame_dir = os.path.join(out_dir, f"_frames_{out_name}")
    os.makedirs(frame_dir, exist_ok=True)
    for i, q in enumerate(frames_q):
        draw_scene(ax, rtb_model, link_shapes, link_names, link_colors, q, obstacles, ee_path[:i + 1], bounds,
                   elev=elev, azim=azim)
        path_png = os.path.join(frame_dir, f"frame_{i:04d}.png")
        fig.savefig(path_png, facecolor="#1c1f26")
        frame_paths.append(path_png)
    # hold on the final frame for a beat so the loop reads as "arrived"
    for h in range(HOLD_FRAMES):
        hold_png = os.path.join(frame_dir, f"frame_{len(frames_q) + h:04d}.png")
        import shutil
        shutil.copy(frame_paths[-1], hold_png)
        frame_paths.append(hold_png)
    plt.close(fig)

    gif_path = os.path.join(out_dir, f"{out_name}_demo.gif")
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, fps=PLAYBACK_FPS, loop=0)
    print(f"wrote {gif_path}")

    mp4_path = os.path.join(out_dir, f"{out_name}_demo.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(PLAYBACK_FPS), "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        mp4_path,
    ], check=True)
    print(f"wrote {mp4_path}")

    poster_path = os.path.join(out_dir, f"{out_name}_poster.png")
    imageio.imwrite(poster_path, images[len(frames_q) - 1])
    print(f"wrote {poster_path}")

    for p in frame_paths:
        os.remove(p)
    os.rmdir(frame_dir)


def _build_rtb_model_from_string(urdf_str: str):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_str)
        path = f.name
    try:
        return _build_rtb_model(path)
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
