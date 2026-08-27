#!/usr/bin/env python3
"""Render the three-panel demonstration for the planner comparison.

The comparison figure reports success, path length, and planning time as
numbers. This renderer shows what those numbers look like on the arm. It
replays three motions for the same start, goal, and obstacle set, side by side
and frame-synchronized:

    1. direct joint interpolation, which drives the arm through an obstacle;
    2. the kinematic_planner RRT*;
    3. the shortest MoveIt solution on the query.

Every frame of every panel is checked with the repository's FCL collision
function. A panel whose current configuration is in collision is drawn with a
red border and its obstacles turn red, so the blocked motion is legible without
reading a caption.

The trajectories come from `results/moveit_comparison.json`, so the animation
replays paths the benchmark actually recorded.

Usage:
    python3 tools/render_moveit_demo.py results/moveit_comparison.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

for _name in [n for n in sys.modules if n == "mpl_toolkits" or n.startswith("mpl_toolkits.")]:
    del sys.modules[_name]

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ament_index_python.packages import get_package_share_directory

from kinematic_planner.planning.interpolate import interpolate_waypoints

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_local_planner import _node, build_stack  # noqa: E402

PANEL_W, PANEL_H = 640, 620
FPS = 20
HOLD_START, HOLD_END = 8, 18
FRAMES = 110
LABEL_H = 92

TABLE_RGBA = "0.55 0.55 0.58 1"  # mounting table, matching the xArm7 demos
OK_RGBA = "0.88 0.53 0.16 1"     # obstacle amber, matching the xArm7 demos
HIT_RGBA = "0.85 0.15 0.12 1"    # obstacle red while the panel is in collision
INK = (24, 26, 30)
RED = (176, 32, 24)
GREEN = (34, 106, 46)

_FONTS = [
    "/usr/share/fonts/truetype/cmu/cmunss.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONTS_BOLD = [
    "/usr/share/fonts/truetype/cmu/cmunsx.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size, bold=False):
    for path in (_FONTS_BOLD if bold else _FONTS):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_scene_model(obstacles, workdir):
    """MuJoCo model holding the xArm7 and the benchmark's obstacle boxes.

    The boxes use the recorded positions and heights, so the rendered scene and
    the collision-checked scene are the same scene.
    """
    urdf_dir = os.path.join(get_package_share_directory("xarm7_description"), "urdf")
    urdf_str = open(os.path.join(urdf_dir, "xarm7.urdf")).read()
    os.symlink(os.path.join(urdf_dir, "meshes"), os.path.join(workdir, "meshes"))
    urdf_path = os.path.join(workdir, "xarm7.urdf")
    with open(urdf_path, "w") as fh:
        fh.write(urdf_str)
    raw = mujoco.MjModel.from_xml_path(urdf_path)
    mjcf_path = os.path.join(workdir, "xarm7.xml")
    mujoco.mj_saveLastXML(mjcf_path, raw)

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    visual = ET.SubElement(root, "visual")
    visual.append(ET.Element("headlight", {"ambient": "0.45 0.45 0.45"}))
    visual.append(ET.Element("global", {"offwidth": str(PANEL_W), "offheight": str(PANEL_H)}))

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "type": "2d", "builtin": "checker", "name": "grid",
        "rgb1": "0.16 0.18 0.22", "rgb2": "0.10 0.11 0.14",
        "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid",
                                      "texrepeat": "4 4", "reflectance": "0.08"})
    ET.SubElement(worldbody, "light", {"pos": "0.6 -0.6 1.6", "dir": "-0.3 0.3 -1",
                                       "diffuse": "0.9 0.9 0.85"})
    ET.SubElement(worldbody, "light", {"pos": "-0.7 0.7 1.3", "dir": "0.4 -0.4 -1",
                                       "diffuse": "0.4 0.4 0.45"})
    ET.SubElement(worldbody, "geom", {"type": "plane", "size": "2 2 0.05",
                                      "pos": "0 0 -0.42", "material": "grid", "group": "1"})
    for i, box in enumerate(obstacles):
        sx, sy, sz = (float(v) / 2.0 for v in box["size"])
        px, py, pz = (float(v) for v in box["pos"])
        # the mounting table stays neutral; only the obstacles signal collision
        is_table = i == 0
        ET.SubElement(worldbody, "geom", {
            "name": f"obstacle_{i}", "type": "box",
            "size": f"{sx:.4f} {sy:.4f} {sz:.4f}",
            "pos": f"{px} {py} {pz}",
            "rgba": TABLE_RGBA if is_table else OK_RGBA, "group": "1",
        })
    out = os.path.join(workdir, "scene.xml")
    tree.write(out)
    return mujoco.MjModel.from_xml_path(out)


def resample(waypoints, n):
    """Interpolate a joint path to exactly n frames."""
    dense = interpolate_waypoints([np.asarray(q, dtype=float) for q in waypoints],
                                  steps_per_segment=12)
    idx = np.linspace(0, len(dense) - 1, n)
    return [dense[int(round(i))] for i in idx]


def label_panel(img, title, subtitle, blocked):
    """Stack a caption above one rendered panel."""
    canvas = Image.new("RGB", (img.width, img.height + LABEL_H), "white")
    canvas.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    draw.text((img.width // 2, 18), title, font=_font(30, bold=True),
              fill=RED if blocked else INK, anchor="mm")
    draw.text((img.width // 2, 58), subtitle, font=_font(23),
              fill=RED if blocked else GREEN, anchor="mm")
    if blocked:
        draw.rectangle([(1, LABEL_H), (img.width - 2, canvas.height - 2)],
                       outline=RED, width=5)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="JSON from the benchmark scripts")
    ap.add_argument("--out-dir", default="media")
    ap.add_argument("--name", default="moveit_demo")
    args = ap.parse_args()

    with open(args.results) as fh:
        data = json.load(fh)
    start, goal, obstacles = data["start"], data["goal"], data["obstacles"]

    by_name = {r["planner"]: r for r in data["results"]}
    local = data.get("local_planner")
    best_moveit = min(
        (r for r in data["results"] if r.get("trajectory")),
        key=lambda r: r["path_length_rad"], default=None,
    )
    if best_moveit is None or local is None or not local.get("trajectory"):
        print("error: results file lacks recorded trajectories; rerun both benchmark "
              "scripts before rendering", file=sys.stderr)
        return 1

    panels = [
        {"title": "Direct interpolation",
         "sub": f"{data['straight_line_in_collision']}/{data['straight_line_samples']} waypoints in collision",
         "path": [start, goal]},
        {"title": "RRT* (kinematic_planner)",
         "sub": f"collision-free, {local['path_length_rad']:.2f} rad",
         "path": local["trajectory"]},
        {"title": best_moveit["planner"].split("/")[-1] + " (MoveIt 2)",
         "sub": f"collision-free, {best_moveit['path_length_rad']:.2f} rad",
         "path": best_moveit["trajectory"]},
    ]

    _, collision_fn = build_stack(obstacles)
    for panel in panels:
        frames = resample(panel["path"], FRAMES)
        panel["frames"] = ([frames[0]] * HOLD_START) + frames + ([frames[-1]] * HOLD_END)
        panel["hits"] = [not collision_fn(_node(q)) for q in panel["frames"]]
        print(f"{panel['title']}: {sum(panel['hits'])}/{len(panel['frames'])} frames in collision")

    n_frames = len(panels[0]["frames"])
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, f"_frames_{args.name}")
    os.makedirs(frame_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as workdir:
        model = build_scene_model(obstacles, workdir)
        mj_data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
        cam = mujoco.MjvCamera()
        cam.lookat = [0.22, 0.0, 0.30]
        cam.distance = 1.85
        cam.azimuth = 138
        cam.elevation = -16
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0
        opt.geomgroup[1] = 1

        # index 0 is the mounting table, which is not part of the collision cue
        obstacle_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}")
                      for i in range(1, len(obstacles))]
        base_rgba = np.array([float(v) for v in OK_RGBA.split()])
        hit_rgba = np.array([float(v) for v in HIT_RGBA.split()])

        paths = []
        for f in range(n_frames):
            tiles = []
            for panel in panels:
                blocked = panel["hits"][f]
                for gid in obstacle_ids:
                    model.geom_rgba[gid] = hit_rgba if blocked else base_rgba
                mj_data.qpos[:7] = panel["frames"][f]
                mujoco.mj_forward(model, mj_data)
                renderer.update_scene(mj_data, camera=cam, scene_option=opt)
                tile = Image.fromarray(renderer.render())
                tiles.append(label_panel(tile, panel["title"], panel["sub"], blocked))
            sheet = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), "white")
            xoff = 0
            for t in tiles:
                sheet.paste(t, (xoff, 0))
                xoff += t.width
            out = os.path.join(frame_dir, f"frame_{f:04d}.png")
            sheet.save(out)
            paths.append(out)
        renderer.close()

    mp4 = os.path.join(out_dir, f"{args.name}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(frame_dir, "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", mp4], check=True)
    palette = os.path.join(out_dir, f"_{args.name}_palette.png")
    gif = os.path.join(out_dir, f"{args.name}.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
                    "-vf", "fps=16,scale=1080:-1:flags=lanczos,palettegen", palette], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i", palette,
                    "-filter_complex",
                    "fps=16,scale=1080:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none",
                    gif], check=True)
    os.remove(palette)

    poster = os.path.join(out_dir, f"{args.name}_poster.png")
    imageio.imwrite(poster, imageio.imread(paths[HOLD_START + FRAMES // 2]))
    for p in paths:
        os.remove(p)
    os.rmdir(frame_dir)
    for f in (mp4, gif, poster):
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
