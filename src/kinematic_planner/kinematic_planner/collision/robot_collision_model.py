#!/usr/bin/env python3

"""In-process robot collision geometry, built once from a URDF string.

Replaces the SolidPrimitive/RigidBodyGeom-message path for planning-time
collision checking. The RigidBodyGeom message path carries one geometry per
link and ignores the `<collision><origin>` transform entirely, both audited
P0 bugs. `build_link_collision_shapes` keeps every `<collision>` element per
link, applies its local origin transform, and supports `<mesh>` geometry via
`trimesh` and `fcl.BVHModel`. `RigidBodyGeom` and `robot_geom_publisher`
remain in place for RViz visualization only.
"""

import os
from dataclasses import dataclass
from typing import Dict, List
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import spatialmath as sm
import trimesh
from ament_index_python.packages import get_package_share_directory


@dataclass
class CollisionShape:
    fcl_geometry: "fcl.CollisionGeometry"
    local_origin: np.ndarray  # 4x4 homogeneous transform, link frame to shape frame


def resolve_ros_package_uri(uri: str) -> str:
    """Resolves a `package://<pkg>/<rel_path>` URI to an absolute filesystem
    path via the installed package's share directory. Non-package URIs pass
    through unchanged."""
    if not uri.startswith("package://"):
        return uri
    rest = uri[len("package://"):]
    pkg_name, _, rel_path = rest.partition("/")
    pkg_share = get_package_share_directory(pkg_name)
    return os.path.join(pkg_share, rel_path)


def _mesh_to_fcl_bvh(filepath: str, scale) -> fcl.BVHModel:
    mesh = trimesh.load(filepath, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    model = fcl.BVHModel()
    model.beginModel(len(verts), len(faces))
    model.addSubModel(verts, faces)
    model.endModel()
    return model


def _parse_origin(origin_el) -> sm.SE3:
    if origin_el is None:
        return sm.SE3()
    xyz = [float(x) for x in origin_el.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in origin_el.get("rpy", "0 0 0").split()]
    return sm.SE3(xyz) * sm.SE3.RPY(rpy, order="xyz", unit="rad")


def build_link_collision_shapes(urdf_root: ET.Element) -> Dict[str, List[CollisionShape]]:
    """Parses every `<collision>` element of every `<link>` in the given
    URDF root, preserving each element's local origin transform and
    supporting box, sphere, cylinder, and mesh geometry."""
    shapes_by_link: Dict[str, List[CollisionShape]] = {}
    for link in urdf_root.findall("link"):
        link_name = link.get("name", "")
        shapes: List[CollisionShape] = []
        for collision_el in link.findall("collision"):
            geometry_el = collision_el.find("geometry")
            if geometry_el is None:
                continue
            local_origin = _parse_origin(collision_el.find("origin")).A

            box_el = geometry_el.find("box")
            sphere_el = geometry_el.find("sphere")
            cylinder_el = geometry_el.find("cylinder")
            mesh_el = geometry_el.find("mesh")

            if box_el is not None:
                size = [float(x) for x in box_el.get("size").split()]
                geom = fcl.Box(*size)
            elif sphere_el is not None:
                radius = float(sphere_el.get("radius"))
                geom = fcl.Sphere(radius)
            elif cylinder_el is not None:
                radius = float(cylinder_el.get("radius"))
                length = float(cylinder_el.get("length"))
                geom = fcl.Cylinder(radius, length)
            elif mesh_el is not None:
                mesh_path = resolve_ros_package_uri(mesh_el.get("filename", ""))
                scale = [float(x) for x in mesh_el.get("scale", "1 1 1").split()]
                geom = _mesh_to_fcl_bvh(mesh_path, scale)
            else:
                continue

            shapes.append(CollisionShape(fcl_geometry=geom, local_origin=local_origin))
        if shapes:
            shapes_by_link[link_name] = shapes
    return shapes_by_link


def link_shapes_to_fcl_objects(shapes: List[CollisionShape],
                                link_world_transform: np.ndarray) -> List[fcl.CollisionObject]:
    """Instantiates one `fcl.CollisionObject` per shape at its world pose,
    composing the link's forward-kinematics transform with each shape's
    local origin transform."""
    objects = []
    for shape in shapes:
        world_transform = link_world_transform @ shape.local_origin
        rotation = world_transform[:3, :3]
        translation = world_transform[:3, 3]
        objects.append(fcl.CollisionObject(shape.fcl_geometry, fcl.Transform(rotation, translation)))
    return objects
