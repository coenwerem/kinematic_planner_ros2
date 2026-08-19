import os
import tempfile
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import pytest
import trimesh

from kinematic_planner.collision.robot_collision_model import (
    CollisionShape,
    build_link_collision_shapes,
    link_shapes_to_fcl_objects,
    resolve_ros_package_uri,
)


ONE_COLLISION_URDF = """
<robot name="test_robot">
  <link name="link_a">
    <collision>
      <origin xyz="1 0 0" rpy="0 0 0"/>
      <geometry><box size="0.2 0.2 0.2"/></geometry>
    </collision>
  </link>
</robot>
"""

TWO_COLLISIONS_URDF = """
<robot name="test_robot">
  <link name="link_b">
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
    <collision>
      <origin xyz="0.3 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.1"/></geometry>
    </collision>
  </link>
</robot>
"""


def test_collision_origin_offset_is_applied_not_ignored():
    root = ET.fromstring(ONE_COLLISION_URDF)
    shapes_by_link = build_link_collision_shapes(root)
    shapes = shapes_by_link["link_a"]
    assert len(shapes) == 1
    assert isinstance(shapes[0].fcl_geometry, fcl.Box)
    translation = shapes[0].local_origin[:3, 3]
    assert np.allclose(translation, [1.0, 0.0, 0.0])


def test_multiple_collision_elements_per_link_are_all_kept():
    root = ET.fromstring(TWO_COLLISIONS_URDF)
    shapes_by_link = build_link_collision_shapes(root)
    shapes = shapes_by_link["link_b"]
    assert len(shapes) == 2
    assert isinstance(shapes[0].fcl_geometry, fcl.Sphere)
    assert isinstance(shapes[1].fcl_geometry, fcl.Cylinder)
    assert np.allclose(shapes[1].local_origin[:3, 3], [0.3, 0.0, 0.0])


def test_mesh_collision_geometry_detects_overlap_and_separation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mesh_path = os.path.join(tmp_dir, "unit_box.obj")
        trimesh.primitives.Box(extents=[0.2, 0.2, 0.2]).export(mesh_path)
        urdf = f"""
        <robot name="test_robot">
          <link name="link_mesh">
            <collision>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry><mesh filename="{mesh_path}" scale="1 1 1"/></geometry>
            </collision>
          </link>
        </robot>
        """
        root = ET.fromstring(urdf)
        shapes = build_link_collision_shapes(root)["link_mesh"]
        assert len(shapes) == 1

        overlapping_box = fcl.CollisionObject(fcl.Box(0.2, 0.2, 0.2), fcl.Transform())
        far_box = fcl.CollisionObject(fcl.Box(0.2, 0.2, 0.2), fcl.Transform(np.eye(3), [10.0, 0.0, 0.0]))

        mesh_objs = link_shapes_to_fcl_objects(shapes, np.eye(4))
        assert len(mesh_objs) == 1
        mesh_obj = mesh_objs[0]

        req, res = fcl.CollisionRequest(), fcl.CollisionResult()
        assert fcl.collide(mesh_obj, overlapping_box, req, res) > 0

        req2, res2 = fcl.CollisionRequest(), fcl.CollisionResult()
        assert fcl.collide(mesh_obj, far_box, req2, res2) == 0


def test_resolve_ros_package_uri_joins_share_directory(monkeypatch):
    import kinematic_planner.collision.robot_collision_model as mod

    monkeypatch.setattr(mod, "get_package_share_directory", lambda name: f"/opt/ros_pkgs/{name}")
    resolved = resolve_ros_package_uri("package://my_robot_description/meshes/link1.stl")
    assert resolved == "/opt/ros_pkgs/my_robot_description/meshes/link1.stl"


def test_resolve_ros_package_uri_passes_through_non_package_paths():
    assert resolve_ros_package_uri("/absolute/path/mesh.stl") == "/absolute/path/mesh.stl"


def test_link_shapes_to_fcl_objects_composes_link_and_local_transforms():
    shape = CollisionShape(fcl_geometry=fcl.Box(0.1, 0.1, 0.1), local_origin=np.eye(4))
    shape.local_origin[:3, 3] = [1.0, 0.0, 0.0]
    link_world_transform = np.eye(4)
    link_world_transform[:3, 3] = [0.0, 2.0, 0.0]
    objs = link_shapes_to_fcl_objects([shape], link_world_transform)
    world_translation = objs[0].getTranslation()
    assert np.allclose(world_translation, [1.0, 2.0, 0.0])
