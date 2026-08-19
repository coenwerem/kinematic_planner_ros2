# Provenance

`urdf/xarm7.urdf` and `urdf/meshes/` are copied from
[`frogger`](https://github.com/albertli24/frogger)'s `models/xarm7/` directory
(`frogger` is MIT-licensed, Copyright (c) 2023 Albert H. Li; see the
`frogger` repository's `LICENSE`), itself derived from UFactory's xArm7 robot
description. No modifications were made to the mesh files. The URDF gained
one addition: a `<mujoco><compiler .../></mujoco>` element, carrying three
MuJoCo-only overrides. ROS, FCL, and `xml.etree` skip unrecognized XML elements without error,
so the `<mujoco>` element has no effect on ROS, FCL, or `xml.etree`. The three overrides are
`balanceinertia="true"` (MuJoCo's URDF importer rejects `link4`'s inertial
block otherwise, an `A + B >= C` triangle-inequality check),
`discardvisual="false"` (MuJoCo drops `<visual><mesh>` references by
default when importing URDF), and `meshdir="meshes/visual"` (MuJoCo
resolves each kept mesh reference by basename under one shared meshdir,
not the mesh's own relative path in the URDF).

The xArm7 URDF and meshes are used here only for kinematic collision-free
motion planning (convex-primitive collision geometry, no dexterous hand
attached). No grasping, force control, or hardware deployment is
implemented or implied.
