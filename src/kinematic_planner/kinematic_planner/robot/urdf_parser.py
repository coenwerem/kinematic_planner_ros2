#!/usr/bin/env python3

"""
     There are also methods for performing simple kinematic computations on chains, viz. Jacobian computation, forward kinematics, singularity detection, and inverse kinematics.

References:
1. K. M. Lynch and F. C. Park, Modern Robotics. Cambridge University Press, 2019.
2. “URDF Primer - MATLAB & Simulink.” [Online]. Available: https://www.mathworks.com/help/sm/ug/urdf-model-import.html
3. “urdf/XML - ROS Wiki.” Available: https://wiki.ros.org/urdf/XML
4. xml.etree docs.


Author: Clinton Enwerem
Developed for the course ENEE467, Robotics Projects Laboratory, Fall 2025, University of Maryland, College Park, MD.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Union
import spatialmath as sm
import numpy as np
from numpy.typing import NDArray
import trimesh

class Origin:
    def __init__(self, rot_euler: List, translation: List):
        if rot_euler and translation:
            # URDF is translate, then rotate
            T = sm.SE3.Trans(translation) * sm.SE3.RPY(rot_euler, order='zyx')
        else:
            T = sm.SE3()
        self.T = T  

    def __mul__(self, other):
        return self.T * other.T if isinstance(other, Origin) else self.T * other

    def __rmul__(self, other):
        return other * self.T

class Geometry:
    def __init__(self,
                 geom_type: str,
                 geom_vals: Tuple = None,
                 mesh_dir: str = "",
                 num_points: int = 0,
                 fit_type: str = "cylinder") -> None:
        self.geom_type = geom_type
        self.geometry = None

        try:
            if geom_type == "box":
                extents = np.array(geom_vals) if geom_vals is not None else np.array([0.1,0.1,0.1])
                self.geometry = trimesh.primitives.Box(extents=extents)

            elif geom_type == "cylinder":
                radius = geom_vals[0] if geom_vals else 0.05
                height = geom_vals[1] if geom_vals and len(geom_vals) > 1 else 0.1
                self.radius = radius
                self.height = height
                self.geometry = trimesh.primitives.Cylinder(radius=self.radius, height=self.height)

            elif geom_type == "sphere":
                self.radius = geom_vals[0] if geom_vals else 0.05
                self.center = geom_vals[1] if geom_vals and len(geom_vals) > 1 else (0,0,0)
                self.geometry = trimesh.primitives.Sphere(center=self.center, radius=self.radius)

            elif geom_type == "mesh":
                # create a reasonable default box if mesh parsing fails
                if not hasattr(Geometry, "_mesh_warning_printed"):
                    print("Mesh geom parsing broken due to pycollada deprecation. Using default proxy box. \n"
                          "Do not use this for collision geometry in practice!")
                    Geometry._mesh_warning_printed = True
                # use provided scale to make the fallback more sensible
                if geom_vals and len(geom_vals) >= 3:
                    fallback_extents = np.array(geom_vals)
                else:
                    fallback_extents = np.array([0.25, 0.1, 0.1])
                self.geometry = trimesh.primitives.Box(extents=fallback_extents)

            else:
                raise ValueError(f"Geometry type {geom_type} not recognized.")

        except Exception as e:
            print(f"Error creating geometry of type {geom_type}: {e}\nUsing tiny fallback box.")
            self.geometry = trimesh.primitives.Box(extents=[1e-3, 1e-3, 1e-3])
        try:
            self.inertia = self.geometry.moment_inertia() if self.geometry is not None else None
        except Exception:
            self.inertia = None

class Link:
    """A class representing a link in a robotic manipulator.

    Attributes:
        name (str): The name of the link.
        visual (Visual): The visual representation of the link.
        collision (Collision): The collision representation of the link.
        inertial (Inertial): The inertial representation of the link.
        origin (Origin): The origin of the link's frame.
    Raises:
        ValueError: If the link is not properly initialized.
    """
    class Visual:
        """A class representing the visual properties of a robotic link.
        
        
        Attributes:
            name (str): The name of the visual element.
            geometry (Geometry): The geometry of the visual element.
            material (Material): The material of the visual element.
        """
        class Material:
            """A class representing the material properties of a visual element.

            Attributes:
                name (str): The name of the material.
                color (Tuple): The RGBA color of the material.
            """
            def __init__(self, 
                         name: str, 
                         color: Tuple) -> None:
                self.name = name
                self.color = color

        def __init__(self,
                     origin: Origin,
                     geometry: Geometry,
                     material: Material) -> None:
            self.origin = origin
            self.geometry = geometry
            self.material = material
    
    class Collision:
        """A class representing the collision properties of a robotic link.

        Attributes:
            origin (Origin): The origin of the collision frame.
            geometry (Geometry): The geometry of the collision element.
        """
        def __init__(self,
                     origin: Origin,
                     geometry: Geometry
                     ) -> None:
            self.origin = origin
            self.geometry = geometry

    class Inertial:
        """A class representing the inertial properties of a robotic link.

        Attributes:
            mass (float): The mass of the link.
            origin (Origin): The origin of the link's inertial frame.
            inertia (np.ndarray): The inertia matrix of the link.
            check (bool): A flag to indicate if the inertia matrix should be checked for validity.
        """
        def __init__(self,
                     mass: float,
                     origin: Origin,
                     collision_geom: Geometry,
                     check: bool=True
                     ) -> None:
            self.mass = mass
            self.origin = origin
            self.check = check
            self.collision_geom = collision_geom
            if self.collision_geom.geom_type == "cylinder":
                self.inertia = self.collision_geom.inertia

    def __init__(self, 
                 name: str, 
                 visual: Visual, 
                 collision: Collision,
                 inertial: Inertial):
        """Initialize a Link object.

        Args:
            name (str): The name of the link.
            visual (Visual): The visual representation of the link.
            collision (Collision): The collision representation of the link.
            inertial (Inertial): The inertial representation of the link.

        Raises:
            ValueError: If the link is not properly initialized.
        """
        if not name:
            print("A link must have a name to distinguish its frame \
                  from other frames in the TF tree!")
            raise ValueError
        self._name = name
        self._visual = visual
        self._collision = collision
        self._inertial = inertial
        self._origin = self._collision.origin if self._collision else None

class Joint:
    """
    A class representing a joint connecting two links in a robotic manipulator.

    Attributes:
        name (str): The name of the joint.
        joint_type (str): The type of the joint (e.g., "revolute", "prismatic").
        parent (Link): The parent link of the joint.
        child (Link): The child link of the joint.
        origin (sm.SE3): The origin of the joint in the parent link's frame.
        axis (List): The axis of motion for the joint.
    """
    def __init__(self, 
                 name: str, 
                 joint_type: str, 
                 parent: Link, 
                 child: Link, 
                 origin: Origin, 
                 axis: List,
                 limits: Tuple[float, float]):
        self.name = name
        self.joint_type = joint_type
        self.parent = parent
        self.child = child
        self.origin = origin
        self.axis = axis
        self.limits = limits

    def _compute_motion(self, q: float) -> sm.SE3:
        """Computes a joint motion based on joint type

        Args:
            q (float): The joint position (in radians) or displacement (in meters).

        Returns:
            sm.SE3: The SE3 transformation representing the joint motion.
        """
        if self.joint_type == "revolute":
            return sm.SE3.Rt(sm.SO3.AngleAxis(q, np.array(self.axis)), np.zeros(3))
        elif self.joint_type == "prismatic":
            return sm.SE3.Rt(sm.SO3(), q * np.array(self.axis))
        else:  # the joint is fixed
            return sm.SE3()

class Robot:
    VALID_DESC_TYPES:List[str] = ['xml', 'xacro', 'urdf']

    class Configuration:
        def __init__(self,
                    joints: List[Joint],
                    joint_values: List[float]) -> None:
            if len(joints) != len(joint_values):
                print("Joint names and values must be of the same length!")
                raise ValueError
            self.joints = joints
            self.joint_values = joint_values
            self.joint_dict: Dict[str, float] = {joint.name: val for joint, val in zip(self.joints, self.joint_values)}
            self.num_joints = len(self.joints)

        def q0(self):
            """Return a zero-valued Configuration with same joints."""
            zero_values = [0.0] * self.num_joints
            return Robot.Configuration(self.joints, zero_values)
        
        def copy(self):
            """Return an independent copy of this Configuration."""
            # copy lists so later modifications won't mutate the original
            return Robot.Configuration(list(self.joints), list(self.joint_values))

        @property
        def is_singular(self) -> bool:
            return
        
        @classmethod
        def zeros_for_joints(cls, joints: List[Joint]):
            """Construct a zero-valued configuration given a joint list."""
            return cls(joints, [0.0] * len(joints))

    
    def __init__(self, desc_fp: str):
        """Initialize a Robot object by specifying the path to the XML (.urdf, .xacro) 
        file that contains its description.
        
        Args:
            desc_fp <str>: An absolute or relative file path to the XML description file.

        Returns a Robot instance.

        """
        self._file_ext = None
        self._desc_fp = None
        self._desc_tag = None
        self._tree: ET.ElementTree = None
        self.links: List[Link] = []
        self.joints: List[Joint] = []

        if not isinstance(desc_fp, str):
            raise ValueError("File path must be a string.")

        # normalize path
        candidate = os.path.abspath(desc_fp)
        if not os.path.exists(candidate):
            # try relative to this file's urdf folder 
            base_dir = os.path.dirname(os.path.abspath(__file__))
            alt = os.path.abspath(os.path.join(base_dir, '..', 'urdf', desc_fp))
            if os.path.exists(alt):
                candidate = alt
            else:
                raise FileNotFoundError(f"URDF file not found: {desc_fp}")

        self._desc_fp = candidate
        self._file_ext = self._desc_fp.split('.')[-1].lower()
        if self._file_ext not in self.VALID_DESC_TYPES:
            raise TypeError("Input to URDF object is not a supported XML format (.urdf/.xacro/.xml)")

        # parse and build
        self._tree = self._parse_urdf(self._desc_fp)
        root = self._tree.getroot()
        self._desc_tag = f"{root.tag}:{root.attrib.get('name', 'unnamed')}"
        print(f"Got description file of type {self._file_ext.upper()} with tag:"+f"\033[92m {self._desc_tag}\033[0m.")
        try:
            self.links, self.joints = self._build_urdf()
        except Exception as e:
            print(f"Error building URDF: {e}")
            raise

    def _build_urdf(self):
        """
        Populates the Robot instance with its links, joints, transmissions, visuals, and collisions.
        """
        if self._tree is None:
            print("URDF instance has no XML tree. Cannot build URDF.")
            raise ValueError
        root = self._tree.getroot() # get root object of XML tree
        links: List[Link] = []
        joints: List[Joint] = []
        for link in root.findall('link'):   
            link_name = link.attrib.get('name', 'default_link')
            if link_name is None:
                print("A link must have a name to distinguish its frame from other frames in the TF tree!")
                continue
            visual_elem = link.find('visual')
            if visual_elem is not None:
                origin_elem = visual_elem.find('origin')
                if origin_elem is not None:
                    xyz = [float(x) for x in origin_elem.attrib.get('xyz', '0 0 0').split()]
                    rpy = [float(r) for r in origin_elem.attrib.get('rpy', '0 0 0').split()]
                    origin = Origin(rot_euler=rpy, translation=xyz)
                else:
                    origin = Origin(rot_euler=[0,0,0], translation=[0,0,0])
                geometry_elem = visual_elem.find('geometry')
                if geometry_elem is not None:
                    if geometry_elem.find('box') is not None:
                        size_str = geometry_elem.find('box').attrib['size']
                        size = tuple(float(s) for s in size_str.split())
                        geometry = Geometry(geom_type='box', geom_vals=size, mesh_dir='', num_points=0)
                    elif geometry_elem.find('cylinder') is not None:
                        radius = float(geometry_elem.find('cylinder').attrib['radius'])
                        length = float(geometry_elem.find('cylinder').attrib['length'])
                        geometry = Geometry(geom_type='cylinder', geom_vals=(radius, length), mesh_dir='', num_points=0)
                    elif geometry_elem.find('sphere') is not None:
                        radius = float(geometry_elem.find('sphere').attrib['radius'])
                        geometry = Geometry(geom_type='sphere', geom_vals=(radius,), mesh_dir='', num_points=0)
                    elif geometry_elem.find('mesh') is not None:
                        filename = geometry_elem.find('mesh').attrib['filename']
                        scale_str = geometry_elem.find('mesh').attrib.get('scale', '1 1 1')
                        scale = tuple(float(s) for s in scale_str.split())
                        geometry = Geometry(geom_type='mesh', geom_vals=scale, mesh_dir=filename, num_points=1000)
                    else:
                        print(f"Unsupported geometry type in visual element of link {link_name}.")
                        continue
                else:
                    print(f"No geometry element found in visual of link {link_name}.")
                    continue
                material_elem = visual_elem.find('material')
                if material_elem is not None:
                    material_name = material_elem.attrib.get('name', 'default')
                    color_elem = material_elem.find('color')
                    if color_elem is not None:
                        rgba_str = color_elem.attrib.get('rgba', '0.8 0.8 0.8 1.0')
                        rgba = tuple(float(c) for c in rgba_str.split())
                    else:
                        rgba = (0.8, 0.8, 0.8, 1.0)
                    material = Link.Visual.Material(name=material_name, color=rgba)
                else:
                    material = Link.Visual.Material(name='default', color=(0.8, 0.8, 0.8, 1.0))
                visual = Link.Visual(origin=origin, geometry=geometry, material=material)
            else:
                visual = None
            collision_elem = link.find('collision')
            if collision_elem is not None:  
                origin_elem = collision_elem.find('origin')
                if origin_elem is not None:
                    xyz = [float(x) for x in origin_elem.attrib.get('xyz', '0 0 0').split()]
                    rpy = [float(r) for r in origin_elem.attrib.get('rpy', '0 0 0').split()]
                    origin = Origin(rot_euler=rpy, translation=xyz)
                else:
                    origin = Origin(rot_euler=[0,0,0], translation=[0,0,0])
                geometry_elem = collision_elem.find('geometry')
                if geometry_elem is not None:
                    if geometry_elem.find('box') is not None:
                        size_str = geometry_elem.find('box').attrib['size']
                        size = tuple(float(s) for s in size_str.split())
                        geometry = Geometry(geom_type='box', geom_vals=size, mesh_dir='', num_points=0)
                    elif geometry_elem.find('cylinder') is not None:
                        radius = float(geometry_elem.find('cylinder').attrib['radius'])
                        length = float(geometry_elem.find('cylinder').attrib['length'])
                        geometry = Geometry(geom_type='cylinder', geom_vals=(radius, length), mesh_dir='', num_points=0)
                    elif geometry_elem.find('sphere') is not None:
                        radius = float(geometry_elem.find('sphere').attrib['radius'])
                        geometry = Geometry(geom_type='sphere', geom_vals=(radius,), mesh_dir='', num_points=0)
                    elif geometry_elem.find('mesh') is not None:
                        filename = geometry_elem.find('mesh').attrib['filename']
                        scale_str = geometry_elem.find('mesh').attrib.get('scale', '1 1 1')
                        scale = tuple(float(s) for s in scale_str.split())
                        geometry = Geometry(geom_type='mesh', geom_vals=scale, mesh_dir=filename, num_points=1000)
                    else:
                        print(f"Unsupported geometry type in collision element of link {link_name}.")
                        continue
                else:
                    print(f"No geometry element found in collision of link {link_name}.")
                    continue
                collision = Link.Collision(origin=origin, geometry=geometry)
            else:
                collision = None
            inertial_elem = link.find('inertial')
            if inertial_elem is not None:
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.attrib.get('value', '1.0'))
                else:
                    mass = 1.0
                origin_elem = inertial_elem.find('origin')
                if origin_elem is not None:
                    xyz = [float(x) for x in origin_elem.attrib.get('xyz', '0 0 0').split()]
                    rpy = [float(r) for r in origin_elem.attrib.get('rpy', '0 0 0').split()]
                    origin = Origin(rot_euler=rpy, translation=xyz)
                else:
                    origin = Origin(rot_euler=[0,0,0], translation=[0,0,0])
                if collision is not None:
                    collision_geom = collision.geometry
                else:
                    collision_geom = Geometry(geom_type='box', geom_vals=(0.1, 0.1, 0.1), mesh_dir='', num_points=0)
                inertial = Link.Inertial(mass=mass, origin=origin, collision_geom=collision_geom)
            else:
                inertial = None
            link_obj = Link(name=link_name, visual=visual, collision=collision, inertial=inertial)
            links.append(link_obj)

        for joint in root.findall('joint'): 
            joint_name = joint.attrib['name']
            joint_type = joint.attrib['type']
            parent_elem = joint.find('parent')
            child_elem = joint.find('child')
            limits_elem = joint.find('limit')
            if parent_elem is not None and child_elem is not None:
                parent_link_name = parent_elem.attrib['link']
                child_link_name = child_elem.attrib['link']
                parent_link = next((l for l in links if l._name == parent_link_name), None)
                child_link = next((l for l in links if l._name == child_link_name), None)
                if parent_link is None or child_link is None:
                    print(f"Parent or child link for joint {joint_name} not found.")
                    continue
            else:
                print(f"Parent or child element missing in joint {joint_name}.")
                continue
            origin_elem = joint.find('origin')
            if origin_elem is not None:
                xyz = [float(x) for x in origin_elem.attrib.get('xyz', '0 0 0').split()]
                rpy = [float(r) for r in origin_elem.attrib.get('rpy', '0 0 0').split()]
                origin = Origin(rot_euler=rpy, translation=xyz)
            else:
                origin = Origin(rot_euler=[0,0,0], translation=[0,0,0])
            axis_elem = joint.find('axis')
            if axis_elem is not None:
                axis_str = axis_elem.attrib.get('xyz', '0 0 1')
                axis = [float(a) for a in axis_str.split()]
            else:
                axis = [0, 0, 1]
            if limits_elem is not None:
                lower = float(limits_elem.attrib.get('lower', -np.pi))
                upper = float(limits_elem.attrib.get('upper', np.pi))
                limits = (lower, upper)
            else:
                limits = (-np.pi, np.pi)
            joint_obj = Joint(name=joint_name, joint_type=joint_type, parent=parent_link, child=child_link, origin=origin, axis=axis, limits=limits)
            joints.append(joint_obj)
        return links, joints

    def _compute_T(self, from_frame: str, to_frame: str, config: Configuration) -> sm.SE3:
        """Compute transform from frame of link1 to frame of link2.
        
        Args:
            from_frame: name of starting link
            to_frame: name of target link
            q: dict mapping joint names -> joint positions (rad or m)
        
        Returns:
            sm.SE3: transform from link1 to link2
        """
        if config is None:
            config = Robot.Configuration.zeros_for_joints(self.joints)
        # validate
        self._set_config(config)
        q = config.joint_dict

        path1 = self._path_to_root(from_frame)
        path2 = self._path_to_root(to_frame)

        # find lowest common ancestor (LCA)
        try:
            lca = next(l for l in path1 if l in path2)
        except StopIteration:
            raise ValueError(f"No common ancestor between {from_frame} and {to_frame}")

        T = sm.SE3()
        current = from_frame
        while current != lca:
            joint = self._get_joint_to_parent(current)
            # if joint name not present in config, assume q=0
            qval = q.get(joint.name, 0.0)
            T = (joint._compute_motion(qval).inv() * joint.origin.inv()) * T
            current = joint.parent._name

        stack: List[Joint] = []
        current = to_frame
        while current != lca:
            joint = self._get_joint_to_parent(current)
            stack.append(joint)
            current = joint.parent._name

        for joint in reversed(stack):
            qval = q.get(joint.name, 0.0)
            T = T * (joint.origin * joint._compute_motion(qval))

        T = np.array(T, dtype=float)

        if T.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 matrix, got shape {T.shape}")
        if not np.isfinite(T).all():
            raise ValueError("Matrix contains NaN or inf")
        try:
            return sm.SE3(T)
        except Exception as e:
            print(f"Error occurred while creating SE3 object: {e}")
            raise

    def _path_to_root(self, link_name: str) -> List[str]:
        """Returns a list of link names from the specified link up to the root link."""
        path = []
        current = link_name
        while current is not None:
            path.append(current)
            try:
                parent = self._get_parent(current)
                current = parent._name
            except ValueError:
                # no parent found, so we've reached the root
                break
        return path
    
    def _get_joint_to_parent(self, link_name: str) -> Joint:
        """
        Returns the joint connecting the specified link to its parent.
        """
        for joint in self.joints:
            if joint.child._name == link_name:
                return joint
        raise ValueError(f"No joint found for link '{link_name}'.")

    def _get_link(self, link_name: str) -> Link:
        """
        Returns the link object corresponding to the specified link name.
        """
        try:
            links = self.links
            for link in links:
                if link._name == link_name:
                    link1 = link 
                    return link1
                else:
                    print(f"Link object for link with name {link_name} not found. \n"
                    "Check the name of the link for typos, and ensure it's a valid\n\
                    link name in the robot's URDF!")
        except Exception as e:
            print(e)

    def _get_parent(self, link_name: str) -> Link:
        """
        Returns the parent link of the specified link.
        """
        for idx, link in enumerate(self.links):
            if link_name != self.base_link._name and self._is_parent(link._name, link_name):
                return self.links[idx]
        raise ValueError(f"No parent found for link '{link_name}'.")
    
    def _is_parent(self, parent_link_name:str, child_link_name:str) -> bool:
        """
        Checks if the specified link is a parent of another link.
        """
        for joint in self.joints:
            if joint.parent._name == parent_link_name and joint.child._name == child_link_name:
                return True
        return False

    def _compute_fk(self, config: Configuration, start:Union[str, None]=None, end: Union[str, None]=None,  pretty_print: bool=True, verbose: bool=False) -> sm.SE3:
            """
            Computes the forward kinematics transformation matrix from a specified start link to an end link for a given robot configuration.


            Args:
                start (str): The name of the starting link in the kinematic chain. If not provided, defaults to the robot's base link.
                end (str): The name of the end link (typically the end-effector). If not provided, defaults to the robot's end-effector link.
                q (Configuration): The robot's joint configuration at which to compute the forward kinematics.
                pretty_print (bool, optional): If True, prints the transformation matrix in a colored, formatted style. If False, prints the raw matrix. Defaults to True.
            Returns:
                sm.SE3: The SE3 transformation matrix representing the pose of the end link relative to the start link at the given configuration.
            Raises:
                Exception: If the transformation cannot be computed, an error message is printed.
            
            """
            try:
                if not start or start is None:
                    base_link: Link = self.base_link
                    start = base_link._name
                if not end or end is None:
                    end = self.ee_link._name
                
                T_FK = self._compute_T(from_frame=start, to_frame=end, config=config)
                if verbose:
                    if pretty_print:
                        self._print_matrix_colored(
                            np.round(T_FK.A, 4)
                            )
                    else:
                        print(
                            np.round(T_FK.A, 4)
                            )
                return T_FK
            except Exception as e:
                print(f"\nCould not compute transformation from {start} link to {end} link: {e}")

    def _compute_jacobian(self, config: Configuration) -> NDArray:
        """
        Computes the chain's (space) Jacobian at the specified configuration.
        """
        config_dict = config.joint_dict
        T_ee: sm.SE3 = self._compute_fk(config=config)
        n = len(config_dict.values())
        J = np.zeros((6, n))  # start with a Numpy array of zeros

        for i, joint in enumerate(self.actuated_joints):
            # get transform to the joint's child frame
            T_0_i_prev = self._compute_fk(end=joint.child._name, config=config) # link i-1 is the child of joint i
            z_i_prev = T_0_i_prev.R @ np.array(joint.axis) 
            p_i = T_0_i_prev.t
            p_ee = T_ee.t

            if joint.joint_type == "revolute":
                Jv = np.cross(z_i_prev, (p_ee - p_i))
                Jw = z_i_prev
            elif joint.joint_type == "prismatic":
                Jv = z_i_prev
                Jw = np.zeros(3)
            else:  # fixed joint
                Jv = np.zeros(3)
                Jw = np.zeros(3)

            # Fill Jacobian
            J[:3, i] = Jv
            J[3:, i] = Jw

        return J
    
    def _compute_ik(self, 
                x_d: sm.SE3, 
                init_guess: Configuration, 
                eps: float = 1e-4, 
                gamma: float = 1.0, 
                damp_factor: float = 0.0, 
                method: str = 'jacinv', 
                max_iters: int = 100) -> Configuration:
        """
        Iterative IK using Jacobian pseudoinverse or DLS.
        """
        config_k = init_guess.copy()

        for _ in range(max_iters):
            # grab current EE pose
            T_ee = self._compute_fk(config=config_k)

            # compute task-space error
            # translation error
            dt = np.asarray(x_d.t) - np.asarray(T_ee.t)

            # rotation error
            dr = self.compute_tsp_rot_error(x_d, T_ee)
            error = np.hstack((dt, dr)).reshape((-1, 1))  # 6x1 column

            # check task-space convergence
            if np.linalg.norm(error) < eps:
                return config_k

            # Jacobian at current q
            J = self._compute_jacobian(config_k)
            if J is None:
                raise RuntimeError("Jacobian computation failed")

            # check joint limits before applying update
            if not self.check_limits(config=config_k):
                print("Joint limits violated for config_k. Clamping values.")
                config_k = self.clamp_limits(config_k, scale=0.8)
                print(self.check_inv(J), self.check_limits(config=config_k))
            # ensure current q is within limits
            # print(self.check_inv(J), self.check_limits(config=config_k))
            if self.check_limits(config=config_k):
                # ensure current q is non-singular
                if self.check_inv(J):
                    # choose update law
                    if method == "dls" and damp_factor > 0.0:
                        JJt = J @ J.T
                        J_pinv = J.T @ np.linalg.inv(JJt + (damp_factor**2) * np.eye(JJt.shape[0]))
                    else:
                        J_pinv = np.linalg.pinv(J)

                    # update iterate
                    delta_q = (J_pinv @ error).flatten() * float(gamma)
                    q_next = np.asarray(config_k.joint_values, dtype=float) + delta_q

                    config_k = Robot.Configuration(self.actuated_joints, q_next.tolist())
                else:
                    # fallback to DLS with small damping
                    damp_eps = 1e-3
                    JJt = J @ J.T
                    J_pinv = J.T @ np.linalg.inv(JJt + (damp_eps**2) * np.eye(JJt.shape[0]))
                    delta_q = (J_pinv @ error).flatten() * float(gamma)
                    q_next = np.asarray(config_k.joint_values, dtype=float) + delta_q
                    config_k = Robot.Configuration(self.actuated_joints, q_next.tolist())

            print(f"Iteration {_}: q_next = {q_next}, error = {np.linalg.norm(error)}")

        # if we exit the loop without convergence
        raise RuntimeError("IK did not converge within max iterations")

    @property
    def ee_link(self) -> Link:
        return self.links[-1] if self.links else None
    
    @property
    def base_link(self) -> Link:
        return self.links[0] if self.links else None

    @property
    def actuated_joints(self) -> List[Joint]:
        return [joint for joint in self.joints if joint.joint_type in ("revolute", "prismatic")]
    
    @staticmethod
    def compute_tsp_rot_error(x_d: sm.SE3, T_ee: sm.SE3) -> NDArray:
        """Computes the orientation error between desired and current end-effector poses.
        using exponential coordinates
        """
        R_d = x_d.R
        R_ee = T_ee.R
        R_err = R_d @ R_ee.T # equation 3.91 Bruno and Siciliano
        dr = 0.5 * np.array([
            R_err[2,1] - R_err[1,2],
            R_err[0,2] - R_err[2,0],
            R_err[1,0] - R_err[0,1]
        ])
        return dr
    
    @staticmethod
    def clamp_limits(config: Configuration, scale: float = 1.0) -> Configuration:
        """Clamps joint values to their respective limits."""
        new_values = []
        for joint, joint_value in zip(config.joints, config.joint_values):
            lower, upper = joint.limits
            if joint_value < lower:
                new_values.append(scale * lower)
            elif joint_value > upper:
                new_values.append(scale * upper)
            else:
                new_values.append(joint_value)
        return Robot.Configuration(config.joints, new_values)

    @staticmethod
    def check_limits(config: Configuration) -> bool:
        """Checks whether a joint configuration is within limits."""
        for joint, joint_value in zip(config.joints, config.joint_values):
            lower, upper = joint.limits
            if not (lower <= joint_value <= upper):
                return False
        return True

    @staticmethod
    def check_inv(jacobian: NDArray, tol:Union[float, None]=1e12) -> bool:
        """Checks whether a configuration is singular by 
        evaluating the condition number of the 
        Jacobian argument
        """
        cond_num = np.linalg.cond(jacobian)
        if cond_num >= tol:
            return False
        else: 
            return True
        
    @staticmethod
    def rodrigues_rot(axis: List, angle: float) -> NDArray:
        """Rodrigues' formula for axis-angle rotation.

        Args:
            axis (List): A list specifying the axis to rotate about.
            angle (float): The angle in radians to rotate <axis> by.

        Returns:
            NDArray: The resulting axis-angle rotation matrix computed.
        """
        try:
            exp_theta: NDArray = np.eye(len(axis)) + np.sin(angle)*Robot.skew(axis) + (1-np.cos(angle)*(Robot.skew(axis) @ Robot.skew(axis)))
            assert exp_theta.shape()[0] == len(axis) and exp_theta.shape()[1] == len(axis)
            return exp_theta
        except Exception as e:
            print(f"Oops! Failed to compute Rodrigues' rotation. Revisit Spong: {e}.")

    @staticmethod
    def skew(v: List) -> NDArray:
        """Returns the skew-symmetric matrix corresponding to a 3-vector for cross-product computation.

        Args:
            v (List): A list or NDArray of shape (3,1) specifying the vector components along the base x, y, and z axes

        Raises:
            Exception: If the list is not a length-3 array.

        Returns:
            NDArray: A 3x3 Numpy array containing the corresponding skew-symmetric matrix.
        """
        try:
            if len(v) == 3:
                return np.array([
                    [   0, -v[2],  v[1]],
                    [v[2],     0, -v[0]],
                    [-v[1], v[0],    0]])
        except Exception as e:
            print(f"The array must be of length 3 to compute a sensible skew-symmetric matrix for 3D vectors: {e}")
        
    @staticmethod
    def _print_matrix_colored(A: sm.SE3):
        """Print a 4x4 matrix with colored formatting for better readability."""
        for i, row in enumerate(A):
            row_str = []
            for j, val in enumerate(row):
                if i < 3 and j < 3 and i == j:
                    row_str.append('\033[91m' + f"{val: .2f}" + '\033[0m')
                elif j == 3 and i < 3:
                    row_str.append('\033[94m' + f"{val: .2f}" + '\033[0m')
                else:
                    row_str.append(f"{val: .2f}")
            print(' '.join(row_str))

    @staticmethod
    def _parse_urdf(desc_fp) -> ET.Element:
        """
        Parses the URDF XML file and returns the ElementTree object.
        """
        return ET.parse(desc_fp)

    @staticmethod
    def _set_config(config: Configuration) -> None:
        """
        Validates that the config has entries for all joints in config.joints.
        """
        missing = [j.name for j in config.joints if j.name not in config.joint_dict]
        if missing:
            raise ValueError(f"Configuration missing joint values for: {missing}")

