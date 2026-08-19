#!/usr/bin/env python3

"""
RobotConfig: a pure-Python dataclass for robot metadata parsed entirely from a URDF string.

Joint limits, link names, and kinematic topology
are extracted directly from the URDF XML using the standard library xml.etree module.

Author: Clinton Enwerem
"""

import xml.etree.ElementTree as ET
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RobotConfig:
    joint_names: List[str]
    joint_limits: List[Tuple[float, float]]   # [(min_rad, max_rad), ...]  ordered with joint_names
    link_names: List[str]                      # links that have <collision> geometry
    base_link_name: str
    ee_link_name: str
    world_frame: str
    disabled_collision_pairs: List[Tuple[str, str]] = field(default_factory=list)

    # derived convenience
    @property
    def num_dof(self) -> int:
        return len(self.joint_names)

    def get_collision_pairs(self) -> List[Tuple[str, str]]:
        """All link pairs that must be checked for collision (disabled pairs excluded)."""
        all_pairs = list(itertools.combinations(self.link_names, 2))
        return [
            (a, b) for (a, b) in all_pairs
            if (a, b) not in self.disabled_collision_pairs
            and (b, a) not in self.disabled_collision_pairs
        ]

    @classmethod
    def from_urdf(
        cls,
        urdf_str: str,
        disabled_pairs: List[Tuple[str, str]] = None,
        world_frame: str = "world",
        actuated_joint_types: Tuple[str, ...] = ("revolute", "prismatic", "continuous"),
    ) -> "RobotConfig":
        """
        Parse joint names, limits, and link names from a URDF string.

        Args:
            urdf_str: Raw URDF XML as a string (e.g. from the robot_description parameter).
            disabled_pairs: Link pairs to skip during self-collision checks.
                            Pass the adjacency pairs from your SRDF here, or leave None
                            to check all link pairs (expensive but correct as a default).
            world_frame: Name of the world / fixed frame used in transforms.
            actuated_joint_types: Joint types treated as degrees of freedom.

        Returns:
            A fully populated RobotConfig instance.
        """
        root = ET.fromstring(urdf_str)

        # --- joint data ---------------------------------------------------------
        child_links = set()   # links serving as a joint's child, each with a parent
        parent_links = set()  # links serving as a joint's parent
        adjacent_link_pairs: List[Tuple[str, str]] = []

        joint_names: List[str] = []
        joint_limits: List[Tuple[float, float]] = []

        for joint in root.findall("joint"):
            jtype = joint.get("type", "fixed")
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            if parent_el is not None:
                parent_links.add(parent_el.get("link", ""))
            if child_el is not None:
                child_links.add(child_el.get("link", ""))
            if parent_el is not None and child_el is not None:
                adjacent_link_pairs.append((parent_el.get("link", ""), child_el.get("link", "")))

            if jtype not in actuated_joint_types:
                continue

            jname = joint.get("name", "")
            limit_el = joint.find("limit")
            if limit_el is not None:
                lo = float(limit_el.get("lower", "-3.14159"))
                hi = float(limit_el.get("upper", "3.14159"))
            else:
                lo, hi = -3.14159, 3.14159

            joint_names.append(jname)
            joint_limits.append((lo, hi))

        # --- link data ----------------------------------------------------------
        all_link_names = [lnk.get("name", "") for lnk in root.findall("link")]

        # base link = has no parent joint
        base_link_name = next(
            (ln for ln in all_link_names if ln not in child_links), ""
        )

        # ee link = has no child joint (last in chain)
        ee_link_name = next(
            (ln for ln in reversed(all_link_names) if ln not in parent_links), ""
        )

        # collision links = links that carry <collision> geometry
        collision_links: List[str] = []
        for lnk in root.findall("link"):
            if lnk.find("collision") is not None:
                collision_links.append(lnk.get("name", ""))

        merged_disabled_pairs = list(disabled_pairs or []) + adjacent_link_pairs

        return cls(
            joint_names=joint_names,
            joint_limits=joint_limits,
            link_names=collision_links,
            base_link_name=base_link_name,
            ee_link_name=ee_link_name,
            world_frame=world_frame,
            disabled_collision_pairs=merged_disabled_pairs,
        )
