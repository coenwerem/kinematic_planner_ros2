#!/usr/bin/env python3

"""Pairwise self-collision checking over the FCL objects RobotCollisionModel
builds for each link. Adjacent-link pairs and explicitly disabled pairs are
excluded upstream by RobotConfig.get_collision_pairs(). check_self_collision
checks the collision_pairs argument directly and adds no filtering of its
own.
"""

from typing import Dict, List, Tuple

import fcl


def check_self_collision(link_fcl_objects: Dict[str, List["fcl.CollisionObject"]],
                          collision_pairs: List[Tuple[str, str]]) -> bool:
    for link_a, link_b in collision_pairs:
        objects_a = link_fcl_objects.get(link_a, [])
        objects_b = link_fcl_objects.get(link_b, [])
        for obj_a in objects_a:
            for obj_b in objects_b:
                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()
                if fcl.collide(obj_a, obj_b, request, result) > 0:
                    return True
    return False
