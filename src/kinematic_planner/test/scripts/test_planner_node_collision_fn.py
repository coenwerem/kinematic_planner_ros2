# src/kinematic_planner/test/scripts/test_planner_node_collision_fn.py
"""Exercises build_collision_fn directly, without constructing an
rclpy.Node, matching the release spec's requirement that planner-
algorithm tests not require an rclpy graph."""
from kinematic_planner.scripts.planner_node import build_collision_fn
from kinematic_planner.planning.tree import TreeNode
import numpy as np


class _FakeRobotConfig:
    base_link_name = "base_link"
    world_frame = "world"

    def get_collision_pairs(self):
        return []


def test_check_collision_false_accepts_every_candidate():
    fn = build_collision_fn(
        robot_config=_FakeRobotConfig(),
        link_shapes={},
        obstacle_geom=None,
        rtb_model=None,
        collision_checker="proximity",
        min_obs_dist=0.1,
        check_collision=False,
    )
    node = TreeNode(np.array([0.0, 0.0, 0.0]))
    node.path_q = [node.q]
    assert fn(node) is True


class _RaisingLink:
    name = "link1"


class _RaisingRTBModel:
    """Stands in for an rtb_model whose fkine() raises, exercising the
    per-link fail-safe: an exception during FK/geometry/FCL work must
    make collision_fn return False rather than propagate."""

    links = [_RaisingLink()]

    def fkine(self, q, end=None, include_base=True):
        raise RuntimeError("fk failed")


class _FakeObstacleGeom:
    obstacle_ids = []


class _LoggerSpy:
    def __init__(self):
        self.errors = []

    def error(self, msg):
        self.errors.append(msg)


def test_collision_fn_returns_false_and_logs_on_per_link_exception():
    logger = _LoggerSpy()
    fn = build_collision_fn(
        robot_config=_FakeRobotConfig(),
        link_shapes={"link1": []},
        obstacle_geom=_FakeObstacleGeom(),
        rtb_model=_RaisingRTBModel(),
        collision_checker="proximity",
        min_obs_dist=0.1,
        check_collision=True,
        get_logger=lambda: logger,
    )
    node = TreeNode(np.array([0.0, 0.0, 0.0]))
    node.path_q = [node.q]
    assert fn(node) is False
    assert logger.errors


class _MalformedObstacleGeom:
    """Stands in for an obstacle_geom whose obstacle_ids/scene_obstacles
    are inconsistent, e.g. shorter than obstacle_ids, so
    obstacle_to_fclobj() raises (an IndexError here) while iterating."""

    obstacle_ids = ["obs_0"]
    obstacle_poses = []
    scene_obstacles = []


def test_collision_fn_returns_false_and_logs_when_obstacle_conversion_raises():
    """Finding 5: obstacle_to_fclobj(obstacle_geom) must not be able to
    raise out of collision_fn uncaught -- a malformed /scene_obstacles
    message must fail safe (treated as a collision) rather than crash
    the /joint_states callback."""
    logger = _LoggerSpy()
    fn = build_collision_fn(
        robot_config=_FakeRobotConfig(),
        link_shapes={},
        obstacle_geom=_MalformedObstacleGeom(),
        rtb_model=_RaisingRTBModel(),  # never reached; obstacle conversion fails first
        collision_checker="proximity",
        min_obs_dist=0.1,
        check_collision=True,
        get_logger=lambda: logger,
    )
    node = TreeNode(np.array([0.0, 0.0, 0.0]))
    node.path_q = [node.q]
    assert fn(node) is False
    assert logger.errors


class _PartialChainLink:
    def __init__(self, name):
        self.name = name


class _PartialChainRTBModel:
    """Stands in for an rtb_model that exposes fewer FK-reachable links than
    the URDF's own <collision> elements list, e.g. a link off the main
    serial chain. fkine() only succeeds for links in `links`; a call for
    any other end name raises, so a regression that stops filtering
    link_shapes down to the RTB-reachable set is caught by this test."""

    links = [_PartialChainLink("link1")]

    def fkine(self, q, end=None, include_base=True):
        if end != "link1":
            raise ValueError(f"{end} is not a valid end-effector link name")

        class _SE3:
            A = np.eye(4)
        return _SE3()


def test_unreachable_urdf_link_is_dropped_not_blanket_failed():
    """Finding 1: a URDF collision link absent from rtb_model.links (e.g.
    off the main serial chain, or any URDF/RTB naming mismatch) must be
    dropped from collision_fn's per-waypoint FK loop and logged once at
    build_collision_fn's construction time, not cause every candidate to
    fail through the shared per-waypoint try/except."""
    logger = _LoggerSpy()
    fn = build_collision_fn(
        robot_config=_FakeRobotConfig(),
        link_shapes={"link1": [], "link_off_chain": []},
        obstacle_geom=_FakeObstacleGeom(),
        rtb_model=_PartialChainRTBModel(),
        collision_checker="proximity",
        min_obs_dist=0.1,
        check_collision=True,
        get_logger=lambda: logger,
    )
    node = TreeNode(np.array([0.0, 0.0, 0.0]))
    node.path_q = [node.q]
    assert fn(node) is True
    assert logger.errors
    assert any("link_off_chain" in msg for msg in logger.errors)


class _SucceedingRTBModel:
    """Stands in for an rtb_model whose fkine() succeeds trivially, so
    collision_fn reaches the self-collision layer without tripping the
    per-link FK fail-safe above it."""

    links = [_RaisingLink()]  # name "link1", matching link_shapes below

    def fkine(self, q, end=None, include_base=True):
        class _SE3:
            A = np.eye(4)
        return _SE3()


def test_collision_fn_returns_false_and_logs_when_self_collision_check_raises(monkeypatch):
    """The self-collision layer must fail safe the same way the other three
    FCL-touching layers in collision_fn already do: an exception raised by
    check_self_collision must not propagate out of collision_fn into
    RRTStar.plan() and crash the /joint_states callback."""
    import kinematic_planner.scripts.planner_node as planner_node_module

    def _raising_check_self_collision(link_fcl_objects, collision_pairs):
        raise RuntimeError("self-collision check failed")

    monkeypatch.setattr(planner_node_module, "check_self_collision", _raising_check_self_collision)

    logger = _LoggerSpy()
    fn = build_collision_fn(
        robot_config=_FakeRobotConfig(),
        link_shapes={"link1": []},
        obstacle_geom=_FakeObstacleGeom(),
        rtb_model=_SucceedingRTBModel(),
        collision_checker="proximity",
        min_obs_dist=0.1,
        check_collision=True,
        get_logger=lambda: logger,
    )
    node = TreeNode(np.array([0.0, 0.0, 0.0]))
    node.path_q = [node.q]
    assert fn(node) is False
    assert logger.errors
