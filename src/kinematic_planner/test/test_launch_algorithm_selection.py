from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
import importlib.util
import pathlib

_LAUNCH_FILE = pathlib.Path(__file__).parents[1] / "launch" / "planner.launch.py"
_SPEC = importlib.util.spec_from_file_location("planner_launch", _LAUNCH_FILE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _node_actions_by_executable(launch_description):
    from launch_ros.actions import Node
    result = {}
    for entity in launch_description.entities:
        if isinstance(entity, Node):
            result[entity._Node__node_executable] = entity
    return result


def _evaluate_with_algorithm(algorithm_value):
    launch_description = _MODULE.generate_launch_description()
    context = LaunchContext()
    for entity in launch_description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.visit(context)
    context.launch_configurations["algorithm"] = algorithm_value
    nodes = _node_actions_by_executable(launch_description)
    return {
        name: (node.condition.evaluate(context) if node.condition is not None else True)
        for name, node in nodes.items()
    }


def test_rrt_star_selection_launches_only_planner_node():
    active = _evaluate_with_algorithm("rrt_star")
    assert active["planner_node"] is True
    assert active["informed_rrt_star_node"] is False


def test_informed_rrt_star_selection_launches_only_informed_node():
    active = _evaluate_with_algorithm("informed_rrt_star")
    assert active["planner_node"] is False
    assert active["informed_rrt_star_node"] is True
