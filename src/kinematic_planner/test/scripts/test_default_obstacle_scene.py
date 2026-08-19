from kinematic_planner.scripts.obstacle_publisher import default_obstacle_scene


def test_sparse_obstacles_rest_on_top_of_the_platform():
    positions, sizes = default_obstacle_scene(is_dense=False, platform_height=0.755)
    height = sizes[2]
    for i in range(2):
        z = positions[3 * i + 2]
        assert abs((z - height / 2) - 0.755) < 1e-9


def test_dense_obstacles_rest_on_top_of_the_platform_for_both_size_classes():
    positions, sizes = default_obstacle_scene(is_dense=True, platform_height=0.755, num_obstacles=10)
    for i in range(10):
        z = positions[3 * i + 2]
        height = sizes[3 * i + 2]
        # every obstacle's bottom face sits exactly at platform_height,
        # regardless of which of the two size classes (even/odd index) it is
        assert abs((z - height / 2) - 0.755) < 1e-9


def test_zero_platform_height_matches_the_original_floating_default():
    positions, _sizes = default_obstacle_scene(is_dense=False, platform_height=0.0)
    assert positions == [0.0, 0.6, 0.5, 0.6, 0.0, 0.5]
