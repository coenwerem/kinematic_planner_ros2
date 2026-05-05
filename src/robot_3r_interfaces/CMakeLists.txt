cmake_minimum_required(VERSION 3.8)
project(robot_3r_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(robot_3r_description REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rcl_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointWaypoint.msg"
  "msg/JointSpacePath.msg"
  "msg/CartesianPath.msg"
  "msg/SceneObstacles.msg"
  "msg/RigidBodyGeom.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs shape_msgs visualization_msgs
)

install(DIRECTORY
  msg
  DESTINATION share/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime std_msgs sensor_msgs geometry_msgs shape_msgs visualization_msgs)
ament_package()
