cmake_minimum_required(VERSION 3.0.2)
project(fruit_harvesting_robot)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
  controller_manager
  joint_state_controller
  robot_state_publisher
  std_msgs
  geometry_msgs
  sensor_msgs
)

find_package(gazebo REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    gazebo_ros
    controller_manager
    std_msgs
    geometry_msgs
    sensor_msgs
)

add_library(robot_control_plugin SHARED
  src/robot_control_plugin.cpp
)

target_link_libraries(robot_control_plugin
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)