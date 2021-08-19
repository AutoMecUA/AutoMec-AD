cmake_minimum_required(VERSION 3.0.2)
project(simulation_environment)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES simulation_environment
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(gazebo_signal_panel src/gazebo_signalling_panel_controller.cpp)
add_dependencies(gazebo_signal_panel ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gazebo_signal_panel
   ${catkin_LIBRARIES}
 )