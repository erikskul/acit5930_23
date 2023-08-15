cmake_minimum_required(VERSION 3.1)
project(terrain_characterizer)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
  genmsg
  sensor_msgs
  cv_bridge
  roscpp
  std_msgs
  image_transport
  pcl_ros
  roscpp
  sensor_msgs
  visualization_msgs
  dynamic_reconfigure
)

add_message_files(
  FILES
  Errors.msg
  ErrorNav.msg
)

generate_messages()

find_package(PCL REQUIRED)

generate_dynamic_reconfigure_options(
    cfg/algorithmParameters.cfg
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)



add_executable(pointCloudPlaneFitter src/pointCloudPlaneFitter.cpp)
target_link_libraries(pointCloudPlaneFitter ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(pointCloudPlaneFitter ${catkin_EXPORTED_TARGETS})

