cmake_minimum_required(VERSION 2.8.3)
project(branch_surface)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

find_package(PCL REQUIRED)

#add_service_files(
#  FILES
#  DetectObject.srv
#  LocaliseObjects.srv
#)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS
  	include
  LIBRARIES
#  	object_detection
  CATKIN_DEPENDS
  	pcl_conversions
  	pcl_ros roscpp
  	rospy sensor_msgs
  	std_msgs
  	message_runtime
  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


## Declare a C++ library
# SurfaceReconstruction Service
 add_executable(
	surface_reconstruction_service
			src/surface_reconstruction_service_node.cpp
			src/filtering.cpp
			src/surface_reconstruction.cpp
	)
	add_dependencies(surface_reconstruction_service 
	object_detection_gencpp)
	target_link_libraries(surface_reconstruction_service ${catkin_LIBRARIES} ${PCL_LIBRARIES})
