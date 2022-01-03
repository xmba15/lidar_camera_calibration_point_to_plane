cmake_minimum_required(VERSION 3.10)
project(lidar_cam_utils)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  message_filters
)

catkin_package(
)
