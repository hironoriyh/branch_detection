/*
 * surface_reconstruction.cpp
 *
 *  Created on: 14, 11, 2017
 *      Author: Hironori Yoshida
 */

#include <branch_surface/surface_reconstruction.hpp>
#include <ros/ros.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "surface_reconstruciont_srv");
  ros::NodeHandle nodeHandle("~");

  surface_reconstruction_srv::SurfaceReconstructionSrv srf_rcon_srv(nodeHandle);

  ros::waitForShutdown();
  return 0;
}
