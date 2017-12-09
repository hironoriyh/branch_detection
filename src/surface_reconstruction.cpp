/*
 * surface_reconstruction.cpp
 *
 *  Created on: 14, 11, 2017
 *      Author: Hironori Yoshida
 */
#include "branch_surface/surface_reconstruction.hpp"
#include <ros/ros.h>
#include <ros/package.h>

#include <vector>

#include <pcl/PCLPointCloud2.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/mls.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/texture_mapping.h>
//#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_hoppe.h>


#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/vtk_io.h>

#include "branch_surface/filtering.hpp"

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>


//using namespace point_cloud_filtering;

namespace surface_reconstruction_srv {

SurfaceReconstructionSrv::SurfaceReconstructionSrv(ros::NodeHandle nodeHandle)
    : nodeHandle_(nodeHandle),
      colored_cloud_vector_(),
      bound_vec_()
{

	nodeHandle.getParam("/surface_reconstruction_service/leaf_size", leaf_size_);
	nodeHandle.getParam("/surface_reconstruction_service/model_folder", model_folder_);

	nodeHandle.getParam("/surface_reconstruction_service/bin_size", bin_size_);
	nodeHandle.getParam("/surface_reconstruction_service/inlier_dist_segmentation", inlier_dist_segmentation_);
	nodeHandle.getParam("/surface_reconstruction_service/segmentation_inlier_ratio", segmentation_inlier_ratio_);
	nodeHandle.getParam("/surface_reconstruction_service/max_number_of_instances", max_number_of_instances_);
	nodeHandle.getParam("/surface_reconstruction_service/max_fitness_score", max_fitness_score_);
	nodeHandle.getParam("/surface_reconstruction_service/inlier_dist_icp", inlier_dist_icp_);
	nodeHandle.getParam("/surface_reconstruction_service/icp_inlier_threshold", icp_inlier_threshold_);
	nodeHandle.getParam("/surface_reconstruction_service/min_inlier_ratio_validation", min_inlier_ratio_validation_);
	nodeHandle.getParam("/surface_reconstruction_service/inlier_dist_validation", inlier_dist_validation_);
	nodeHandle.getParam("/surface_reconstruction_service/grid_res", grid_res_);

	nodeHandle.getParam("/surface_reconstruction_service/number_of_average_clouds", number_of_average_clouds_);
	nodeHandle.getParam("/surface_reconstruction_service/number_of_median_clouds", number_of_median_clouds_);
	nodeHandle.getParam("/surface_reconstruction_service/z_threshold", z_threshold_);
	nodeHandle.getParam("/surface_reconstruction_service/planarSegmentationTolerance", planarSegmentationTolerance_);
	nodeHandle.getParam("/surface_reconstruction_service/min_number_of_inliers", min_number_of_inliers_);
	nodeHandle.getParam("/surface_reconstruction_service/xmin", xmin_);
	nodeHandle.getParam("/surface_reconstruction_service/xmax", xmax_);
	nodeHandle.getParam("/surface_reconstruction_service/ymin", ymin_);
	nodeHandle.getParam("/surface_reconstruction_service/ymax", ymax_);
	nodeHandle.getParam("/surface_reconstruction_service/zmin", zmin_);
	nodeHandle.getParam("/surface_reconstruction_service/zmax", zmax_);
	nodeHandle.getParam("/surface_reconstruction_service/point_cloud_topic", point_cloud_topic_);
  nodeHandle.getParam("/surface_reconstruction_service/use_saved_pc", use_saved_pc_);


	bound_vec_.push_back(xmin_);
	bound_vec_.push_back(xmax_);
	bound_vec_.push_back(ymin_);
	bound_vec_.push_back(ymax_);
	bound_vec_.push_back(zmin_);
	bound_vec_.push_back(zmax_);

	ros::ServiceServer service = nodeHandle_.advertiseService("/get_surface", &SurfaceReconstructionSrv::callGetSurface, this);

	ROS_INFO("Ready to define surface.");
	ros::spin();
}

SurfaceReconstructionSrv::~SurfaceReconstructionSrv()
{
}

bool SurfaceReconstructionSrv::callGetSurface(std_srvs::EmptyRequest &req,
                                              std_srvs::EmptyResponse &resp)
{

  std::cout << "service called!" << std::endl;
//   // Sample clouds
//   ros::Subscriber sub = nodeHandle_.subscribe(point_cloud_topic_, 1,
//                                               &SurfaceReconstructionSrv::saveCloud, this);
//   ros::Rate r(60);
//   std::cout << "Waiting for point cloud images..." << std::endl << std::endl;
//
//   while (colored_cloud_vector_.size() < number_of_median_clouds_ + number_of_average_clouds_) {
// //    ROS_INFO_STREAM("cloud vector num: " << cloud_vector_.size());
//     ros::spinOnce();
//     r.sleep();
//   }
//   std::cout << "Clouds are sampled." << "  width = " << colored_cloud_vector_[0].width << "  height = "
//             << colored_cloud_vector_[0].height << "  size = " << colored_cloud_vector_[0].size() << std::endl;


  // if(use_saved_pc_)
  PointCloud<PointType>::Ptr cloud_ptr (new PointCloud<PointType>);
  // else PointCloud<PointType>::Ptr cloud_ptr (new PointCloud<PointType>(cloud_vector_[0]));
  PointCloud<Normal>::Ptr cloud_normals(new PointCloud<Normal>);

  // loadPCD or pcd
  std::string file_name ="/home/hyoshdia/Documents/realsense_pcl/cloud_raw.pcd";
  if (pcl::io::loadPCDFile<PointType>(file_name, *cloud_ptr) == -1){
    PCL_ERROR("Couldn't read pcd file \n");
  }

	preprocess(cloud_ptr);
	DownSample(cloud_ptr);
	computeNormals(cloud_ptr, cloud_normals);

//  std::cout << "begin passthrough filter" << std::endl;
//  PointCloud<PointType>::Ptr filtered(new PointCloud<PointType>());
//  PassThrough<PointType> filter;
//  filter.setInputCloud(cloud_ptr);
//  filter.filter(*filtered);
//  std::cout << "passthrough filter complete" << std::endl;
//
//  std::cout << "begin moving least squares" << std::endl;
//  MovingLeastSquares<PointType, PointType> mls;
//  mls.setInputCloud(filtered);
//  mls.setSearchRadius(0.01);
//  mls.setPolynomialFit(true);
//  mls.setPolynomialOrder(2);
//  mls.setUpsamplingMethod(MovingLeastSquares<PointType, PointType>::SAMPLE_LOCAL_PLANE);
//  mls.setUpsamplingRadius(0.005);
//  mls.setUpsamplingStepSize(0.003);
//  PointCloud<PointType>::Ptr cloud_smoothed(new PointCloud<PointType>());
//  mls.process(*cloud_smoothed);
//  std::cout << "MLS complete" << std::endl;
//
//  std::cout << "begin normal estimation" << std::endl;
//  NormalEstimationOMP<PointType, Normal> ne;
//  ne.setNumberOfThreads(8);
//  ne.setInputCloud(cloud_ptr);
//  ne.setRadiusSearch(0.01);
//  Eigen::Vector4f centroid;
//  compute3DCentroid(*cloud_ptr, centroid);
//  ne.setViewPoint(centroid[0], centroid[1], centroid[2]);
//
//  ne.compute(*cloud_normals);
//  std::cout << "normal estimation complete" << std::endl;
//  std::cout << "reverse normals' direction" << std::endl;
//
//  for (size_t i = 0; i < cloud_normals->size(); ++i) {
//    cloud_normals->points[i].normal_x *= -1;
//    cloud_normals->points[i].normal_y *= -1;
//    cloud_normals->points[i].normal_z *= -1;
//  }

  std::cout << "combine points and normals" << std::endl;
  PointCloud<PointXYZRGBNormal>::Ptr cloud_smoothed_normals(new PointCloud<PointXYZRGBNormal>());

//	PointCloud<PointXYZ>::Ptr no_color_cloud_ptr(new PointCloud<PointXYZ>);
//	copyPointCloud(*cloud_ptr, *no_color_cloud_ptr);
	concatenateFields(*cloud_ptr, *cloud_normals, *cloud_smoothed_normals);


  std::string path = "/home/hyoshdia/Documents/realsense_pcl/cloud.pcd";
  io::savePCDFileASCII(path, *cloud_ptr);
  path = "/home/hyoshdia/Documents/realsense_pcl/cloud.ply";
  io::savePLYFile(path, *cloud_ptr);

  std::cout << "begin poisson reconstruction" << std::endl;
  Poisson<PointXYZRGBNormal> poisson;
  poisson.setDepth(9);
  poisson.setManifold(0);
  // poisson.setSmaplesPerNode(1.0);
  // poisson.setRatioDiameter(1.25);
  poisson.setDegree(2);
  poisson.setIsoDivide(8);
  // poisson.setSolveDivide(8);
  poisson.setOutputPolygons(0);
  poisson.setInputCloud(cloud_smoothed_normals);
  PolygonMesh mesh;
  poisson.reconstruct(mesh);
  path = "/home/hyoshdia/Documents/realsense_pcl/poisson.ply";
  io::savePLYFile(path, mesh);

//  std::cout << "begin marching cube" << std::endl;
//  MarchingCubes<PointNormal> * mc;
//  mc = new MarchingCubesHoppe<PointNormal> ();
//  float iso_level = 0.0f;
//  int hoppe_or_rbf = 0;
//  float extend_percentage = 0.0f;
//  int grid_res = grid_res_;
//  float off_surface_displacement = 0.01f;
//  mc->setIsoLevel (iso_level);
//  mc->setGridResolution (grid_res, grid_res, grid_res);
//  mc->setPercentageExtendGrid (extend_percentage);
//  mc->setInputCloud (cloud_smoothed_normals);
//  mc->reconstruct(mesh);
//  path = "/home/hyoshdia/Documents/realsense_pcl/mc.ply";
//  io::savePLYFile(path, mesh);
//  delete mc;

  // Create search tree*
//  std::cout << "begin GreedyProjectionTriangulation" << std::endl;
//  search::KdTree<PointNormal>::Ptr tree2(new search::KdTree<PointNormal>);
//  tree2->setInputCloud(cloud_smoothed_normals);
//  // Initialize objects
//  GreedyProjectionTriangulation<PointNormal> gp3;
//  gp3.setSearchRadius(0.025);   // Set the maximum distance between connected points (maximum edge length)
//  gp3.setMu(2.5);   // Set typical values for the parameters
//  gp3.setMaximumNearestNeighbors(300);
//  gp3.setMaximumSurfaceAngle(M_PI / 4);  // 45 degrees
//  gp3.setMinimumAngle(M_PI / 18);  // 10 degrees
//  gp3.setMaximumAngle(2 * M_PI / 3);  // 120 degrees
//  gp3.setNormalConsistency(false);
//
//  // Get result
//  gp3.setInputCloud(cloud_smoothed_normals);
//  gp3.setSearchMethod(tree2);
//  gp3.reconstruct(mesh);
//  path = "/home/hyoshdia/Documents/realsense_pcl/gp.ply";
//  io::savePLYFile(path, mesh);
//
//  //visualization
//  std::shared_ptr<visualization::PCLVisualizer> viewer1;
//  std::shared_ptr<visualization::PCLVisualizer> viewer2;
//
//  viewer1 = normalsVis(cloud_ptr, cloud_normals);
//  viewer2 = rgbVis(colored_cloud_ptr);
//  while (!viewer1->wasStopped()) {
//    viewer1->spinOnce(100);
//    viewer2->spinOnce(100);
//    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//  }

  std::cout << "service done!" << std::endl;

  return true;
}

bool SurfaceReconstructionSrv::preprocess(PointCloud<PointType>::Ptr preprocessed_cloud_ptr_) {
  // DO NOT MODIFY! Parameter recalculation
  std::vector<float> boundaries;
  boundaries.push_back(xmin_);
  boundaries.push_back(xmax_);
  boundaries.push_back(ymin_);
  boundaries.push_back(ymax_);
  boundaries.push_back(zmin_);
  boundaries.push_back(zmax_);

  point_cloud_filtering::filtering filtering;
  filtering.setNumberOfMedianClouds(number_of_median_clouds_);
  filtering.setNumberOfAverageClouds(number_of_average_clouds_);
  filtering.setInputClouds(cloud_vector_);
  filtering.setClippingBoundaries(boundaries);
  filtering.setZThreshold(z_threshold_);
  filtering.setPlanarSegmentationTolerance(planarSegmentationTolerance_);
  filtering.setMinNumberOfInliers(min_number_of_inliers_);
  filtering.compute(preprocessed_cloud_ptr_);
  unsigned int preprocessed_size_ = preprocessed_cloud_ptr_->size();

  std::cout << "object detection, preprocess() " << preprocessed_size_ << std::endl;
  return true;
}

bool SurfaceReconstructionSrv::DownSample(PointCloud<PointType>::Ptr &cloud_) {
  ROS_INFO("Downsampled");
  ROS_INFO_STREAM("num of samples before down sample: " << cloud_->size());

  PointCloud<PointType>::Ptr cloud_downsampled(
      new PointCloud<PointType>());
  VoxelGrid<PointType> sor;
  sor.setInputCloud(cloud_);
  sor.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  sor.filter(*cloud_downsampled);

  *cloud_ = *cloud_downsampled;

  ROS_INFO_STREAM("num of samples after down sample: " << cloud_->size());
  return true;
}


bool SurfaceReconstructionSrv::computeNormals(const PointCloud<PointType>::ConstPtr &cloud_,
                    PointCloud<Normal>::Ptr &normals_)
{
  ROS_INFO("compute normals");
  NormalEstimation<PointType, Normal> n;
  search::KdTree<PointType>::Ptr tree(new search::KdTree<PointType>);
  tree->setInputCloud(cloud_);
  n.setInputCloud(cloud_);
  n.setSearchMethod(tree);
  n.setRadiusSearch(0.01);
  n.compute(*normals_);
  return true;
}


std::shared_ptr<visualization::PCLVisualizer> SurfaceReconstructionSrv::normalsVis(
    PointCloud<PointType>::Ptr &cloud, PointCloud<Normal>::Ptr &normals)
{
  std::shared_ptr<visualization::PCLVisualizer> viewer(
      new visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(1.0, 1.0, 1.0);
  viewer->addPointCloud<PointType>(cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0,
                                           "sample cloud");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2,
                                           "sample cloud");

  viewer->addPointCloudNormals<PointType, Normal>(cloud, normals, 10, 0.05, "normals");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0,
                                           "normals");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 0.01,
                                           "normals");

  viewer->addCoordinateSystem(0.1);
  viewer->setCameraPosition(0, 0, 0.1, 0.5, 0.0, 0.0, 0.1);
  viewer->initCameraParameters();
  return (viewer);
}

std::shared_ptr<visualization::PCLVisualizer> SurfaceReconstructionSrv::rgbVis (PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  std::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (1.0, 1.0, 1.0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

  viewer->addCoordinateSystem (0.1);
  viewer->setCameraPosition(0,0,0.1, 0.5, 0.0, 0.0, 0.1);  viewer->initCameraParameters ();
  return (viewer);
}

void SurfaceReconstructionSrv::saveCloud(const sensor_msgs::PointCloud2& cloud)
{
  std::cout << "the length of cloud sensor msg is " << cloud.data[100] << std::endl;

//  PointCloud <PointXYZRGB> new_cloud;
//  fromROSMsg(cloud, new_cloud);
//  colored_cloud_vector_.push_back(new_cloud);

  PointCloud<PointType> new_cloud_2;
  fromROSMsg(cloud, new_cloud_2);
//  copyPointCloud(new_cloud, new_cloud_2);
  cloud_vector_.push_back(new_cloud_2);
}

}/*end namespace*/
