/*
 * surface_reconstruction.cpp
 *
 *  Created on: 14, 11, 2017
 *      Author: Hironori Yoshida
 */
#include "branch_surface/surface_reconstruction.hpp"
#include "branch_surface/filtering.hpp"

#include <vector>
#include <Eigen/Geometry>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_broadcaster.h>
//#include <tf/transform_listener.h>

// ros
#include <ros/ros.h>
#include <ros/package.h>

#include <pcl/PCLPointCloud2.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>

// io
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// point types etc
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/mls.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

// surface reconstruction
#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/texture_mapping.h>
//#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_hoppe.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/vtk_io.h>


// region growing
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/search/search.h>

#include <pcl/common/centroid.h>
#include <pcl/filters/project_inliers.h>

namespace surface_reconstruction_srv {

SurfaceReconstructionSrv::SurfaceReconstructionSrv(ros::NodeHandle nodeHandle)
    : nodeHandle_(nodeHandle)
{

  save_package_ = "branch_surface";

  nodeHandle.getParam("/surface_reconstruction_service/leaf_size", leaf_size_);
  nodeHandle.getParam("/surface_reconstruction_service/model_folder", model_folder_);

  nodeHandle.getParam("/surface_reconstruction_service/bin_size", bin_size_);
  nodeHandle.getParam("/surface_reconstruction_service/bin_size_hough", bin_size_hough_);

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
  nodeHandle.getParam("/surface_reconstruction_service/save_clouds", save_clouds_);
  nodeHandle.getParam("/surface_reconstruction_service/save_path", save_path_);
  nodeHandle.getParam("/surface_reconstruction_service/save_package", save_package_);

  nodeHandle.getParam("/surface_reconstruction_service/use_all_points", use_all_points_);

  nodeHandle.getParam("/surface_reconstruction_service/object_frame", object_frame_);
  nodeHandle.getParam("/surface_reconstruction_service/camera_frame", camera_frame_);
  nodeHandle.getParam("/surface_reconstruction_service/normal_flip", normal_flip_);
  nodeHandle.getParam("/surface_reconstruction_service/reorient_cloud", reorient_cloud_);


  bound_vec_.push_back(xmin_);
  bound_vec_.push_back(xmax_);
  bound_vec_.push_back(ymin_);
  bound_vec_.push_back(ymax_);
  bound_vec_.push_back(zmin_);
  bound_vec_.push_back(zmax_);

  ros::ServiceServer service = nodeHandle_.advertiseService("/get_surface", &SurfaceReconstructionSrv::callGetSurface, this);
  cameraPoseStateSubscriber_ = nodeHandle_.subscribe<geometry_msgs::PoseStamped>("/aruco/pose", 1, &SurfaceReconstructionSrv::CameraPoseCallback, this);
  camera_pose_pub_ = nodeHandle_.advertise<geometry_msgs::PoseStamped>("/camera_pose", 1, true);

  ROS_INFO("Ready to define surface.");
  ros::spin();
}

SurfaceReconstructionSrv::~SurfaceReconstructionSrv()
{
}

void SurfaceReconstructionSrv::CameraPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
	camera_pose_ = *msg;
	camera_pose_.header.frame_id = object_frame_;

	camera_pose_.pose.position.x *= -0.01;
	camera_pose_.pose.position.y *= -0.01;
	camera_pose_.pose.position.z *= -0.01;

	const ros::Time time = ros::Time::now();
	Eigen::Quaterniond opencv_to_ros(0, 1, 0, 0);
	Eigen::Quaterniond link_to_rgb_optical_quat(0.5, -0.5, 0.5, -0.5);
	Eigen::Quaterniond camera_to_marker(camera_pose_.pose.orientation.w, camera_pose_.pose.orientation.x, camera_pose_.pose.orientation.y, camera_pose_.pose.orientation.z);
	Eigen::Quaterniond marker_to_camera = camera_to_marker.inverse();
	Eigen::Vector3d marker_to_camera_pos(camera_pose_.pose.position.x, camera_pose_.pose.position.y, camera_pose_.pose.position.z);
	Eigen::Vector3d marker_to_link_pos = marker_to_camera_pos - Eigen::Vector3d(0.05, 0.005, 0);
	Eigen::Quaterniond marker_to_link = marker_to_camera * link_to_rgb_optical_quat.inverse();

	Eigen::Isometry3d matrix = Eigen::Translation3d(marker_to_link_pos) * marker_to_link * opencv_to_ros;
	Eigen::Matrix4d& m_ = matrix.matrix();

	geometry_msgs::PoseStamped camera_pose_2;
	camera_pose_2 = camera_pose_;
	camera_pose_2.header.frame_id = object_frame_;
	geometry_msgs::Pose pose_only;
	Eigen::Quaterniond quat(matrix.rotation());
//	quat = quat.inverse();
	pose_only.position.x = m_.col(3)[0];
	pose_only.position.y = m_.col(3)[1];
	pose_only.position.z = m_.col(3)[2];
	pose_only.orientation.x = quat.x();
	pose_only.orientation.y = quat.y();
	pose_only.orientation.z = quat.z();
	pose_only.orientation.w = quat.w();
//	tf::poseEigenToMsg(matrix, pose_only); // somehow didn't work
	camera_pose_2.pose = pose_only;

	static tf::TransformBroadcaster br;
	tf::Transform transform;
	tf::poseMsgToTF(camera_pose_2.pose, transform);
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), object_frame_ , "camera_link"));
	camera_pose_pub_.publish(camera_pose_2);

}

bool SurfaceReconstructionSrv::callGetSurface(DetectObject::Request &req, DetectObject::Response &resp) {

	std::string model_name = req.models_to_detect[0].data;
	std::cout << "service called! " << model_name << std::endl;
	save_path_ = ros::package::getPath(save_package_) + "/models/" + model_name;
	std::cout << "save path is updated: " << save_path_ << std::endl;

	PointCloud<PointType>::Ptr cloud_ptr(new PointCloud<PointType>);
	PointCloud<PointXYZ>::Ptr cloud_ptr_xyz(new PointCloud<PointXYZ>);
	PointCloud<Normal>::Ptr cloud_normals(new PointCloud<Normal>);
	PointCloud<PointType>::Ptr keypoint_cloud_ptr (new PointCloud<PointType>);
	PointCloud<FPFHSignature33>::Ptr FPFH_signature_scene(new PointCloud<FPFHSignature33>);
	PointCloud<ReferenceFrame>::Ptr FPFH_LRF_scene(new PointCloud<ReferenceFrame>);
	PointCloud<PointType>::Ptr cloud_transformed(new PointCloud<PointType>);

	// loadPCD or pcd
	if (use_saved_pc_) {
		std::string file_name = save_path_ + "/pcd/Preprocessed_0.pcd";
		if (io::loadPCDFile<PointType>(file_name, *cloud_ptr) == -1)
			PCL_ERROR("Couldn't read pcd file \n");
	} else {

		// Sample clouds
		ros::Subscriber sub = nodeHandle_.subscribe(point_cloud_topic_, 1, &SurfaceReconstructionSrv::saveCloud, this);
		ros::Rate r(60);
		std::cout << "Waiting for point cloud images..." << std::endl << std::endl;

		while (cloud_vector_.size() < number_of_median_clouds_ + number_of_average_clouds_) {
			//    ROS_INFO_STREAM("cloud vector num: " << cloud_vector_.size());
			ros::spinOnce();
			r.sleep();
		}
		std::cout << "Clouds are sampled." << "  width = " << cloud_vector_[0].width << "  height = " << cloud_vector_[0].height << "  size = " << cloud_vector_[0].size() << std::endl;
		std::string path;
		if (save_clouds_) {
			path = save_path_ + "/cloud_raw.ply";
			io::savePLYFile(path, cloud_vector_[0]);
		}

		preprocess(cloud_ptr);
		if(reorient_cloud_) reorientModel(cloud_ptr, cloud_transformed);
		else  *cloud_transformed = *cloud_ptr;
		DownSample(cloud_transformed);
	}

	// keypoints are not working with matching...
	if (!use_all_points_)  computeKeypoints(cloud_transformed, keypoint_cloud_ptr);
	else *keypoint_cloud_ptr = *cloud_transformed;

	computeNormals(keypoint_cloud_ptr, cloud_normals);

	computeFPFHDescriptor(cloud_transformed, keypoint_cloud_ptr, cloud_normals, FPFH_signature_scene);

	computeFPFHLRFs(cloud_transformed, keypoint_cloud_ptr, cloud_normals, FPFH_LRF_scene);

	// region growing
	copyPointCloud(*keypoint_cloud_ptr, *cloud_ptr_xyz);
	std::cout << "xyz point has " << cloud_ptr_xyz->points.size() << " points." << std::endl;
	regionGrowing(cloud_ptr_xyz, cloud_normals);
	regionGrowingRGB(keypoint_cloud_ptr, cloud_normals);

	std::cout << "service done!" << std::endl;

	return true;
}

bool SurfaceReconstructionSrv::reorientModel(PointCloud<PointType>::Ptr cloud_ptr_, PointCloud<PointType>::Ptr cloud_transformed_)
{

	ROS_INFO("reorient_cloud");

	std::string path = save_path_ + "/cloud_transformed.ply";

	ROS_INFO("reorient_cloud");
	const ros::Time time = ros::Time::now();
    tf::StampedTransform transform;
	try {
		const ros::Duration timeout(1);
		const ros::Duration polling_sleep_duration(4);
		std::string* error_msg = NULL;
		tf_listener_.waitForTransform(object_frame_, camera_frame_, time, timeout, polling_sleep_duration, error_msg);
        tf_listener_.lookupTransform(object_frame_, camera_frame_, ros::Time(0), transform);
        ROS_INFO_STREAM("camera to world: " << transform.getRotation().getX() << " , " << transform.getRotation().getY()
        									<< " , " << transform.getRotation().getZ() << " , " << transform.getRotation().getW());
	} catch (tf2::TransformException &ex) {
		ROS_WARN("%s", ex.what());
		ros::Duration(1.0).sleep();
	}

	Eigen::Quaternionf camera_quat(transform.getRotation().getW(), transform.getRotation().getX(),
			transform.getRotation().getY(), transform.getRotation().getZ()); // w, x, y, z
	Eigen::Vector3f camera_pos(transform.getOrigin().getX(), transform.getOrigin().getY(), transform.getOrigin().getZ());
	Eigen::Affine3f matrix;
	matrix = Eigen::Translation3f(camera_pos) * camera_quat;
	Eigen::Matrix4f& m_ = matrix.matrix();
	transformPointCloud(*cloud_ptr_, *cloud_transformed_, m_);

	return true;
}

bool SurfaceReconstructionSrv::projectCloud(PointCloud<PointType>::Ptr cloud_ptr){
	// projection
	PointCloud<PointType>::Ptr cloud_projected(new PointCloud<PointType>());
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	coefficients->values.resize(4);
	coefficients->values[0] = coefficients->values[1] = 0;
	coefficients->values[2] = 1;
	coefficients->values[3] = 0;

	ProjectInliers<PointType> proj;
	proj.setModelType(pcl::SACMODEL_PLANE);
	proj.setInputCloud(cloud_ptr);
	proj.setModelCoefficients(coefficients);
	proj.filter(*cloud_projected);
	std::string path = save_path_ + "/projected.ply";
	io::savePLYFile(path, *cloud_projected);

	// concatination
	PointCloud<PointType>::Ptr cloud_combined(new PointCloud<PointType>());
	cloud_combined = cloud_ptr;
	*cloud_combined += *cloud_projected;
	path = save_path_ + "/combined.ply";
	io::savePLYFile(path, *cloud_combined);

	PointCloud<Normal>::Ptr cloud_normals_(new PointCloud<Normal>());
	PointCloud<PointXYZRGBNormal>::Ptr cloud_smoothed_normals(new PointCloud<PointXYZRGBNormal>());

	std::cout << "combine points and normals" << std::endl;
	computeNormals(cloud_combined, cloud_normals_);
	concatenateFields(*cloud_combined, *cloud_normals_, *cloud_smoothed_normals);
	poisson(cloud_smoothed_normals);

	return true;
}


bool SurfaceReconstructionSrv::preprocess(PointCloud<PointType>::Ptr preprocessed_cloud_ptr_)
{
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
//  std::string path = save_path_ + "/Preprocessed_0.ply";
//  io::savePLYFile(path, *preprocessed_cloud_ptr_);
  std::cout << "object detection, preprocess() " << preprocessed_size_ << std::endl;
  return true;
}

bool SurfaceReconstructionSrv::DownSample(PointCloud<PointType>::Ptr &cloud_)
{
  ROS_INFO("Downsampled");
  ROS_INFO_STREAM("num of samples before down sample: " << cloud_->size());

  PointCloud<PointType>::Ptr cloud_downsampled(new PointCloud<PointType>());
  VoxelGrid<PointType> sor;
  sor.setInputCloud(cloud_);
  sor.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  sor.filter(*cloud_downsampled);
  *cloud_ = *cloud_downsampled;
  std::string path = save_path_ + "/Downsampled_0.ply";
  io::savePLYFile(path, *cloud_downsampled);

  path = save_path_ + "/Keypoints_0.pcd";  // same as Downsampled.ply
  io::savePCDFile(path, *cloud_downsampled);

  path = save_path_ + "/Preprocessed_0.pcd";  // same as Downsampled.ply
  io::savePCDFile(path, *cloud_downsampled);

  ROS_INFO_STREAM("num of samples after down sample: " << cloud_downsampled->size());
  return true;
}

bool SurfaceReconstructionSrv::computeNormals(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<Normal>::Ptr &normals_)
{
  ROS_INFO("compute normals");
  NormalEstimation<PointType, Normal> n;
  search::KdTree<PointType>::Ptr tree(new search::KdTree<PointType>);
  tree->setInputCloud(cloud_);
  n.setInputCloud(cloud_);
  n.setSearchMethod(tree);
  n.setRadiusSearch(0.01);
  n.compute(*normals_);


  ROS_INFO_STREAM("normal size: " << normals_->size());
  if(normal_flip_){
	  for(int i =0; i < normals_->size(); i++){
	//    ROS_INFO_STREAM("before: " << normals_->at(i).normal_x);
		  if(normals_->at(i).normal_z  < 0){
				normals_->at(i).normal_x *= -1.0;
				normals_->at(i).normal_y *= -1.0;
				normals_->at(i).normal_z *= -1.0;
		  }
	//    ROS_INFO_STREAM("after: " << normals_->at(i).normal_x);
  	  }
  }

  //visualization
  std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer = normalsVis(cloud_, normals_);
  ros::Rate r(40);
  while (!viewer->wasStopped())
  {
    viewer->spinOnce ();
    r.sleep();
  }
  std::cout<<"Stopped the Viewer"<<std::endl;

  return true;
}

bool SurfaceReconstructionSrv::computeKeypoints(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<PointType>::Ptr &keypoint_model_ptr_)
{
  ROS_INFO("compute Keypoints");

  std::cout << "size of the input file points: " << cloud_->size() << std::endl;
  ISSKeypoint3D<PointType, PointType> iss_detector;
  search::KdTree<PointType>::Ptr tree(new search::KdTree<PointType>());

  iss_detector.setSearchMethod(tree);
  iss_detector.setSalientRadius(0.02);  //
  iss_detector.setNonMaxRadius(0.1 * 0.02);
//  iss_detector.setNormalRadius(0.01); // -> somehow it worked after it's removed!
//  iss_detector.setBorderRadius(1.1 * 0.02); // -> somehow it worked after it's removed!
  iss_detector.setThreshold21(0.985);
  iss_detector.setThreshold32(0.985);
  iss_detector.setMinNeighbors(5);
  iss_detector.setNumberOfThreads(4);

  iss_detector.setInputCloud(cloud_);
  ROS_INFO("iss detector set input");
  iss_detector.compute(*keypoint_model_ptr_);
//  ROS_INFO_STREAM("output size: " << keypoint_model_ptr_->size());

  std::cout << "size of the input file points: " << keypoint_model_ptr_->size() << std::endl;
  std::string path = save_path_ + "/Keypoints_0.ply";
  io::savePLYFile(path, *keypoint_model_ptr_);
  return true;
}

bool SurfaceReconstructionSrv::computeFPFHDescriptor(const 	PointCloud<PointType>::ConstPtr &cloud_,
															PointCloud<PointType>::Ptr &keypoint_model_ptr_,
															PointCloud<Normal>::Ptr &normals_,
															PointCloud<FPFHSignature33>::Ptr FPFH_signature_scene_)
{
  FPFHEstimation<PointType, Normal, FPFHSignature33> fpfh;
  search::KdTree<PointType>::Ptr search_method(new search::KdTree<PointType>);
  PointIndicesPtr indices = boost::shared_ptr<PointIndices>(new PointIndices());

  // defining indices
  for (int k = 0; k < keypoint_model_ptr_->size(); k++) {
    indices->indices.push_back(k);
  }

  fpfh.setSearchMethod(search_method);
  fpfh.setIndices(indices);
  fpfh.setInputCloud(keypoint_model_ptr_);
  fpfh.setSearchSurface(cloud_);
  fpfh.setInputNormals(normals_);
  fpfh.setRadiusSearch(0.02);  //FPFH_radius_
  fpfh.compute(*FPFH_signature_scene_);

  std::string path = save_path_ + "/Signature_0.ply";
  io::savePLYFile(path, *FPFH_signature_scene_);
  return true;
}

bool SurfaceReconstructionSrv::computeFPFHLRFs(const PointCloud<PointType>::ConstPtr &cloud_,
		PointCloud<PointType>::Ptr &keypoint_model_ptr_,
		PointCloud<Normal>::Ptr &normals_,
                                               PointCloud<ReferenceFrame>::Ptr FPFH_LRF_scene_)
{
  BOARDLocalReferenceFrameEstimation<PointType, Normal, ReferenceFrame> rf_est;
  rf_est.setFindHoles(true);
  double lrf_search_radius_ = 0.01;
  rf_est.setRadiusSearch(lrf_search_radius_);
  rf_est.setInputCloud(keypoint_model_ptr_);
  rf_est.setInputNormals(normals_);
  rf_est.setTangentRadius(lrf_search_radius_);
  rf_est.setSearchSurface(cloud_);
  rf_est.compute(*FPFH_LRF_scene_);
  std::string path = save_path_ + "/LRFs_0.ply";
  io::savePLYFile(path, *FPFH_LRF_scene_);

  return true;
}

bool SurfaceReconstructionSrv::regionGrowing(const PointCloud<PointXYZ>::ConstPtr &cloud_, PointCloud<Normal>::Ptr &normals_)
{
  std::cout << "begin region growing" << std::endl;

  search::Search<PointXYZ>::Ptr tree = boost::shared_ptr<search::Search<PointXYZ> >(new search::KdTree<PointXYZ>);
  PointCloud<Normal>::Ptr local_normals(new PointCloud<Normal>);
  NormalEstimation<PointXYZ, Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud_);
  normal_estimator.setKSearch(50);
  normal_estimator.compute(*local_normals);

  IndicesPtr indices(new std::vector<int>);
  PassThrough<PointXYZ> pass;
  pass.setInputCloud(cloud_);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, 1.0);
  pass.filter(*indices);

  RegionGrowing<PointXYZ, Normal> reg;
  reg.setMinClusterSize(500);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(30);
  reg.setInputCloud(cloud_);
  //reg.setIndices (indices);
  reg.setInputNormals(local_normals);
  reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(1.0);

  std::vector<PointIndices> clusters;
  reg.extract(clusters);

  if (clusters.size() > 0) {
    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size() << " points." << endl;
    std::cout << "These are the indices of the points of the initial" << std::endl << "cloud that belong to the first cluster:" << std::endl;

    PointCloud<PointType>::Ptr cloud_filtered = reg.getColoredCloud();
    int j = 0;
    for (std::vector<PointIndices>::const_iterator it = clusters.begin(); it != clusters.end(); ++it) {
      PointCloud<PointType>::Ptr cloud_cluster(new PointCloud<PointType>);
      for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        cloud_cluster->points.push_back(cloud_filtered->points[*pit]);
      cloud_cluster->width = cloud_cluster->points.size();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;

      std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
      std::string path = save_path_ + "/Regiongrow" + std::to_string(j) + ".ply";
      io::savePLYFile(path, *cloud_cluster);
      j++;
    }
    return true;
  } else {
    ROS_ERROR("region growing didn't find cluster");
    return false;
  }

}

bool SurfaceReconstructionSrv::regionGrowingRGB(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<Normal>::Ptr &normals_)
{
  std::cout << "begin rgb region growing" << std::endl;

  IndicesPtr indices(new std::vector<int>);
  PassThrough<PointType> pass;
  pass.setInputCloud(cloud_);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, 1.0);
  pass.filter(*indices);
  pass.setFilterLimits(0.0, 1.0);
  pass.filter(*indices);

  RegionGrowingRGB<PointType> reg;
  reg.setInputCloud(cloud_);
  reg.setIndices(indices);
  search::Search<PointType>::Ptr tree_rgb(new search::KdTree<PointType>);
  reg.setSearchMethod(tree_rgb);
  reg.setDistanceThreshold(5);
  reg.setPointColorThreshold(6);
  reg.setRegionColorThreshold(3);
  reg.setMaxClusterSize(100000);
  reg.setMinClusterSize(1000);

  std::vector<PointIndices> clusters;
  reg.extract(clusters);

  if (clusters.size() > 0) {
    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size() << " points." << endl;
    std::cout << "These are the indices of the points of the initial" << std::endl << "cloud that belong to the first cluster:" << std::endl;

    PointCloud<PointType>::Ptr cloud_filtered = reg.getColoredCloud();
    int j = 0;
    for (std::vector<PointIndices>::const_iterator it = clusters.begin(); it != clusters.end(); ++it) {
      PointCloud<PointType>::Ptr cloud_cluster(new PointCloud<PointType>);
      for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        cloud_cluster->points.push_back(cloud_filtered->points[*pit]);
      cloud_cluster->width = cloud_cluster->points.size();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;

      std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
      std::string path = save_path_ + "/RGBregiongrow" + std::to_string(j) + ".ply";
      io::savePLYFile(path, *cloud_cluster);
      j++;
    }
    return true;

  } else {
    ROS_ERROR("region growing didn't find cluster");
    return false;
  }
}

bool SurfaceReconstructionSrv::poisson(const PointCloud<PointXYZRGBNormal>::Ptr &cloud_smoothed_normals)
{
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
//  save_path_ = ros::package::getPath("urdf_models") + "/models/" + model_name;
  std::string path = save_path_ + "/poisson.ply";
  io::savePLYFile(path, mesh);

  return true;
}

std::shared_ptr<visualization::PCLVisualizer> SurfaceReconstructionSrv::normalsVis(PointCloud<PointType>::Ptr &cloud, PointCloud<Normal>::Ptr &normals)
{
  std::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(1.0, 1.0, 1.0);
  viewer->addPointCloud<PointType>(cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "sample cloud");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");

  viewer->addPointCloudNormals<PointType, Normal>(cloud, normals, 10, 0.05, "normals");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "normals");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_LINE_WIDTH, 0.01, "normals");

  viewer->addCoordinateSystem(0.1);
  viewer->setCameraPosition(0, 0, 0.1, 0.5, 0.0, 0.0, 0.1);
  viewer->initCameraParameters();
  return (viewer);
}

std::shared_ptr<visualization::PCLVisualizer> SurfaceReconstructionSrv::rgbVis(PointCloud<PointType>::ConstPtr cloud)
{
  std::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(1.0, 1.0, 1.0);
  visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
  viewer->addPointCloud<PointType>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem(0.1);
  viewer->setCameraPosition(0, 0, 0.0, 0.0, 0.0, 0.0, 0.1);
  viewer->initCameraParameters();
  return (viewer);
}

std::shared_ptr<visualization::PCLVisualizer> SurfaceReconstructionSrv::xyzVis(PointCloud<PointXYZ>::ConstPtr cloud)
{
  std::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(1.0, 1.0, 1.0);
//  visualization::PointCloudColorHandlerRGBField<PointXYZ> rgb(cloud);
  viewer->addPointCloud<PointXYZ>(cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "sample cloud");
  viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem(0.1, "sample cloud");
  viewer->setCameraPosition(0, 0, 0.0, 0.0, 0.0, 0.0, 0.1);
  viewer->initCameraParameters();
  return (viewer);
}

std::shared_ptr<visualization::PCLVisualizer> SurfaceReconstructionSrv::normalsVis (
    const PointCloud<PointType>::ConstPtr &cloud, PointCloud<Normal>::Ptr &normals)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  std::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
//  pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
  pcl::visualization::PointCloudColorHandlerCustom<PointType>
     rgb_color(cloud, 255, 0, 0);
  viewer->addPointCloud<PointType> (cloud, rgb_color, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

  viewer->addPointCloudNormals<PointType, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
  viewer->addCoordinateSystem (0.1);
  viewer->initCameraParameters ();
  return (viewer);
}

void SurfaceReconstructionSrv::saveCloud(const sensor_msgs::PointCloud2& cloud)
{
  std::cout << "the length of cloud sensor msg is " << cloud.data[100] << std::endl;
  std::cout << "cloud frame" << cloud.header.frame_id << std::endl;


  PointCloud<PointType> new_cloud, could_transformed;
  fromROSMsg(cloud, new_cloud);
//  const ros::Time time = ros::Time::now();
//  Eigen::Quaternionf camera_quat(0, 0.707, 0, -0.707);
//  Eigen::Vector3f camera_pos(0.047, -0.068, 0.917);
//
//  Eigen::Affine3f matrix;
//  matrix = Eigen::Translation3f(camera_pos) * camera_quat;
//  Eigen::Matrix4f& m_ = matrix.matrix();
//  tf::StampedTransform transform;
//
//  try {
//    tf_listener_.lookupTransform(world_frame_, camera_frame_, time, transform);
////    tf_listener_.waitForTransform(world_frame_, camera_frame_, time, timeout, polling_sleep_duration, error_msg);
//    ROS_INFO_STREAM("tf_transform" << transform.frame_id_);
////    << " , " << transform.getBasis() << " , " << transform.getRotation());
//    transformPointCloud(new_cloud, could_transformed, m_.inverse());
//
//  } catch (tf2::TransformException &ex) {
//    ROS_WARN("%s", ex.what());
//    ros::Duration(1.0).sleep();
//  }

  cloud_vector_.push_back(new_cloud);
}

}/*end namespace*/
