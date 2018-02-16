/*
 * surface_reconstruction_srv.hpp
 *
 *  Created on: Apr 29, 2016
 *      Author: hironori yoshida
 */

#ifndef INCLUDE_OBJECT_DETECTION_SURFACE_RECONSTRUCTION_SERVICE_HPP_
#define INCLUDE_OBJECT_DETECTION_SURFACE_RECONSTRUCTION_SERVICE_HPP_

#include <vector>
#include <pcl/PCLPointCloud2.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>

#include <std_srvs/Empty.h>
//#include <branch_surface/>

// keypoint descriptors
#include <pcl/keypoints/iss_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/board.h>

#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf/transform_listener.h>

#include <branch_surface/DetectObject.h>

using namespace pcl;
typedef PointXYZRGB PointType;
typedef branch_surface::DetectObject DetectObject;

namespace surface_reconstruction_srv {
/*!
 * Reads PointClouds from topic /camera/depth/points and preprocesses them with temporal median and average filters.
 */
class SurfaceReconstructionSrv {

//  typedef PointType PointType;

public:

	/*!
	 * Constructor.
	 * @param nodeHandle the ROS node handle.
	 */
	SurfaceReconstructionSrv(ros::NodeHandle nodeHandle);

	/*!
	 * Destructor.
	 */
	virtual ~SurfaceReconstructionSrv();

private:

	bool callGetSurface(DetectObject::Request &req, DetectObject::Response &resp);
	void saveCloud(const sensor_msgs::PointCloud2& cloud);
	std::shared_ptr<visualization::PCLVisualizer> normalsVis(PointCloud<PointType>::Ptr &cloud, PointCloud<Normal>::Ptr &normals);
	std::shared_ptr<visualization::PCLVisualizer> rgbVis(PointCloud<PointType>::ConstPtr cloud);
	std::shared_ptr<visualization::PCLVisualizer> xyzVis(PointCloud<PointXYZ>::ConstPtr cloud);

private:
	bool planarSegmentation(PointCloud<PointType>::Ptr cloud_ptr_);

	bool preprocess(PointCloud<PointType>::Ptr preprocessed_cloud_ptr_);

	bool DownSample(PointCloud<PointType>::Ptr &cloud_);

	bool computeNormals(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<Normal>::Ptr &normals_);

	bool regionGrowing(const PointCloud<PointXYZ>::ConstPtr &cloud_, PointCloud<Normal>::Ptr &normals_);

	bool regionGrowingRGB(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<Normal>::Ptr &normals_);

	bool poisson(const PointCloud<PointXYZRGBNormal>::Ptr &cloud_smoothed_normals);

	bool computeKeypoints(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<PointType>::Ptr &keypoint_model_ptr_);

	bool computeFPFHDescriptor(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<PointType>::Ptr &keypoint_model_ptr_, PointCloud<Normal>::Ptr &normals_, PointCloud<FPFHSignature33>::Ptr FPFH_signature_scene_);

	bool computeFPFHLRFs(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<PointType>::Ptr &keypoint_model_ptr_, PointCloud<Normal>::Ptr &normals_, PointCloud<ReferenceFrame>::Ptr FPFH_LRF_scene__);

	ros::NodeHandle nodeHandle_;

	std::vector<PointCloud<PointType>> cloud_vector_;
	search::KdTree<PointType>::Ptr tree_;

	float leaf_size_;
	std::string model_folder_;

	// Segmentation
	double bin_size_;
	double inlier_dist_segmentation_;
	double segmentation_inlier_ratio_;

	int max_number_of_instances_;
	double max_fitness_score_;
	double inlier_dist_icp_;
	double icp_inlier_threshold_;
	double min_inlier_ratio_validation_;
	double inlier_dist_validation_;

	int number_of_average_clouds_;
	int number_of_median_clouds_;
	double z_threshold_;
	double planarSegmentationTolerance_;
	int min_number_of_inliers_;
	double xmin_;
	double xmax_;
	double ymin_;
	double ymax_;
	double zmin_;
	double zmax_;

	bool use_saved_pc_;
	bool compute_keypoints_;
	bool save_clouds_;
	std::string save_path_;
  std::string save_package_;


  tf::TransformListener tf_listener_;
  std::string world_frame_;
  std::string camera_frame_;

	int grid_res_;
	std::vector<double> bound_vec_;
	std::string point_cloud_topic_;

	//Keypoint Detection Parameters
//	double normal_radius_;
//	double salient_radius_;
//	double border_radius_;
//	double non_max_radius_;
//	double gamma_21_;
//	double gamma_32_;
//	double min_neighbors_;
//	int threads_;
//  bool use_all_points_;

// Clustering
//	bool use_hough_;
	double bin_size_hough_;
//	double threshold_hough_;
//	double bin_size_gc_;
//	double threshold_gc_;
};

}/* end namespace object_detection_srv */

#endif /* INCLUDE_OBJECT_DETECTION_OBJECT_DETECTION_SERVICE_HPP_ */
