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

#include <std_srvs/Empty.h>
using namespace pcl;
typedef pcl::PointXYZRGB PointType;

namespace surface_reconstruction_srv {
/*!
 * Reads PointClouds from topic /camera/depth/points and preprocesses them with temporal median and average filters.
 */
class SurfaceReconstructionSrv {

//  typedef pcl::PointType PointType;

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

	bool callGetSurface(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &resp);
	void saveCloud(const sensor_msgs::PointCloud2& cloud);
	std::shared_ptr<visualization::PCLVisualizer> normalsVis(PointCloud<PointType>::Ptr &cloud, PointCloud<Normal>::Ptr &normals);
	std::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);

private:

	bool preprocess(PointCloud<PointXYZ>::Ptr preprocessed_cloud_ptr_);

	bool DownSample(PointCloud<PointType>::Ptr &cloud_);

	bool computeNormals(const PointCloud<PointType>::ConstPtr &cloud_, PointCloud<Normal>::Ptr &normals_);

	ros::NodeHandle nodeHandle_;

	std::vector<PointCloud<PointXYZRGB> > colored_cloud_vector_;
    std::vector<PointCloud<PointType>> cloud_vector_;


	float leaf_size_;
	std::string model_folder_;

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

	int grid_res_;
	std::vector<double> bound_vec_;
	std::string point_cloud_topic_;

};

}/* end namespace object_detection_srv */

#endif /* INCLUDE_OBJECT_DETECTION_OBJECT_DETECTION_SERVICE_HPP_ */
