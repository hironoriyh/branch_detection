/*
 * filtering.hpp
 *
 *  Created on: Apr. 2, 2016
 *      Author: Yves Zimmermann
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#pragma once

// ROS
#include <ros/ros.h>

// STD
#include <string>
#include <vector>

//PCL
#include <pcl/common/common_headers.h>


typedef pcl::PointXYZRGB PointType;

namespace point_cloud_filtering {

/*!
 * Reads PointClouds from topic /camera/depth/points and preprocesses them with temporal median and average filters.
 */
class filtering
{
 public:

  /*!
   * Constructor.
   * @param nodeHandle the ROS node handle.
   */
  filtering();

  /*!
   * Destructor.
   */
  virtual ~filtering();

  /*!
   * Applies median-, average-, passthrough- and segmentation-filter on the input clouds
   */
  bool compute(pcl::PointCloud<PointType>::Ptr preprocessed_cloud_ptr);

  /*!
   * Sets number of clouds to be averaged
   */
  bool setNumberOfAverageClouds(int number_of_averaged_clouds);

  /*!
   * Sets number of clouds to be used for median
   */
  bool setNumberOfMedianClouds(int number_of_median_clouds);

  /*!
   * Sets the input clouds. Vector length must coincide with sum of clouds for average and median!
   */
  bool setInputClouds(std::vector <pcl::PointCloud <PointType> > cloud_vector);

  /*!
   * Sets the boundaries for passthrough filtering
   */
  bool setClippingBoundaries(std::vector<float> boundaries);

  /*!
   * Sets the Tolerance for planar segmentation
   */
  bool setPlanarSegmentationTolerance(float planarSegmentationTolerance);

  /*!
   * Sets the minimum amount of inliers
   */
  bool setMinNumberOfInliers(int min_number_of_inliers);

  /*!
   * Sets the depth threshold for averaging
   */
  bool setZThreshold(float z_threshold);

  /*!
   * Sets if information should be printed to the command window
   */
  bool setFullOutput(bool full_output);



 private:

  /*!
   * Applies median filter using number_of_clouds and writes resulting PointCloud in median_cloud
   * @return true if successful, false otherwise.
   */
  bool medianFilter(pcl::PointCloud<PointType>& median_cloud);

  /*!
   * Applies average filter on base_cloud using number_of_clouds and writes resulting PointCloud in base_cloud
   */
  bool averageFilter(pcl::PointCloud<PointType>& base_cloud);

  /*!
   * Applies average filter on base_cloud using number_of_clouds and writes resulting PointCloud in base_cloud
   */
  bool planarSegmentation(pcl::PointCloud<PointType>::Ptr preprocessed_cloud_ptr);

 private:

  //! Number of clouds to be used for averaging filter
  int number_of_average_clouds_;

  //! Number of clouds to be used for median filter
  int number_of_median_clouds_;

  //! Number of clouds to be used for median filter
  int cloud_width_;

  //! Number of clouds to be used for median filter
  int cloud_height_;

  //! Number of clouds to be used for median filter
  int cloud_size_;

  //! Threshold of z Position of average Cloud towards base_cloud
  float z_threshold_;

  //! Tolerance for the distance to the segmented plane
  float planarSegmentationTolerance_;

  //! Minimum number of inlayers for segmentation
  int min_number_of_inliers_;

  //! clipping boundary
  float xmin_;

  //! clipping boundary
  float ymin_;

  //! clipping boundary
  float zmin_;

  //! clipping boundary
  float xmax_;

  //! clipping boundary
  float ymax_;

  //! clipping boundary
  float zmax_;

  //! set if information should be printed to command window
  bool full_output_;

  std::vector <pcl::PointCloud <PointType> > cloud_vector_;

};

} /* namespace */
