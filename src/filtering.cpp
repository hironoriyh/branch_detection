/*
 * PointCloudPreprocessing.cpp
 *
 *  Created on: Apr. 2, 2016
 *      Author: Yves Zimmermann
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#include "../include/branch_surface/filtering.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

namespace point_cloud_filtering {

filtering::filtering()
    :  number_of_average_clouds_(0),
       number_of_median_clouds_(1),
       cloud_width_(0),
       cloud_height_(0),
       cloud_size_(0),
       cloud_vector_(),
       z_threshold_(0.02),
       planarSegmentationTolerance_(0.02),
       min_number_of_inliers_(1),
       xmin_(-1),
       xmax_(1),
       ymin_(-1),
       ymax_(1),
       zmin_(0.01),
       zmax_(2),
       full_output_(true)
{
}

filtering::~filtering()
{
}

bool filtering::compute(pcl::PointCloud<PointType>::Ptr preprocessed_cloud_ptr){
if(number_of_average_clouds_ + number_of_median_clouds_ > cloud_vector_.size()){
  std::cout << "There are too few clouds in the input vector, for these filter parameters!" << std::endl;
      return false;
}

if(!medianFilter(*preprocessed_cloud_ptr))
  return false;

if(!averageFilter(*preprocessed_cloud_ptr))
  return false;

if(!planarSegmentation(preprocessed_cloud_ptr))
  if (full_output_)
  std::cout << "Couldn't find a plane!" << std::endl;

  return true;
}

bool filtering::medianFilter(pcl::PointCloud<PointType>& median_cloud){

  //get the cloud dimensions
  if (cloud_width_ == 0 || cloud_height_ == 0){
  cloud_width_ = cloud_vector_[0].width;
  cloud_height_ = cloud_vector_[0].height;
  cloud_size_ = cloud_width_ * cloud_height_;
  }

  double z_values [number_of_median_clouds_];
  double z_sorted [number_of_median_clouds_];

  for(int i_point = 0; i_point < cloud_size_; i_point++)
  {
    for (int j_cloud = 0; j_cloud < number_of_median_clouds_; j_cloud++)
    {
      z_values[j_cloud] = cloud_vector_[j_cloud].points[i_point].z;
      z_sorted[j_cloud] = z_values[j_cloud];
    }
    std::sort(z_sorted, z_sorted + number_of_median_clouds_);
    for (int k = 0; k < number_of_median_clouds_; k++)
    {
      if(z_sorted[(number_of_median_clouds_-1)/2] == z_values[k])
      {
        cloud_vector_[0].points[i_point] = cloud_vector_[k].points[i_point];
      }
    }
  }
  median_cloud = cloud_vector_[0];

  if (full_output_)
  std::cout << "Took median from " << number_of_median_clouds_ << " clouds." << std::endl;

  return true;
}

bool filtering::averageFilter(pcl::PointCloud<PointType>& base_cloud){

//get the cloud dimensions
  if (cloud_width_ == 0 || cloud_height_ == 0){
  cloud_width_ = cloud_vector_[0].width;
  cloud_height_ = cloud_vector_[0].height;
  cloud_size_ = cloud_width_ * cloud_height_;
  }

  pcl::PointCloud<PointType> unorganized_cloud;

  for (int k = 0; k < number_of_average_clouds_; ++k)
    {
        //iterate over points
        for (int i = 0; i < cloud_width_; i++)
        {
          for (int j = 0; j < cloud_height_; j++)
          {
            PointType point_base_cloud = base_cloud(i,j);
            PointType point_new_cloud = cloud_vector_[k + number_of_median_clouds_](i,j);

            if (point_base_cloud.z != 0 && point_new_cloud.z != 0 && std::abs(point_base_cloud.z - point_new_cloud.z) < z_threshold_)
            {
              point_base_cloud.x = (point_base_cloud.x * (k) + point_new_cloud.x)/(k+1);
              point_base_cloud.y = (point_base_cloud.y * (k) + point_new_cloud.y)/(k+1);
              point_base_cloud.z = (point_base_cloud.z * (k) + point_new_cloud.z)/(k+1);
//              point_base_cloud.r = (point_base_cloud.r * (k) + point_new_cloud.r)/(k+1);
//              point_base_cloud.g = (point_base_cloud.g * (k) + point_new_cloud.g)/(k+1);
//              point_base_cloud.b = (point_base_cloud.b * (k) + point_new_cloud.b)/(k+1);

            }
//            else if (point_new_cloud.z != 0 && point_base_cloud.z == 0)
//            {
//              point_base_cloud.x = point_new_cloud.x;
//              point_base_cloud.y = point_new_cloud.y;
//              point_base_cloud.z = point_new_cloud.z;
//              point_base_cloud.r = point_new_cloud.r;
//              point_base_cloud.g = point_new_cloud.g;
//              point_base_cloud.b = point_new_cloud.b;
//            }

            base_cloud(i,j) = point_base_cloud;

            bool ispartofcloud =  point_base_cloud.x >= xmin_ && point_base_cloud.x <= xmax_ &&
                                  point_base_cloud.y >= ymin_ && point_base_cloud.y <= ymax_ &&
                                  point_base_cloud.z >= zmin_ && point_base_cloud.z <= zmax_;
            if (k == number_of_average_clouds_-1 && ispartofcloud)
            {
              unorganized_cloud.push_back(point_base_cloud);
            }

          }
        }
    }
  // write unorganized cloud
  base_cloud.clear();
  base_cloud = unorganized_cloud;
  if (full_output_){
  std::cout << "Averaged with " << number_of_average_clouds_ << " clouds." << std::endl;
  std::cout << "Reduced cloud by clipping to  " << base_cloud.size() << " points." << std::endl;
  }
  return true;
}

bool filtering::planarSegmentation(pcl::PointCloud<PointType>::Ptr preprocessed_cloud_ptr){

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  // Create the segmentation object
  pcl::SACSegmentation<PointType> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (planarSegmentationTolerance_);
  seg.setInputCloud (preprocessed_cloud_ptr);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () < min_number_of_inliers_)
  {
    if (full_output_)
    std::cout << "Could not find a plane with more than " << min_number_of_inliers_ << " inliers." << std::endl;
  }
  else
  {
      //Move inliers to zero
      for (int i = 0; i < inliers->indices.size(); i++)
      {
        preprocessed_cloud_ptr->points[inliers->indices[i]].x = 0;
        preprocessed_cloud_ptr->points[inliers->indices[i]].y = 0;
        preprocessed_cloud_ptr->points[inliers->indices[i]].z = 0;
      }

      //Remove inliers from cloud
      pcl::PointCloud<PointType>::Ptr segmentedCloud (new pcl::PointCloud<PointType>);
      for (int i_point = 0; i_point < preprocessed_cloud_ptr->size(); i_point++)
      {
        if (preprocessed_cloud_ptr->points[i_point].z != 0)
          segmentedCloud->push_back(preprocessed_cloud_ptr->points[i_point]);
      }
      preprocessed_cloud_ptr->points = segmentedCloud->points;
      preprocessed_cloud_ptr->width = segmentedCloud->width;
      preprocessed_cloud_ptr->height = segmentedCloud->height;

      if (full_output_)
      std::cout << "Removed " << inliers->indices.size() << " points as part of a plane." << std::endl;
  }

  return true;
}


bool filtering::setNumberOfAverageClouds(int number_of_average_clouds){
  number_of_average_clouds_ = number_of_average_clouds;
  return true;
}


bool filtering::setNumberOfMedianClouds(int number_of_median_clouds){
  if (number_of_median_clouds % 2 == 0){
    std::cout << "ERROR: must input an odd number of clouds" << std::endl;
    return false;
  }
  number_of_median_clouds_ = number_of_median_clouds;
  return true;
}

bool filtering::setInputClouds(std::vector <pcl::PointCloud <PointType> > cloud_vector){
  cloud_vector_ = cloud_vector;
  return true;
}

bool filtering::setClippingBoundaries(std::vector<float> boundaries){
  if (boundaries.size()!=6){
    std::cout << "Error: Wrong number of boundary values." << std::endl;
    return false;
  }

  xmin_ = boundaries[0];
  xmax_ = boundaries[1];
  ymin_ = boundaries[2];
  ymax_ = boundaries[3];
  zmin_ = boundaries[4];
  zmax_ = boundaries[5];

  return true;
}

bool filtering::setZThreshold(float z_threshold){
  z_threshold_ = z_threshold;
  return true;
}

bool filtering::setPlanarSegmentationTolerance(float planarSegmentationTolerance){
  planarSegmentationTolerance_ = planarSegmentationTolerance;
  return true;
}

bool filtering::setMinNumberOfInliers(int min_number_of_inliers){
  min_number_of_inliers_ = min_number_of_inliers;
  return true;
}

bool filtering::setFullOutput(bool full_output){
  full_output_ = full_output;
  return true;
}

} /* namespace */
