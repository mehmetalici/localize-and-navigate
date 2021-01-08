#ifndef LIDAR_SEG_H
#define LIDAR_SEG_H

//Definitions
#define PAN_RANGE	70.6//57.5  //Camera Horizontal FOV
#define TILT_RANGE  60.0//43.5  //Camera Vertical FOV

#define RGB_PAN_RANGE 84.1
#define RGB_TILT_RANGE 53.8

#define RGB_FOCAL_X     525.0
#define RGB_FOCAL_Y     525.0
#define RGB_C_X         319.5
#define RGB_C_Y         239.5

#define DEPTH_FOCAL_X     575.8157348632812
#define DEPTH_FOCAL_Y     575.8157348632812
#define DEPTH_C_X         314.5
#define DEPTH_C_Y         235.5


#include <pcl/point_types.h>
//#include <pcl/common/point_operators.h>
#include <pcl/common/io.h>
#include <pcl/common/centroid.h>
#include <vector>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/opencv.hpp>

#include <chrono> 
#include <ctime>
using namespace std;
using namespace pcl;


//Kinect Translation and Rotation matrix
const float RDepthAxisToRgbAxis[9] =
{
         9.99915096e-01,  1.30030094e-02, -8.50065837e-04,
         -1.30146237e-02,  9.99795239e-01, -1.54951321e-02,
         6.48408430e-04,  1.55048798e-02,  9.99879582e-01,
};

const float TDepthAxisToRgbAxis[3] =
{
       0.4536562,-0.19260606,0.24194937,
};



class lidar_seg{

public:

lidar_seg(double pan_res, double tilt_res); //constructors

void set_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr  in_cloud );
void set_cloud( cv::Mat depth);
void set_params(double pan_eps, double tilt_eps, double dist_eps, int min_points);
void segment();
void segment(cv::Mat depth, cv::Mat rgb);
void take_colored_cloud(pcl::PointCloud<pcl::PointXYZRGB> &colored_cloud);

private:


void get_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr  in_cloud);
void get_cons(double pan_res, double tilt_res);
void get_params(double pan_eps, double tilt_eps, double dist_eps, int min_points);
void to_sphere() ;
void to_sphere(cv::Mat depth, cv::Mat rgb);
void dist(int index, int core_index, std::vector<int> &result_neigh);
void boundingbox(); 
vector<int> region_query(int iterator,bool core);
vector<vector<int> > index_vec;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

pcl::PointCloud<pcl::PointXYZRGB> sphere_cloud ;


double pan_resolution, tilt_resolution, pan_epsilon, tilt_epsilon, dist_epsilon;
int seg_min ;
vector<bool> visited;
vector<int> clusters;
int pan_direction, tilt_direction;
int min_cluster_var;
int total_pan;
int total_tilt;
int cluster = 0;
int total_cluster;

Eigen::MatrixXd range_image;


Eigen::Matrix4f DepthtoRGBMat;

};





#endif
