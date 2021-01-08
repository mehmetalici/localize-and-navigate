// PCL specific includes
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>
#include <fstream>
#include <string>

 #include <ros/ros.h>
#include <ros/callback_queue.h>

#include <pcl/common/time.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>
//#include  <chrono
#include <vector>
#include <ctime>
#include <pcl/common/time.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <Eigen/Geometry>

#include "lidar_seg.h"
#include "ground_segmentation.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

 #include <opencv2/imgproc/imgproc.hpp>
 #include <opencv2/highgui/highgui.hpp>

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat)
#include <opencv2/videoio.hpp>  // Video write

#include <cstdio>
#include <ctime>


 // Include CvBridge, Image Transport, Image msg
 #include <image_transport/image_transport.h>
 #include <cv_bridge/cv_bridge.h>
 #include <sensor_msgs/image_encodings.h>


#include <chrono> 

using namespace std::chrono;
using namespace std;
using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;
using namespace std;
using namespace pcl;
using namespace cv;


// Depth segmentation parameters
double  depth_pan_eps =2;
double  depth_tilt_eps = 5;
double  depth_dist_eps = 0.75;
int     depth_min_p = 50;

//Lidar segmentation parameters
double  lidar_pan_eps = 3;
double  lidar_tilt_eps = 5;
double  lidar_dist_eps = 0.2;
int     lidar_min_p = 5;
double  lidar_pan_res = 0.2;
double  lidar_tilt_res = 1.99;

//Ground segmentation parameters

int num_iter = 10; //number of iteration to estimate plane
int num_lpr = 2000; //number of seed points
double th_seeds = 0.2; // The threshold for seeds
double th_dist = 0.5;
float initial_height = 0.45;// Initial sensor height from the ground plane


Mat CloudtoMat(pcl::PointCloud<pcl::PointXYZRGB> cloud, int height, int width)
{
    Mat image = cv::Mat(height, width, CV_8UC3);
 
#pragma omp parallel for
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            pcl::PointXYZRGB point = cloud[image.cols *y + x];
            image.at<Vec3b>(y, x)[0] = point.b;
            image.at<Vec3b>(y, x)[1] = point.g;
            image.at<Vec3b>(y, x)[2] = point.r;
        }
    }

    return image;
}

Mat GroundCloudToMat(pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud,int width,int height)
{
    Mat image(cv::Size(width, height), CV_16UC1, Scalar(0));

    for (int i = 0; i<ptCloud->size(); i++)
    {

        float x_multiplier = ptCloud->points[i].x / ptCloud->points[i].y;
        float y_multiplier = ptCloud->points[i].z / ptCloud->points[i].y;
        int x = round((x_multiplier)*DEPTH_FOCAL_X + DEPTH_C_X);
        int y = round(-(y_multiplier)*DEPTH_FOCAL_Y + DEPTH_C_Y);
        float z = ptCloud->points[i].y* sqrt(x_multiplier * x_multiplier + y_multiplier * y_multiplier + 1);
        if (0<=y && y < height && 0<= x && x < width)
        {
            image.at<unsigned short>(y, x) = 1000* z;
        }
        
    }
    return image;
}

/*
Author: malici U.
Date: 11/25/2020
Method: Convert Mat object that stores depth information to Point cloud

*/

pcl::PointCloud<pcl::PointXYZ>::Ptr MatToPoinXYZ(cv::Mat depthMat)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud(new pcl::PointCloud<pcl::PointXYZ>);

    // calibration parameters
    float focal_x = depthMat.cols / (2 * tan(PAN_RANGE * M_PI / 360.0));
    float focal_y = depthMat.rows / (2 * tan(TILT_RANGE * M_PI / 360.0));
    unsigned char* data =(depthMat.data);
    for (int i = 0; i < depthMat.rows; i++)
    {
        unsigned char* row_ptr = depthMat.ptr<unsigned char>(i);
        for (int j = 0; j < depthMat.cols; j++)
        {
            unsigned short p = *reinterpret_cast <unsigned short*>(data);
            pcl::PointXYZ point;
            float fov_vertical = -(TILT_RANGE / 2) + i * (TILT_RANGE / depthMat.rows);
            float fov_horizontal = PAN_RANGE / 2 - j * (PAN_RANGE / depthMat.cols);
           // float x_multiplier = (j - depthMat.cols / 2)/focal_x;
           // float y_multiplier = (depthMat.rows / 2 - i) / focal_y;
            float x_multiplier = (j - DEPTH_C_X) / DEPTH_FOCAL_X;
            float y_multiplier = (DEPTH_C_Y-i) / DEPTH_FOCAL_Y;

            point.y = ((double)(p)/1000 ) / sqrt(x_multiplier * x_multiplier + y_multiplier * y_multiplier + 1);
            point.z = point.y* y_multiplier;
            point.x = point.y * x_multiplier;

            //point.z += 90;
           // point.y += 90; //0 and 2*M_PI
            ptCloud->points.push_back(point);
            data++;
        }
    }
    ptCloud->width = (int)depthMat.cols;
    ptCloud->height = (int)depthMat.rows;
    return ptCloud;

}

//Performs RGB-D segmentation
pcl::PointCloud<pcl::PointXYZRGB>
cloud_cb (Mat depth,Mat rgb)
{  
  
  pcl::PointCloud<pcl::PointXYZRGB> color_cloud;

  double pan_res = PAN_RANGE / depth.cols;
  double tilt_res = TILT_RANGE / depth.rows;

  clock_t begin = clock();

  lidar_seg seglid(pan_res,tilt_res); //set the range image resolutions (The first one is pan, the second one is tilt)
  seglid.set_cloud( depth); //Set the cloud
  seglid.set_params(depth_pan_eps,depth_tilt_eps, depth_dist_eps,depth_min_p); //Set segmentation parameters
  seglid.segment(depth,rgb); //Start the segmentation
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout<<"DSA is successfully calculated :), elapsed time: "<<1/elapsed_secs <<" Hz"<<endl;
  seglid.take_colored_cloud(color_cloud); // Take colored cloud

  Mat img = CloudtoMat(color_cloud,depth.rows,depth.cols);
  imwrite("/home/malici/Desktop/segmented.jpg", img);

  return color_cloud;
}


pcl::PointCloud<pcl::PointXYZRGB>
cloud_cb(pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud)
{
    std::vector<int> indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZRGB> color_cloud;

    pcl::removeNaNFromPointCloud(*ptCloud, *outputCloud, indices);
    clock_t begin = clock();
    //LIDAR segmentation
    lidar_seg seglid(lidar_pan_res, lidar_tilt_res); //set the range image resolutions (The first one is pan, the second one is tilt)
    seglid.set_cloud(outputCloud); //Set the cloud
    seglid.set_params(lidar_pan_eps, lidar_tilt_eps, lidar_dist_eps, lidar_min_p); //Set segmentation parameters
    seglid.segment(); //Start the segmentation
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "DSA is successfully calculated :), elapsed time: " << 1 / elapsed_secs << " Hz" << endl;
    seglid.take_colored_cloud(color_cloud); // Take colored cloud


    return color_cloud;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr seg_ground(pcl::PointCloud<pcl::PointXYZ> laserCloudIn)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());


    ground_seg* ground_segmentation = new ground_seg(num_iter, num_lpr, th_seeds, th_dist);
    ground_segmentation->segment(laserCloudIn, ground_cloud, non_ground_cloud);
    cout <<"Ground size: " << laserCloudIn.size()- non_ground_cloud->size() << "\n"  ;
    return non_ground_cloud;
}
void calc3D(Mat depth, Mat rgb, const float depthCoor[12][2], const float rgbCoor[12][2])
{
    float depth3D[12][3];
    float rgb3D[12][3];
    
    for (int i = 0; i < 12; i++)
    {
       
        unsigned char* data = (depth.data);
        int depth_x = int(depthCoor[i][0]);
        int depth_y = int(depthCoor[i][1]);
        
        for (int j = 0; j < depth_y * 640 + depth_x; j++)
        { 
            data++;
            
        }
           
        unsigned short p = *reinterpret_cast <unsigned short*>(data);
        float x_multiplier = (depth_x - DEPTH_C_X) / DEPTH_FOCAL_X;
        float y_multiplier = (depth_y - DEPTH_C_Y) / DEPTH_FOCAL_Y;

        // Get 3D coordinates
        depth3D[i][0] = double(p) / 1000 / sqrt(x_multiplier * x_multiplier + y_multiplier * y_multiplier + 1);
        depth3D[i][1]= depth3D[i][0] * x_multiplier;
        depth3D[i][2] = depth3D[i][0] * y_multiplier;
        cout << "depth: "<< depth3D[i][0] <<" " <<depth3D[i][1] <<" " <<depth3D[i][2]<<endl;
        rgb3D[i][0] = depth3D[i][0];
        rgb3D[i][1] = rgb3D[i][0] * ((rgbCoor[i][0] - RGB_C_X) / RGB_FOCAL_X);
        rgb3D[i][2] = rgb3D[i][0] * ((rgbCoor[i][1] - RGB_C_Y) / RGB_FOCAL_Y);
        cout << "rgb: " << rgb3D[i][0] << " " << rgb3D[i][1] << " " << rgb3D[i][2] << endl;
    }

}
cv::Mat depthImg(480, 640, CV_8UC1);
void depth_cb(const sensor_msgs::ImageConstPtr& msg)
 {
    cout << "depth cb\n";
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_bridge::CvImagePtr bridge;
    try
    {
        bridge = cv_bridge::toCvCopy(msg, "32FC1");
    }
     catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Failed to transform depth image.");
        return;
    }
    try
    {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1);

    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    double min_range_=0.5;
    double max_range_=5.5;
    cv::Mat mono8_img ;//cv::Mat(cv_ptr->image.size(), CV_16UC1);

    for(int i = 0; i < bridge->image.rows; i++)
    {
        float* Di = bridge->image.ptr<float>(i);
        char* Ii = depthImg.ptr<char>(i);
        //std::cout<<*Di<<std::endl;
        for(int j = 0; j < bridge->image.cols; j++)
        {   
            Ii[j] = (char) (255*((Di[j]-min_range_)/(max_range_-min_range_)));
        }   
    }
    cv::imwrite("/home/malici/Desktop/a.jpg",depthImg);
    cout << "Depth image was saved.\n";
   // cv::imwrite("/home/malici/Desktop/depth.jpg",mono8_img);
   // mono8_img=cv::imread("/home/malici/Desktop/depth.jpg");
     // Output modified video stream
 }
 cv::Mat rgbImg = cv::Mat(480,640, CV_32FC3);
 void rgb_cb(const sensor_msgs::ImageConstPtr& msg)
 {
     cout << "RGB cb\n";
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msg);

    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      cout << "RGB image was not saved.";
      return;
    }

    rgbImg=cv_ptr->image.clone();
    //cv::convertScaleAbs(cv_ptr->image, rgbImg, 255, 0.0);
    
    cv::imwrite("/home/malici/Desktop/rgb.jpg",rgbImg);
    cout << "RGB image was saved.";
 }

int
main (int argc, char** argv)
{

   // const float DepthCoor[12][2] = { {133,163},{388,55},{183,184},{238,41},{574,132},{474,166},{227.0, 6.0}, {111.0, 5.0}, {233.0, 352.0}, {267.0, 375.0}, {158.0, 384.0}, {219.0, 139.0} };
   // const float RGBCoor[12][2] = { {146,182},{378,83},{190,198},{237,67},{550,155},{454,190}, {227.0, 34.0}, {136.0, 17.0}, {235.0, 359.0}, {273.0, 380.0}, {164.0, 377.0}, {223.0, 152.0} };
   // Mat depth_image = imread("depth224.jpg", CV_8UC1);
    //Mat rgb_image = imread("rgb224.jpg");
    //calc3D(depth_image, rgb_image, DepthCoor, RGBCoor);
    //Ground segmentation
    ros::init (argc, argv, "segmentation");
    ros::NodeHandle nh;
  
    // Create a ROS subscriber for the input point cloud
    //ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/ground_segmentation", 1, cloud_cb);
    ros::Subscriber sub = nh.subscribe("/zed/color/image_raw", 1, rgb_cb);
    ros::Subscriber sub2 = nh.subscribe("/zed/depth/image_raw", 1, depth_cb);
    
    // Spin
    while (ros::ok())
    { 
        ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.1));

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = MatToPoinXYZ(depthImg);
        /*pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
        viewer.showCloud(cloud);
        cv::waitKey(150000);*/
        pcl::PointCloud<pcl::PointXYZ>::Ptr grounded = seg_ground(*cloud);
        Mat dst = GroundCloudToMat(grounded, depthImg.cols, depthImg.rows);

        //RGB-D segmentation
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr depthCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        *depthCloud=cloud_cb(dst, rgbImg);
        
    }
}
