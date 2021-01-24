#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include <iostream>
#include <sstream>

#define CAM_WIDTH 640
#define CAM_HEIGHT 480
#define PI 3.14
#define RX_LIM PI / 3.6
#define RY_LIM PI / 3.6
#define REFLECT_AXIS_X 1
#define REFLECT_AXIS_Y 1

ros::Publisher panRefPub;
ros::Publisher tiltRefPub;
float current_rx = 0, current_ry = 0;

float calcRef(float target, float *current_r, int maxRes, float r_lim, bool is_reflected){
    float center = maxRes / 2;
    float diff = target - center;
    float new_r = (diff + center) / (maxRes) * r_lim - r_lim / 2;
    if (is_reflected){
        new_r = -new_r;
    }
    new_r += *current_r;
    *current_r = new_r;
    std::cout << *current_r;
    return new_r;
}

void targetXCb(const std_msgs::Float64::ConstPtr& targetX){
    std_msgs::Float64 panRef;
    panRef.data = calcRef(targetX->data, &current_rx, CAM_WIDTH, RX_LIM, REFLECT_AXIS_X);
    panRefPub.publish(panRef);
}

void targetYCb(const std_msgs::Float64::ConstPtr& targetY){
    std_msgs::Float64 tiltRef;
    tiltRef.data = calcRef(targetY->data, &current_ry, CAM_HEIGHT, RY_LIM, REFLECT_AXIS_Y);
    tiltRefPub.publish(tiltRef);
}


int main(int argc, char **argv){

    ros::init(argc, argv, "pt_cmd");

    ros::NodeHandle nh;
    
    panRefPub = nh.advertise<std_msgs::Float64>("/pan_controller/command", 1);
    tiltRefPub = nh.advertise<std_msgs::Float64>("/tilt_controller/command", 1);

    ros::Subscriber targetXSub = nh.subscribe("targetX", 1, targetXCb);
    ros::Subscriber targetYSub = nh.subscribe("targetY", 1, targetYCb);

    while (true){
        ros::spinOnce();
    }
    
}