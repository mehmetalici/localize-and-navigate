import rospy
from sensor_msgs import msg
import numpy as np
from PIL import Image
from pathlib import Path
from cv_bridge import CvBridge, CvBridgeError
from cv2 import cv2
import pixellib
from pixellib.semantic import semantic_segmentation
import tensorflow as tf
import ros_numpy
import open3d as o3d
import pcl


def pc_cb(data, cb_args):
    xyz_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_arr)
    o3d.visualization.draw_geometries([pcd])
    cloud = pcl.PointCloud2()
    cloud.fromarray(xyz_arr)
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    seg.set_normal_distance_weight(0.01)
    seg.set_max_iterations(100)
    indices, coefficients = seg.segment()
    pass

def run_detection():
    rospy.init_node("detector_pc")
    pub = rospy.Publisher("/segmask_pc", msg.Image, queue_size=50)
    rospy.Subscriber("/zed/depth/points", msg.PointCloud2, callback=pc_cb, callback_args=pub)
    rospy.spin()

if __name__ == "__main__":
    run_detection()
