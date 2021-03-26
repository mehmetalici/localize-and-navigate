import rospy
from sensor_msgs import msg
from std_msgs.msg import Float64
import tensorflow as tf
import os
from object_detection.utils import label_map_util
import time
import numpy as np
from PIL import Image
import warnings
from cv_bridge import CvBridge
from object_detection.utils import visualization_utils as viz_utils
from cv2 import cv2
from std_msgs.msg import Int32MultiArray, Empty

TRACKER_TYPES = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


class Tracker:
    def __init__(self, tracker_name="MIL"):
        self.target_depth_pub = rospy.Publisher("/targetD", Float64, queue_size=1)
        self.box_pubs = [rospy.Publisher(name, Float64, queue_size=1) for name in ["targetX", "targetY"]]
        self.image = None
        self.depth = None
        self.bridge = CvBridge()
        self.tracker = get_tracker(tracker_name)
        self.pred_interval = 10
        self.pred_cnt = 0
        self.bounding_box = None
        self.bounding_box_sub = rospy.Subscriber("/bounding_box", Int32MultiArray, self.bounding_box_cb)
        self.detection_starter_pub = rospy.Publisher("/start_detection", Empty)
        self.new_bbox_arrived = False
        self.mode = "IDLE"

    def bounding_box_cb(self, data):
        self.bounding_box = data.data
        self.new_bbox_arrived = True

    def depth_cb(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        self.depth = cv_image

    def publish_target_distance(self):
        [x, y, w, h] = list(map(int, self.convert_rel_to_norm_xywh(self.obj["box"])))
        mask = np.zeros(self.depth.shape, np.uint8)
        mask[y: y+h, x: x+w] = 255
        hist = cv2.calcHist([self.depth], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
        target_depth = np.argmax(hist.flatten())
        print(f"Target distance is {target_depth} m.")
        self.target_depth_pub.publish(target_depth)

    def rgb_cb(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        self.image = cv_image


    def spinOnce(self):
        if self.mode == "IDLE":
            self.detection_starter_pub.publish()
            self.mode = "WAITING"
        
        if self.mode == "WAITING":
            if self.new_bbox_arrived:
                self.mode = "TRACK"
                self.new_bbox_arrived = False

        if self.mode == "TRACK":
            ok, bbox = self.tracker.update(self.image)
            if ok:
                # draw bbox to img
                # publish img
                self.publish_box_center(bbox)
                self.publish_target_distance()
                self.pred_cnt += 1
                if self.pred_cnt == self.pred_interval:
                    self.pred_cnt = 0
                    self.mode = "IDLE"
            else:
                print("Tracking failure.")

        if self.mode not in ["IDLE", "WAITING", "TRACK"]:
            print("Undefined state")
            return

    def publish_img(self, image, depth=None):
        output_img_msg = self.bridge.cv2_to_imgmsg(image)
        self.img_pub.publish(output_img_msg)
        if depth is not None:
            depth = np.array(depth, dtype=np.uint8)
            output_depth_msg = self.bridge.cv2_to_imgmsg(depth)
            self.img_depth_pub.publish(output_depth_msg)

    def publish_box_center(self, box):
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        centers = [center_x, center_y]
        for p, c in zip(self.box_pubs, centers):
            p.publish(c)
            pass
            
        #print(f"center X: {center_x}\n centerY: {center_y}")
        

def get_tracker(tracker_type):
    
    # Set up tracker.
    # Instead of MIL, you can also use
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type in TRACKER_TYPES:
            if tracker_type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()
        else:
            raise Exception(f"Tracker type {tracker_type} is not supported.")

    return tracker

    
