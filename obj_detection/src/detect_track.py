import rospy
from sensor_msgs import msg
import os
from object_detection.utils import label_map_util
import time
import numpy as np
from PIL import Image
import warnings
from cv_bridge import CvBridge, CvBridgeError
from object_detection.utils import visualization_utils as viz_utils
import argparse
from detect_tracker import Detector
import threading
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("object")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    detector = Detector(args.object)
    rospy.init_node("detector_rgb")
    rospy.Subscriber("/zed/color/image_raw", msg.Image, callback=detector.rgb_cb)
    rospy.Subscriber("/zed/depth/image_raw", msg.Image, callback=detector.depth_cb)

    while True:
        if detector.new_img_arrived:
            detector.new_img_arrived = False
            detector.spinOnce()
        else:
            time.sleep(0.1)
    
            
        


if __name__ == "__main__":
    main()
