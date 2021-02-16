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
import tensorflow as tf


class Detector():
    
    def rgb_cb(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        self.image = cv_image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.new_img_arrived = True

    def got_object(self, detections):
        idxs = [i for i, c in enumerate(detections["detection_classes"]) if c == self.obj["id"]]
        if not len(idxs):
            return False

        boxes = detections["detection_boxes"][idxs]
        scores = detections["detection_scores"][idxs]
        max_score = np.max(scores)
        max_score_idx = np.argmax(scores)

        if max_score < self.certainty_threshold:
            return False   
                        
        self.obj["box"] = boxes[max_score_idx]        
        self.obj["score"] = max_score
        return True

    def get_detections(self):
        input_tensor = tf.convert_to_tensor(self.image)
        input_tensor = input_tensor[tf.newaxis, ...]
        start_time = time.time()
        detections = self.detect_fn(input_tensor)
        end_time = time.time()
        print(f"Inference took {end_time-start_time} seconds.")
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return detections


def load_model():
    print('Loading model...', end='')
    start_time = time.time()
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def get_label_map_data():
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
    return category_index


rospy.init_node("detector")
detector = Detector()
rospy.Subscriber("/zed/color/image_raw", msg.Image, callback=detector.rgb_cb)
