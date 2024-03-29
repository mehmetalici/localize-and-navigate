from tensorflow.python.ops.gen_array_ops import Empty
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
import threading
import tensorflow as tf
import sensor_msgs
import std_msgs


warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_MODEL_DIR = '/home/malici/.keras/datasets/ssd_mobilenet_v2_320x320_coco17_tpu-8'
PATH_TO_LABELS = '/home/malici/.keras/datasets/mscoco_label_map.pbtxt'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
NO_BOX = [0, 0, 0, 0]


class Detector:
    def __init__(self, obj_to_detect, certainty_threshold=0.4):
        self.category_index = get_label_map_data()
        self.detect_fn = load_model()
        self.img_pub = rospy.Publisher("/rgb_with_dets", msg.Image, queue_size=50)
        self.img_depth_pub = rospy.Publisher("/depth_with_dets", msg.Image, queue_size=50)
        self.bridge = CvBridge()
        self.certainty_threshold = certainty_threshold
        self.bounding_box_publisher = rospy.Publisher("/bounding_box", std_msgs.msg.Int32MultiArray, queue_size=50)
        self.detection_starter_sub = rospy.Subscriber("/start_detection", std_msgs.msg.Empty, callback=self.start_detection_cb)
        self.box_pubs = [rospy.Publisher(name, std_msgs.msg.Float64, queue_size=1) for name in ["targetX", "targetY"]]
        self.image = None
        self.mode = "IDLE"
        self.start_detection = False
        self.obj = self.get_obj_as_dict(obj_to_detect)
        self.new_img_arrived = False

    def get_obj_as_dict(self, obj_to_detect):
        for k, v in self.category_index.items():
            if obj_to_detect == v["name"]:
                return {"id": v["id"], "name": v["name"], "box": np.array([]), "score": np.NaN}

        raise Exception(f"Object {obj_to_detect} not in category index defined in {PATH_TO_LABELS}")


    def start_detection_cb(self, data):
        self.start_detection = True

    def rgb_cb(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        self.image = cv_image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.new_img_arrived = True

    def depth_cb(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        self.depth = cv_image

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

    def convert_rel_to_norm_xywh(self, box):
        y1 = box[0] * self.height
        x1 = box[1] * self.width
        y2 = box[2] * self.height
        x2 = box[3] * self.width

        width = x2 - x1
        height = y2 - y1

        return np.array([x1, y1, width, height])

    def convert_norm_xywh_to_rel(self, box):
        x1, y1, width, height = [b for b in box]
        x2 = x1 + width
        y2 = y1 + height
        return np.array([y1/self.height, x1/self.width, y2/self.height, x2/self.width])

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


    def draw_obj_to_img(self, obj, img=None, show_score=True, draw_depth=False):
        score = None
        if show_score:
            score = np.array([obj["score"]])
        
        if img is None:
            image_copy = self.image.copy()
        else:
            image_copy = img.copy()

        images = [image_copy]
        if draw_depth:
            rows_d, cols_d = self.depth.shape
            depth_as_rgb_copy = np.array([[[self.depth[i, j]] * 3 for j in range(cols_d)] for i in range(rows_d)])
            images.append(depth_as_rgb_copy)

        for img_elt in images:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                img_elt,
                np.expand_dims(obj["box"], axis=0),
                np.array([obj["id"]]),
                score,
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)
        
        if len(images) == 1:
            return images[0]

        return images

    def publish_box_center(self, box):
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        centers = [center_x, center_y]
        for p, c in zip(self.box_pubs, centers):
            p.publish(c)
            pass
            
    def publish_bounding_box(self, box):
        msg = std_msgs.msg.Int32MultiArray()
        msg.data = box.astype(np.int32)
        self.bounding_box_publisher.publish(msg)



    def spinOnce(self):
        if self.mode == "IDLE":
            if self.start_detection:
                self.start_detection = False 
                self.mode = "WAITING"
                print(f"Mode changed to {self.mode}")
                

        if self.mode == "WAITING":
            if self.new_img_arrived:
                self.mode = "DETECT"
                print(f"Mode changed to {self.mode}")
        
        if self.mode == "DETECT":
            detections = self.get_detections()
            if self.got_object(detections):
                print("Found a {} at {}".format(self.obj["name"], self.obj["box"]))
                self.calibration_cnt = 0
                image_w_box = self.draw_obj_to_img(self.obj)
                self.publish_img(image_w_box)
                self.publish_bounding_box(self.convert_rel_to_norm_xywh(self.obj["box"]))
                self.mode = "IDLE"
                print(f"Mode changed to {self.mode}")
            else:
                print("Could not find the object. Searching...")
                self.publish_box_center(NO_BOX)
                self.publish_img(self.image)

        if self.mode not in ["IDLE", "DETECT"]:
            print("Undefined state")
            return

    def publish_img(self, image, depth=None):
        output_img_msg = self.bridge.cv2_to_imgmsg(image)
        self.img_pub.publish(output_img_msg)
        if depth is not None:
            depth = np.array(depth, dtype=np.uint8)
            output_depth_msg = self.bridge.cv2_to_imgmsg(depth)
            self.img_depth_pub.publish(output_depth_msg)

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

def load_model():
    print('Loading model...', end='')
    start_time = time.time()
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn

