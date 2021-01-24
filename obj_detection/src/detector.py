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
from cv_bridge import CvBridge, CvBridgeError
from object_detection.utils import visualization_utils as viz_utils
from cv2 import cv2

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


PATH_TO_MODEL_DIR = '/home/malici/.keras/datasets/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
PATH_TO_LABELS = '/home/malici/.keras/datasets/mscoco_label_map.pbtxt'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
TRACKER_TYPES = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


class Detector:
    def __init__(self, obj_to_detect, tracker_name="MIL", certainty_threshold=0.7):
        self.category_index = get_label_map_data()
        self.obj = self.get_obj_as_dict(obj_to_detect)
        self.detect_fn = load_model()
        self.img_pub = rospy.Publisher("/rgb_with_dets", msg.Image, queue_size=50)
        self.box_pubs = [rospy.Publisher(name, Float64, queue_size=1) for name in ["targetX", "targetY"]]
        self.new_img_arrived = False
        self.image = None
        self.bridge = CvBridge()
        self.mode = "IDLE"
        self.certainty_threshold = certainty_threshold
        self.tracker = get_tracker(tracker_name)
        self.width = None
        self.height = None
        self.calibration_max = 1
        self.calibration_cnt = 0
        self.last_target_loc = None
        self.pred_interval = 10
        self.pred_cnt = 0

    def get_obj_as_dict(self, obj_to_detect):
        for k, v in self.category_index.items():
            if obj_to_detect == v["name"]:
                return {"id": v["id"], "name": v["name"], "box": np.array([]), "score": np.NaN}
        raise Exception(f"Object {obj_to_detect} not in category index defined in {PATH_TO_LABELS}")

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


    def draw_obj_to_img(self, obj, show_score=True):
        score = None
        if show_score:
            score = np.array([obj["score"]])
        image_copy = self.image.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_copy,
            np.expand_dims(obj["box"], axis=0),
            np.array([obj["id"]]),
            score,
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
        return image_copy

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

    def spinOnce(self):
        if self.mode == "IDLE":
            self.mode = "SEARCH"
            print("Searching the {}".format(self.obj["name"]))
        if self.mode == "SEARCH":   
            detections = self.get_detections()
            if self.got_object(detections):
                print("Found object with box {}".format(self.obj["box"]))
                self.calibration_cnt += 1
                #print("Waiting {}%".format(int(self.calibration_cnt/self.calibration_max*100)))
                if self.calibration_cnt == self.calibration_max:
                    self.calibration_cnt = 0
                    image_w_box = self.draw_obj_to_img(self.obj)
                    self.publish_img(image_w_box)
                    self.tracker.init(self.image, self.convert_rel_to_norm_xywh(self.obj["box"])) 
                    self.mode = "TRACK"
            else:
                print("Could not find the object. Searching...")
                self.calibration_cnt = 0
        if self.mode == "TRACK":
            ok = self.track()
            if ok:
                image_with_box = self.draw_obj_to_img(self.obj, show_score=False)
                self.publish_img(image_with_box)
                self.publish_box_center(self.convert_rel_to_norm_xywh(self.obj["box"]))
                self.pred_cnt += 1
                if self.pred_cnt == self.pred_interval:
                    self.pred_cnt = 0
                    self.mode = "SEARCH"
            else:
                print("Tracking failure.")
            #opencv api
            #derinlik
            print("Tracking the object {} at {}".format(self.obj["name"], self.obj["box"]))
    
    def publish_img(self, image):
        output_img_msg = self.bridge.cv2_to_imgmsg(image)
        self.img_pub.publish(output_img_msg)

    def publish_box_center(self, box):
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        centers = [center_x, center_y]
        for p, c in zip(self.box_pubs, centers):
            p.publish(c)
            pass
            
        print(f"center X: {center_x}\n centerY: {center_y}")

    def track(self):
        timer = cv2.getTickCount()
        ok, bbox = self.tracker.update(self.image)
        self.obj["box"] = self.convert_norm_xywh_to_rel(bbox)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        return ok
        

def load_model():
    print('Loading model...', end='')
    start_time = time.time()
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn


def get_label_map_data():
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
    return category_index


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

    