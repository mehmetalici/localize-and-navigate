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

EXPORT_IMG = True
graph = None
segment_frame = None
is_model_loaded = None
bridge = CvBridge()

image_path = ""

class DepthCounter:
    c = 0

class ColorCounter:
    c = 0

def export_color(image):
    save_folder = Path("./imgs").absolute()
    image_name = f"img_{ColorCounter.c}_color.jpg"
    image_path = str(Path(save_folder, image_name))

    image = Image.fromarray(image, "RGB")
    
    image.save(image_path)
    print(f"Current image was saved to {image_path}")
    ColorCounter.c += 1


def export_depth(image):
    save_folder = Path("./imgs").absolute()
    image_name = f"img_{DepthCounter.c}_depth.jpg"
    image_path = str(Path(save_folder, image_name))
    #To save image as png
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=1), cv2.COLORMAP_BONE)
    cv2.imwrite(image_path, depth_colormap)
    #depth_array = np.asanyarray(image)
    #Or you use 
    #depth_array = depth_array.astype(np.uint16)
    #cv2.imwrite(image_path, depth_array)
    print(f"Depth image was saved to {image_path}")
    DepthCounter.c += 1

def color_cb(data, cb_args):
    pub = cb_args
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
    #global graph
    if EXPORT_IMG:
        export_color(cv_image) 
    else:
        global segment_frame
        global is_model_loaded
        print("Callback in")
        if not is_model_loaded:
            print("is in?")
            #graph = tf.compat.v1.get_default_graph()
            segment_frame = semantic_segmentation()
            #with graph.as_default():
            segment_frame.load_ade20k_model("models_px/deeplabv3_xception65_ade20k.h5")
            
            is_model_loaded = True
        
        #with graph.as_default():
        segmask, output = segment_frame.segmentAsAde20k(cv_image, process_frame=True)
        print("cb out")
        output_img_msg = bridge.cv2_to_imgmsg(output)
        pub.publish(output_img_msg)

def depth_cb(data, cb_args):
    pub = cb_args
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
    export_depth(cv_image)




def run_detection():
    rospy.init_node("detector")
    pub = rospy.Publisher("/segmask", msg.Image, queue_size=50)
    rospy.Subscriber("/zed/color/image_raw", msg.Image, callback=color_cb, callback_args=pub)
    rospy.Subscriber("/zed/depth/image_raw", msg.Image, callback=depth_cb, callback_args=pub)
    
    rospy.spin()

if __name__ == "__main__":
    run_detection()