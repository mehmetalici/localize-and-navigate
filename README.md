# Object Localization and Relative Navigation on Gazebo
An object detection and relative navigation scheme for our custom robot on Gazebo.

## Getting Started
### Prerequisites
1. ROS Melodic

    Install following the instructions on http://wiki.ros.org/melodic/Installation.
2. ROS Package Dependencies
    
    Install them via,
    ``` 
    sudo apt-get install ros-melodic-costmap-2d ros-melodic-robot-localization ros-melodic-yocs-cmd-vel-mux ros-melodic-effort-controllers ros-melodic-navigation ros-melodic-geometry2 ros-melodic-nmea-msgs ros-melodic-bfl ros-melodic-arbotix-python
    ```

3. Tensorflow Object Detection API

    Install following the tutorial under the link,
    
     https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install

4. Python Dependencies

    Activate the Conda environment created in Step 3 and run,
    ```
    conda install opencv-python
    ```
### Installing
#### Robot and Environment
1. Create a new Catkin workspace,
    ``` 
    mkdir -p ~/navi_ws/src
    ```
2. Clone the repository to the workspace,
    ``` 
    cd ~/navi_ws/src
    git clone https://github.com/mehmetalici/localize-and-navigate.git
    ```


4. Build the packages in the workspace,
    ``` 
    cd ~/navi_ws
    catkin_make
    ``` 
5. Source the workspace and the robot,
    ``` 
    cd ~/navi_ws
    source devel/setup.bash
    source src/localize-and-navigate/segway_v3/segway_v3_config/std_configs/segway_config_RMP_220.bash
    ``` 
6. Test the robot and environment running,
    ``` 
    roslaunch segway_gazebo segway_empty_world.launch
    ``` 
    You should see our robot and world in the Gazebo simulation software.

#### Detection
1. Download a model
    
    Activate your environment, and run,
    ``` 
    cd ~/navi_ws/src/localize-and-navigate/obj_detection/src
    python download_model.py 20200711 ssd_mobilenet_v2_320x320_coco17_tpu-8
    ``` 
    to download the model with name and date `ssd_mobilenet_v2_320x320_coco17_tpu-8` and `20200711`, respectively. 

    You can change the model name and date according to the models in the following list, which involves all pre-trained models,
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

#### Tracking
1. Download a specific version of OpenCV Bridge.

    Apply,
    ``` 
    cd ~
    git clone https://github.com/mehmetalici/opencv-bridge-err-resolver.git
    cd opencv-bridge-err-resolver
    catkin clean -b
    catkin build
    ``` 

## Robot Control

### Control of Head
A pan-tilt device rotates the head of the robot. The joints of Pan-Tilt are controlled by their PID controllers. 

The head features a ZED RGM-D camera and it is possible to control the head by providing (x, y) coordinates of the target in the RGB image. To test it, first run,
``` 
roscd robot
rosrun rviz rviz -d rviz/rviz_cfg.rviz
``` 
and 
For example, to move to a position of 0.5 rads positive to the axis of rotation for both joints, run,
``` 
rostopic pub /pan_controller/command std_msgs/Float64 "data: 0.5" 
rostopic pub /tilt_controller/command std_msgs/Float64 "data: 0.5" 
``` 



### Control of Navigation
The robot can be navigated by commanding to RMP220.

For instance, to move the robot through a circle, run,
``` 
rostopic pub -r 10 /segway/teleop/cmd_vel geometry_msgs/Twist "linear:
  x: 10.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 1.0"
``` 

### Control of Sensors
The robot is equipped with a Velodyne Puck LIDAR and a ZED RGB-D Camera. The sensors are always active and publishing under their respective topics.

Sensor information can be visualized using `RViz`. To do this, follow the instructions [here](http://gazebosim.org/tutorials?tut=drcsim_visualization&cat=drcsim
). 


## Detection and Tracking
1. Activate your environment and source your workspace. 
2. Run,
    ```
    roscd obj_detection/src
    source ~/opencv-bridge-err-resolver/install/setup.bash
    ```
3. To detect a `person`, run,
    ```
    python detect_track.py person
    ```
    To see a complete list of avaliable objects you can pass to the script, go to the directory in which you downloaded the model and open `mscoco_label_map.pbtxt`.

4. To visualize detection, run,
    ```
    rosrun rviz rviz
    ```
    Then select the topics `/rgb_with_det` or `/rgb_with_det`, which produces images with boxes of target object, on RViz GUI.

## Navigation
1. Source the workspace and run,

    ```
    roscd obj_detection/src
    python navigate_goal.py 
    ```
2. Run the APF algorithm,
    ```
    roscd apfrl/src
    python apf_rl_0707.py
    ```

