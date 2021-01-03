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

3. SDL Library Dependencies

    Run,
    ``` 
    sudo apt-get install libsdl-image1.2-dev libsdl-dev
    ```


### Installing
1. Create a new Catkin workspace,
    ``` 
    mkdir -p ~/navi_ws/src
    ```
2. Clone the repository to the workspace,
    ``` 
    cd ~/navi_ws/src
    git clone https://github.com/mehmetalici/localize-and-navigate.git
    ```

3. Convert `PLUGINLIB_DECLARE_CLASS` to `PLUGINLIB_DECLARE_CLASS` following [this link](http://docs.ros.org/en/jade/api/pluginlib/html/class__list__macros_8h.html).

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
6. Test the installation running,
    ``` 
    roslaunch segway_gazebo segway_empty_world.launch
    ``` 
    You should see our robot and world in the Gazebo simulation software.


## Usage

### Control of Head
A pan-tilt device rotates the head of the robot. The joints of Pan-Tilt are controlled by their PID controllers. 

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


## Acknowledgements
The repository is under active development. Our future plans involve object detection and navigation.
