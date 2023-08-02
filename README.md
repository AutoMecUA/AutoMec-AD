# AutoMec AD
![ROS](https://img.shields.io/badge/ros-%230A0FF9.svg?style=for-the-badge&logo=ros&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

<<<<<<< HEAD
This repository serves as the main repository for the autonomous RC car team of AutoMec. The repository consist of ML code
for the lane detection, template matching code for signal recognition, all incorporated in a ROS framework.
=======
>>>>>>> 40254a4244cb09b758f7b297e9e651befd35b496


This repository serves as the main repository for the autonomous RC car team of AutoMec AD. 
The software developed uses a CNN for lane detection and template matching for signal recognition.
All the software is incorporated into the ROS framework.
<br>
<br>
<a href="url"><img src="images/AutomecLogo.png" align="center"></a>


## How to run:

All the setup, commands and methods used are described in the [wiki](https://github.com/AutomecUA/AutoMec-AD/wiki). <br>
![alt text](images/AnimationAutomec-Compressed.gif)

This is still a WIP. Some errors are to be expected. If you have any doubts or want to report a bug,
feel free to use the Issues or [send us an email](mailto:dem-automec@ua.pt)!

<<<<<<< HEAD
## Quick cheat-sheet

To correctly run this code, please insert in your .bashrc/.zshrc:

```
export GAZEBO_MODEL_PATH="`rospack find prometheus_gazebo`/models:${GAZEBO_MODEL_PATH}"
export automec_developer="*place your name here"
```

And please install the following dependencies:

```
sudo apt-get install ros-noetic-ros-controllers ros-noetic-ackermann-msgs ros-noetic-navigation
```

And place the following package inside your workspace:

https://github.com/CIR-KIT/steer_drive_ros (on the `melodic-devel`)

To run the gazebo arena:
```
roslaunch prometheus_gazebo arena.launch
```

To run the signal panel:

```
roslaunch prometheus_gazebo signal_panel.launch
```

To launch the car in simulation:
```
roslaunch prometheus_bringup bringup.launch sim:=true
```

Optionally, you can also add some visualization, and control with a game console controller:

```
roslaunch prometheus_bringup bringup.launch sim:=true visualize:=true controller:=true
```

You can also change the linear velocity:

```
roslaunch prometheus_bringup bringup.launch sim:=true linear_velocity:=0.5
```

To write the dataset:

```
roslaunch prometheus_driving dataset_writing.launch
```

To train the model, run the script `ml_training.ipynb`, from package `prometheus_driving` in a jupyter notebook. 

To run the dataset:

```
roslaunch prometheus_driving ml_driving.launch model:=*insert model name*
```

And to launch the signal detection:

```
roslaunch prometheus_signal_recognition signal_recognition.launch 
```

If you want to run the model without using the signal detection, please launch the following command to start the driving:

```
rostopic pub /signal_detected std_msgs/String "pForward"
```

If you want to run the complete driving experience, please use:

```
roslaunch prometheus_bringup main_driving.launch model_name:=*insert model name*
```
=======

## Simulation Environment:
The software is developed and tested on a virtual environment named Gazebo. 


![alt text](images/gazebo_track.png)

## Challenges
The autonomous vehicle aims to:
- Drive autonomously in a closed circuit
- Avoid objects
- Detect traffic lights and vertical signals
- Autonomously park


### Driving
![alt text](images/AnimationAutomec2-Compressed.gif)

### Parking 
![alt text](images/Parking-Compressed.gif)




>>>>>>> 40254a4244cb09b758f7b297e9e651befd35b496
