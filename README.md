# AutoMec-AD

This repository serves as the main repository for the autonomous RC car team of AutoMec. The repository consist of ML code
for the lane detection, template matching code for signal recognition, all incorporated in a ROS framework.

![alt text](https://raw.githubusercontent.com/AutomecUA/AutoMec-AD/main/images/car.jpeg)

All the setup, commands and methods used are described in the [wiki](https://github.com/AutomecUA/AutoMec-AD/wiki). <br>
This is still a WIP. Some errors are to be expected. If you have any doubts or want to report a bug,
feel free to use the Issues or [send us an email](mailto:dem-automec@ua.pt)!

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
