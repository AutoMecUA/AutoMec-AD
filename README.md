# AutoMec-AD

This repository serves as the main repository for the autonomous RC car team of Automec. The repository consist of ML code
for the lane detection, template matching code for signal recognition, all incorporated in a ROS framework.

![alt text](https://raw.githubusercontent.com/AutomecUA/AutoMec-AD/main/images/car.jpeg)

All the setup, commands and methods used are described in the [wiki](https://github.com/AutomecUA/AutoMec-AD/wiki). <br>
This is still a WIP. Some errors are to be expected. If you have any doubts or want to report a bug,
feel free to use the Issues or [send us an email](mailto:dem-automec@ua.pt)!

## Quick cheat-sheet

To run the gazebo arena:
```
roslaunch prometheus_gazebo arena.launch
```

To launch the car in simulation:
```
roslaunch prometheus_bringup bringup.launch sim:=true
```

Optionally, you can also add some visualization, and control with a game console controller:

```
roslaunch prometheus_bringup bringup.launch sim:=true visualize:=true controller:=true
```

To write the dataset:

```
roslaunch prometheus_driving dataset_writing.launch
```

To train the model, run the script `ml_training.ipynb`, from package `prometheus_driving` in a jupyter notebook. 

The model running and the signal recognition will soon be ported. 