# AutoMec-AD

This project goal is to develop a fully autonomous car to compete in the National Robotics Festival of Portugal. This repository is where the main code of University of Aveiro car is hosted. 

![alt text](https://i.imgur.com/7FCETQ1.png)

# Dependencies
Make sure you have these installed before booting 

  - Python3
  - Ubuntu 20.04
  - Ros Noetic
  - numpy==1.20.1
  - opencv-python==4.5.1.48
  - matplotlib==3.3.4

# How to run
The code used in this repository is meant to drive a real world car you can do so by folliwing the instructions on [Manual Driving](https://github.com/DanielCoelho112/AutoMec-AD/tree/readme/core/src/ManualDriving#manual-driving), however using ROS we can simulate an enviroment where we can test our robot (car). To do so, continue following the Setting up ROS Enviroment section.

First we need to create a catkin **workspace** to build our project. Choose an apropriate location for this.

## Setting up ROS Enviroment
```bash
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```
Next you need to source catkin to your setup.bash

If you are using bash terminal

```bash
$ source ~/catkin_ws/devel/setup.bash
```

Using zsh
```bash
$ source ~/catkin_ws/devel/setup.zsh
```

Next move in to the catkin src folder if you havent already and clone the repo.
```bash
cd catkin_ws
cd src
git clone https://github.com/DanielCoelho112/AutoMec-AD.git
git checkout dev
cd ..

```
Now you have the **main code** of our application but we still need to add some extra dependencies, ​this new repository that includes the turtle bot file and the arena used to simulate.

```bash
cd catkin_ws
cd src
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations
git clone https://github.com/ROBOTIS-GIT/turtlebot3
git clone https://github.com/callmesora/AutoMec-Deppendencies
git clone https://github.com/prolo09/pari_trabalho3
cd ..

```

All that's left to do is to run catkin_make, bhis will build your ros packages and install any depencies they have. When running catkin_make any dependencies installed in previous packages will still run in your new ones so be carefull when creating a new package
.

```bash
catkin_make
```

Add the following to your .bashrc  file
```bash
export TURTLEBOT3_MODEL=waffle_pi 
```


If you don't know how to edit your .bashrc file type

```bash
nano ~/.bashrc
```

And add the previous line to the end of your file. Same procedure if you use zsh terminal but with .zshrc
To Save press Cntrl+O , Enter . Cntrl+X to exit

# Ackerman dependencies
    sudo apt install ros-noetic-ros-controllers
    sudo apt-get install ros-noetic-ackermann-msgs

# Launch woth ackerman
    roslaunch ackermann_vehicle_gazebo ackermann_robot_with_arena_conversion.launch

# Running the simulation enviroment

Execute the following on one terminal 
```bash
roslaunch robot_bringup bringup_gazebo.launch
roslaunch robot_bringup spawn.launch
```
On a secound terminal.

```bash

roslaunch robot_bringup spawn.launch
```
First command will launch gazebo, secound one will spawn the robotcar. 
After this you should see gazebo opening with the racing track as shown.
![Gazebo](https://i.imgur.com/w7EFh7k.png)

# How to drive 
To test drive the car  type in gazebo  run
```bash
rqt
```
Now head down to pluggins --> Robot Tools --> Robot Stearing and select /robot/cmd_vel as it's topic.

![Driving](https://i.imgur.com/ME4mgl7.png)

After this we need to add the camera Go to  **Plugins --> Vizualization --> Image View** and select  */robot/camera/rgb/image_raw*

# Vision Code
During this project multiple aproaches are being tested , raw computer vision with opencv and a Machine Learning aproach. 

## Lane_Recognition
To test. cd into /core/src/VisionCode/Lane_Recognition
Launch gazebo and the robot as in the **Running the simulation** section

and run the script

```bash
rosrun core gazebo_lines.py
```

You should see multiple camera windows with different filters show up.

## Signal Recognition
To test. cd into /core/src/VisionCode/Signal Recognition

Open a terminal and run

```bash
roscore
```
On a secound terminal
```bash
python3 SignalRecognition.py
```


# Code structure
This project uses ROS as it's building blocks. It's divided into 4 packages

## core 

This package is divided into 4 folders.

ArduinoCode: All arduino code made.

VisionCode: All attempts at vision codes made.

ManualDriving: Only contains the files for manual driving to be possible, latest version of the Arduino, twist converter.

CommonFiles: Files that do not fit elsewhere, initialization files, etc.

## robot_bringup
Code to launch the gazebo and the robot in the simulation

## robot_core
 ​
Vision code ready to be implemented in gazebo

Automatic driving using a ML Aproach

## robot_description

All files regarding robot description and stats





# ML driving

CATBOOST:
          https://streamable.com/ol18mb

          https://streamable.com/acich8


          With images delivered to the catboost model:
          https://streamable.com/kfi7j6

CNN:
          https://streamable.com/ysugtn

## Dependencies for CNN
    sudo apt install python3-pip

    pip3 install opencv-python

    pip3 install pandas

    pip3 install sklearn

    pip3 install tensorflow

    pip3 install imgaug     



## License
https://streamable.com/t5thi0

