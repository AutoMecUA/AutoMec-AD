# Manual Driving 

# Indice
 - [How this works](#How-this-works?)
 - [Comand Sequence](#Comand-sequence!)
 - [App RosControl](#App-RosControl)
 - [PS3 Controller](#PS3Controller)


# How this works?

the aim of this project is to use the arduino to control an RC car. In this section, we must connect the ROS to the arduino serial port. For this, the main objective is that the arduino can hear the topics pub_dir and pub_vel, something like the following image.

![Goal](Images/rqp_graph.png)

# How make this work?

to work it is necessary to run a publisher for each speed, still just an imaginary of the dir done, and a subcritor that runs on the arduino and will read the messages that come to it

# Comand sequence!
(of course, first off all download the .ino file to arduino)

#### 1º run a master for all nodes with this command 

    roscore

#### 2º run rosserial

    rosrun rosserial_python serial_node.py <arduino port>

example of arduino port:

- /dev/ttyACM1 

#### 3º run the RosControl Conversor

    rosrun core AndroidConversor.py



   



 # App RosControl

* Ros Control Instalation

The first step is to install the following android application: 

[Ros Control](https://play.google.com/store/apps/details?id=com.robotca.ControlApp&hl=en&gl=US)

* Ros Control Configuration

Click Add Robot, in the Robot name textbox, you can assign any name.
In the Master URI textbox you have to write something like this: http: // <ip_ros_master>: 1131

Where <ip_ros_master> is the ip of the computer where the rosmater is running.

Now, click in Show Advanced Options and change the Joystick Topic to "android_input", as shown in the next image.

<img src="Images/AppConfig.jpg" width="250"/>


It should be noted that the two devices need to be connected to the same network. If there are problems with the wireless network, make a hotspot with mobile data.

Now, just press OK!













