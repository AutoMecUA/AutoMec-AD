# How this works?

the aim of this project is to use the arduino to control an RC car. In this section, we must connect the ROS to the arduino serial port. For this, the main objective is that the arduino can hear the topics pub_dir and pub_vel, something like the following image (still incomplete)

![Goal](pics/rqp_graph.png)

# How make this work?

to work it is necessary to run a publisher for each speed, still just an imaginary of the dir done, and a subcritor that runs on the arduino and will read the messages that come to it

# Comand sequence!
(of course, first off all download the .ino file to arduino)

#### 1 run a master for all nodes with this command 

    roscore

#### 2 run the publisher

    ./Arduino/Ros/publish_dir.py 

#### 3 run rosserial

    rosrun rosserial_python serial_node.py <arduino port>

example of arduino port:

    /dev/ttyACM1    
