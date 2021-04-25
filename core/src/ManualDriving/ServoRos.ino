/*
 * rosserial Servo Control Example
 *
 * This sketch demonstrates the control of hobby R/C servos
 * using ROS and the arduiono
 * 
 * For the full tutorial write up, visit
 * www.ros.org/wiki/rosserial_arduino_demos
 *
 * For more information on the Arduino Servo Library
 * Checkout :
 * http://www.arduino.cc/en/Reference/Servo
 */

// !!!!!!!!!!!!!!!!!   IMPORTANTE   !!!!!!!!!!!
// rosrun rosserial_python serial_node.py /dev/ttyACM1
// rosrun rosserial_python serial_node.py /dev/ttyACM1
// rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0 _baud:=9600
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
#else
  #include <WProgram.h>
#endif

#include <Servo.h> 
#include <ros.h>
#include <std_msgs/Int16.h>

ros::NodeHandle  nh;

Servo servo;
Servo ESC;

void servo_dir( const std_msgs::Int16 & cmd_msg){
  servo.write(cmd_msg.data); //set servo angle, should be from 0-180  
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
}

void servo_vel( const std_msgs::Int16 & cmd_msg){
  ESC.write(cmd_msg.data); //set servo angle, should be from 0-180  
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
}


ros::Subscriber<std_msgs::Int16> sub_dir("pub_dir", servo_dir);
ros::Subscriber<std_msgs::Int16> sub_vel("pub_vel", servo_vel);

void setup(){
  //pinMode(LED_BUILTIN, OUTPUT);

  nh.initNode();
  nh.subscribe(sub_dir);
  nh.subscribe(sub_vel);
  
  servo.attach(9); //attach it to pin 9
  ESC.attach(6);
  ESC.write(90);
  delay(2000);
}

void loop(){
  nh.spinOnce();
  delay(1);
} 


