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


#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <ros.h>
#include <std_msgs/Int16.h>

ros::NodeHandle  nh;


#define ESCMIN  1000 // This is the 'minimum' pulse length count (out of 4096)
#define ESCMAX  1960 // This is the 'maximum' pulse length count (out of 4096)
#define SERVOMIN  500 // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  2500 // This is the 'maximum' pulse length count (out of 4096)
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates
#define servo 1
#define ESC 0

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();


void servo_dir( const std_msgs::Int16 & cmd_msg){
  int servoms = map(cmd_msg.data, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(servo, servoms);
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
}

void servo_vel( const std_msgs::Int16 & cmd_msg){
  int escms = map(cmd_msg.data, 0, 180, ESCMIN, ESCMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(ESC, escms);  
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
}


ros::Subscriber<std_msgs::Int16> sub_dir("pub_dir", servo_dir);
ros::Subscriber<std_msgs::Int16> sub_vel("pub_vel", servo_vel);

int potValue=90;  // value from the analog pin
void setup() {
  nh.initNode();
  nh.subscribe(sub_dir);
  nh.subscribe(sub_vel);
  
  pwm.begin();
  pwm.setOscillatorFrequency(26315700);
  pwm.setPWMFreq(50);  // Analog servos run at ~50 Hz updates
  int escms = map(90, 0, 180, ESCMIN, ESCMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(ESC, escms);
  int servoms = map(90, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(servo, servoms);
  delay(2000);
}

void loop() {
  nh.spinOnce();
  delay(1);
}


