/*
 * rosserial Servo Control Example
 *
 * This sketch demonstrates the control of hobby R/C servos
 * using ROS and the arduino
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
#include <std_msgs/String.h>

// PCA9685 definitions
#define ESCMIN  1000 // This is the 'minimum' pulse length count of the ESC
#define ESCMAX  1960 // This is the 'maximum' pulse length count of the ESC
#define SERVOMIN  500 // This is the 'minimum' pulse length count of the steer Servo
#define SERVOMAX  2500 // This is the 'maximum' pulse length count of the steer SErvo
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

// Defines channel pins in PCA9685
#define gearbox 4 // Defines the channel of the gearbox servo
#define dif_back 3 // Defines the channel of the back diferencial lock servo
#define dif_front 2 // Defines the channel of the front diferencial lock servo
#define dir 1 // Defines the channel of the steer servo
#define ESC 0 // Defines the chanel for the esc

// Encoder Definitions
#define bodyEncoderLeftFunctionA bodyEncoderLeftCounterA
#define bodyEncoderLeftFunctionB bodyEncoderLeftCounterB
#define bodyEncoderLeftPinA 34 // A pin the interrupt pin
#define bodyEncoderLeftPinB 35 // B pin the interrupt pin

// encoder pulses
volatile signed int bodyEncoderLeftTotalPulses = 0;


// Ros handler
ros::NodeHandle  nh;

// Initializes the pwm board generator
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();


// Function that will generate the pwm when receive a message from ROS
void servo_dir( const std_msgs::Int16 & cmd_msg){
  int dirms = map(cmd_msg.data, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(dir, dirms);
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
}

// Function that will generate the pwm when receive a message from ROS
void servo_vel( const std_msgs::Int16 & cmd_msg){
  int escms = map(cmd_msg.data, 0, 180, ESCMIN, ESCMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(ESC, escms);  
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
}

// Function that will generate the pwm when receive a message from ROS
void servo_dif_front( const std_msgs::Int16 & cmd_msg){
  int dif_frontms = map(cmd_msg.data, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(dif_front, dif_frontms);  
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
  // 50 is closed, 110 is open
}

// Function that will generate the pwm when receive a message from ROS
void servo_dif_back( const std_msgs::Int16 & cmd_msg){
  int dif_backms = map(cmd_msg.data, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(dif_back, dif_backms);  
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
  // 70 is open, 120 is closed
}

// Function that will generate the pwm when receive a message from ROS
void servo_gearbox( const std_msgs::Int16 & cmd_msg){
  int gearboxms = map(cmd_msg.data, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(gearbox, gearboxms);  
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));  //toggle led  
}


ros::Subscriber<std_msgs::Int16> sub_dir("pub_dir", servo_dir);
ros::Subscriber<std_msgs::Int16> sub_vel("pub_vel", servo_vel);
ros::Subscriber<std_msgs::Int16> sub_dif_front("dif_front_cmd", servo_dif_front);
ros::Subscriber<std_msgs::Int16> sub_dif_back("dif_back_cmd", servo_dif_back);
ros::Subscriber<std_msgs::Int16> sub_gearbox("gearbox_cmd", servo_gearbox);
std_msgs::Int16 int16_msg;
ros::Publisher pub_encoder("encoder_msg", &int16_msg);

// Arduino initialization
void setup() {
  // Initializes the node of ROS
  nh.initNode();
  nh.subscribe(sub_dir); 
  nh.subscribe(sub_vel);
  nh.subscribe(sub_dif_front);
  nh.subscribe(sub_dif_back);
  nh.subscribe(sub_gearbox);
  nh.advertise(pub_encoder);
  
  // PCA9685 board
  pwm.begin(); // Begins comunnication
  pwm.setOscillatorFrequency(26315700); // This frequency can be tuned to adjust the generated pwm wave
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates
  int escms = map(90, 0, 180, ESCMIN, ESCMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(ESC, escms);  // Generates the PWM wave
  int servoms = map(90, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(dir, servoms); // Generates the PWM wave
  int dif_frontms = map(50, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(dif_front, dif_frontms); // Generates the PWM wave
  int dif_backms = map(120, 0, 180, SERVOMIN, SERVOMAX);   // scale it to use it with the servo library (value between 0 and 180)
  pwm.writeMicroseconds(dif_back, servoms); // Generates the PWM wave
  pwm.writeMicroseconds(gearbox, servoms); // Generates the PWM wave

  // Encoder
  pinMode(bodyEncoderLeftPinA,INPUT_PULLUP);
  pinMode(bodyEncoderLeftPinB,INPUT_PULLUP);


  attachInterrupt(bodyEncoderLeftPinA, bodyEncoderLeftFunctionA, CHANGE);
  attachInterrupt(bodyEncoderLeftPinB, bodyEncoderLeftFunctionB, CHANGE);
  delay(2000); // Waits 2 seconds
}

void loop() {
  int16_msg.data = bodyEncoderLeftTotalPulses;
  pub_encoder.publish( &int16_msg );
  nh.spinOnce();
  delay(1);
}


void bodyEncoderLeftCounterA() {

  // look for a low-to-high on channel A
  if (digitalRead(bodyEncoderLeftPinA) == HIGH) {
    // check channel B to see which way encoder is turning
    if (digitalRead(bodyEncoderLeftPinB) == LOW) {
      bodyEncoderLeftTotalPulses = bodyEncoderLeftTotalPulses - 1;         // CW
    } else {
      bodyEncoderLeftTotalPulses = bodyEncoderLeftTotalPulses + 1;         // CCW
    }
  } else {
    // its low-to-high-to-low on channel A
    // check channel B to see which way encoder is turning
    if (digitalRead(bodyEncoderLeftPinB) == HIGH) {
      bodyEncoderLeftTotalPulses = bodyEncoderLeftTotalPulses - 1;          // CW
    } else {
      bodyEncoderLeftTotalPulses = bodyEncoderLeftTotalPulses + 1;          // CCW
    }
  }
}


void bodyEncoderLeftCounterB() {

  // look for a low-to-high on channel B
  if (digitalRead(bodyEncoderLeftPinB) == HIGH) {

    // check channel A to see which way encoder is turning
    if (digitalRead(bodyEncoderLeftPinA) == HIGH) {
      bodyEncoderLeftTotalPulses = bodyEncoderLeftTotalPulses - 1;         // CW
    }
    else {
      bodyEncoderLeftTotalPulses = bodyEncoderLeftTotalPulses + 1;         // CCW
    }
  }

  // Look for a high-to-low on channel B
  else {
    // check channel B to see which way encoder is turning
    if (digitalRead(bodyEncoderLeftPinA) == LOW) {
      bodyEncoderLeftTotalPulses = bodyEncoderLeftTotalPulses - 1;          // CW
    }
    else {
      bodyEncoderLeftTotalPulses = bodyEncoderLeftTotalPulses + 1;          // CCW
    }
  }
}