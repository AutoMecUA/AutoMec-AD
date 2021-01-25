/*
        Arduino Brushless Motor Control
     by Dejan, https://howtomechatronics.com
*/

#include <Servo.h>

Servo ESC;     // create servo object to control the ESC

int potValue;  // value from the analog pin

void setup() {
  // Attach the ESC on pin 9
  ESC.attach(9); // (pin, min pulse width, max pulse width in microseconds) 
  Serial.begin(9600);
}


void loop() {
  potValue = analogRead(A0);   // reads the value of the potentiometer (value between 0 and 1023)
  potValue = map(potValue, 0, 1023, 0, 180);   // scale it to use it with the servo library (value between 0 and 180)  só ira neste caso andar par tras
  ESC.write(potValue);    // Send the signal to the ESC
  Serial.println(digitalRead(9));
}

//void loop(){
//  ESC.write(90); //posição neutra
//  delay(1000);
//
//  //inicio de aceleração forward
//  for(int pos=90; pos<180; pos=pos+1){
//    ESC.write(pos);
//    delay(500);
//    }
//
//  //retorno a posição neutra  
//  ESC.write(90);
//  delay(1000);
//
//  //rampa de aceleração backward
//  for(int pos=90; pos>0; pos=pos-1){
//    ESC.write(pos);
//    delay(500);
//    }
//
//
//  //retorno a posição neutra  
//  ESC.write(90);
//  delay(1000);
//  }
