# How ESC Works?

The ESC is a controller to drive the current for the coils in the motor by the right order.


# How run then in arduino?

In the arduino, control a ESC is equal how you control a server motor, so using a **Servo.h** library available on arduino you just give them the pin how you would create a PWM signal and after that just write the position.



# How position works on a 360 servo motor, normally knowed by continuous servo?

In a continuos servo, the standard values of  position are: 

- 0 -> it will make our motor run __Backward__

- 90 -> it will make our motor **Stop**

- 180 -> it will make our motor run **Forward**

where we have three possibilities for our motor:

    1. run backward and forward [0,180]
    2. run only backward [0,90]
    3. run only forward [90,180]

lets talk a little bit mora about this possibilities.



<!-- Mode 1 -->
## 1. Run Forward and Backward
In this mode the max value for write is **180** and the lowest is **0**.

- For **lowest** value we will get the max velocity for **backward**

- For **highest** value we will get the max velocity for **forward**

- For **config point** the value of that is the middle between max and lower, on this case **90**

So on this mode:
- **0 up 90** -> we will decrease velocity of the motor on backward way

- **90 up 180** -> we will increase the speed of motor on forward way

<!-- Mode 2 -->
## 2. Run  Backward
In this mode the max value for write is **90** and the lowest is **0**.

- For **lowest** value we will get the **min** velocity

- For **highest** value we will get the **max** velocity 

- For **config point** the value of that is **0**, on this case

So on this mode:
- **0 up 90** -> we will increase velocity of the motor

<!-- Mode 3 -->
## 3. Run  Backward
In this mode the max value for write is **180** and the lowest is **90**.

- For **lowest** value we will get the **max** velocity

- For **highest** value we will get the **min** velocity 

- For **config point** the value of that is **180**, on this case

So on this mode:
- **90 up 180** -> we will decrease velocity of the motor



### **NOTE:** all of this possibilities are available, and the most quick away to find out where are config point is locking for the led in ESC and change the value of what are being writing and when the led blink you will know this value is the config point

# What exactly is the config point?

The config point is the starting point of our control of velocity, and for a good setup we should start writing this value.
