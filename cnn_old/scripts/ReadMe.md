# Model Theory

This model is based on the arquitecture of NVIDIA's dave-2 CNN to replicate end to end driving model.

It uses an IMG as in input and outputs an angle.
In this specific case we only use 1 camera.

![NVIDIA](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/training-624x291.png)

The model arquitecture in this repo is pretty close to the original, soon we will inplemet some Dropout layers as the testing goes on. 


![Arquitetura](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

# Dependencies
-tensorflow 
-sklearn

Use pip , or pip3 to install. 
````bash
pip install -U scikit-learn

````
````bash
pip install --upgrade tensorflow
````
# Collect Data
To Collect Data First open gazebo and spawn the robot , run the following on two separate terminal windows

```bash
roslaunch robot_bringup bringup_gazebo.launch
roslaunch robot_bringup spawn.launch
```

Now run the "write_data.py".



Then all you have to do is drive the car.  To open up the controller
```bash
rqt
```


The images will be stored under /data/IMG and you will now notice a file named "driving_log.csv"  being created.

# Training the model

After that run the script **"TrainingSimulation.py"**

You will be asked if you want to create a new model. **Say No (N)**.
If you wish to train a new model from zero (press y)

This will create a model called "model_teste.h5" if you have a good working model , change the name of this file name, ex "model_20_03_2021.h5" and feel free to commit it under the folder models_files.

## Intreperting the results

Two images will show up when training, first , the steering angle distribuition. You should have a semi equal left / right distribuiton and mostly 0 degrees. The secound plot trims the excess biased data. You can ignore these plots for now.

You will then see a plot of the loss function. Please note if you have a very small number of epochs these numbers are meaningless. 


## Ajusting the FrameRate (HZ)
You can ajust the Frame Rate of capture under the file "write_data.py" on the save_IMG() method 

# Driving
Run the ml_driving-dave.py script
