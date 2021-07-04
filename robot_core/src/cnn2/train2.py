import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

image_path = './data/dataset08/IMG/'

total_left_angles = 0
total_right_angles = 0
total_straight_angles = 0

images = []
angles = []
velocity = []
with open('./data/dataset08/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    k = 0
    for line in reader:
        # k += 1
        # if (k%2)==0:
        #     continue
        if k==1: continue
        center_image = cv2.imread(image_path + line[0],0)
        center_image = cv2.resize(center_image, (320, 160))
        center_image_temp = np.expand_dims(center_image, axis=2)

        images.append(center_image_temp)
        angles.append(float(line[1]))
        velocity.append(float(line[2]))
        
        # Change image and store the same angle, with some distortion, both in image and in angle (to increase robustness)
        
        # flipped
        
        center_image_temp = np.expand_dims(cv2.flip(center_image, 1), axis=2)

        images.append(center_image_temp)
        angles.append(-float(line[1]))
        velocity.append(float(line[2]))
        
        # add more distortions
        # ... 
        
        if(float(line[1]) < -0.1):
            total_left_angles += 1
        elif(float(line[1]) > 0.1):
            total_right_angles += 1
        else:
            total_straight_angles += 1


left_to_straight_ratio = total_straight_angles/total_left_angles
right_to_straight_ratio = total_straight_angles/total_right_angles

print('angles are: ' + str(list(angles)))

print('Total Samples : ', len(images))
print()
print('Initial Angle Distribution')
print('Total Left Angles : ', total_left_angles)
print('Total Right Angles : ', total_right_angles)
print('Total Straight Angles : ', total_straight_angles)
print('Left to Straight Ratio : ', left_to_straight_ratio)
print('Right to Straight Ratio : ', right_to_straight_ratio)


images = np.asarray(images)
images = images.astype('float32')
images /= 255
target = np.column_stack((angles,velocity))



print('####################################3')
print(target[:,1])
print('####################################3')



X_train, X_test, y_train, y_test = train_test_split(images, target, test_size=0.00001)


train_samples_size = len(X_train)
validation_samples_size = len(X_test)



# print('####################################3')
# print(y_train[0])
# print('####################################3')






total_left_angles = 0
total_right_angles = 0
total_straight_angles = 0

for train_sample in y_train[:,0]:
    if(float(train_sample) < -0.15):
        total_left_angles += 1
    elif(float(train_sample) > 0.15):
        total_right_angles += 1
    else:
        total_straight_angles += 1

left_to_straight_ratio = 0
right_to_straight_ratio = 0

left_to_straight_ratio = total_straight_angles/total_left_angles
right_to_straight_ratio = total_straight_angles/total_right_angles

print()
print('Train Sample Size : ', train_samples_size)
print('Validation Sample Size : ', validation_samples_size)

print()
print('After Train-Test split, Angle Distribution')
print('Total Left Angles : ', total_left_angles)
print('Total Right Angles : ', total_right_angles)
print('Total Straight Angles : ', total_straight_angles)
print('Left to Straight Ratio : ', left_to_straight_ratio)
print('Right to Straight Ratio : ', right_to_straight_ratio)


batch_size = 10
nb_epoch = 40



model = Sequential()
model.add(Convolution2D(24, (5,5),(2, 2), input_shape = (160,320,1), activation='elu'))
model.add(Convolution2D(36,(5,5),(2, 2), activation='elu'))
model.add(Convolution2D(48, (5,5),(2, 2), activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Flatten())
model.add(Dense(100,activation='elu'))
model.add(Dense(50,activation='elu'))
model.add(Dense(10,activation='elu'))
model.add(Dense(2))

model.summary()

model.compile(Adam(lr=0.0001),loss='mse')
model.fit(X_train, y_train,epochs=nb_epoch, validation_data = (X_test, y_test))

model.save('cnn2-model5.h5')
