#!/usr/bin/python3

# Data manipulation
from imgaug import augmenters as iaa
import numpy as np
import cv2

import torch
from PIL import Image
from torchvision import transforms

class DatasetSemantic(torch.utils.data.Dataset):
    def __init__(self,dataset,augmentation=True):
        # Image dataset paths
        self.augmentation = augmentation
        self.images_original = []
        self.image_label = []
        for image_set in dataset:
            self.images_original.append(image_set[0])
            self.image_label.append(image_set[1])
        self.num_images= len(self.images_original)
        self.image_width = 224
        self.image_height = 224
        # Create a set of transformations
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transforms_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def pre_processing(self,img, img_width=320, img_height=160, normalization="yes"):
        # Cropping Region of intrest, Ajust with Gazebo and use Andre Code in the Future
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # For better jornalization 
        
        # img = cv2.GaussianBlur(img, (3, 3), 0) 
        
        img = cv2.resize(img, (img_width, img_height))  # NIVIDA uses 200x66
        #TODO normalize with mean and std
        # if normalization == "yes":
        #     img = img/255

        return img 

    def augmentImage(self,imgPath, imgLabel):
        # Function: Add randomness to the data set by applying random "filters"
        img = Image.open(imgPath)

        img_label = Image.open(imgLabel)

        img = np.asarray(img)

        img_label = np.asarray(img_label)

        #img = mpimg.imread(imgPath)

        # Translation... PAN
        if np.random.rand() < 0.30:
            pan = iaa.Affine(translate_percent = {'x': (-0.1, 0.1) , 'y': (-0.1, 0.1)} )
            img = pan.augment_image(img)  # Add a pan to img 5555 5   

        # Zoom 
        if np.random.rand() < 0.30:
            zoom = iaa.Affine(scale=(1, 1.2))
            img = zoom.augment_image(img)

        # Brightness
        if np.random.rand() < 0.30:
            brightness = iaa.Multiply((0.2, 1.2))
            img = brightness.augment_image(img)
        
        #Motion Blur
        if np.random.rand() < 0.30:
            motion_blur = iaa.MotionBlur(k=15, angle=[-60, 60])
            img = motion_blur.augment_image(img)
            
        #Contrast  
        if np.random.rand() < 0.30:
            lin_contrast = iaa.LinearContrast((0.5, 1.9))
            img = lin_contrast.augment_image(img)
        
        # Shear Operation
        if np.random.rand() < 0.30:
            shear_img = iaa.Affine(shear=(-8, 8))
            img = shear_img.augment_image(img)
        
        # Noise
        if np.random.rand() < 0.30:
            noise = iaa.SaltAndPepper(0.05)
            img = noise.augment_image(img)
        
        # Random Ereasing / Occlusion
        if np.random.rand() < 0.30:
            occlusion = iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)
            img = occlusion.augment_image(img)
            
        # # Flip
        # if np.random.rand() < 0.30:
        #     img = cv2.flip(img, 1)  
        #     steering = - steering

        return img, img_label

    def __getitem__(self,index): # returns a specific x,y of the datasets
        # Get the image
        if self.augmentation == True:
            image, label = self.augmentImage(self.images_original[index], self.image_label[index])
        else:
            image = Image.open(self.images_original[index])
            image = np.asarray(image)
            label = Image.open(self.image_label[index])
            label = np.asarray(label)
        # Preprocess the image
        image = self.pre_processing(image, self.image_width, self.image_height)
        label = self.pre_processing(label, self.image_width, self.image_height)
        image = self.transforms_tensor(np.array(image))
        label = self.transforms_tensor(np.array(label))

        return image , label
       
    def __len__(self):
        return self.num_images
