#!/usr/bin/python3

# Visualization tools
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px

# Data manipulation
from imgaug import augmenters as iaa
import pandas as pd
import numpy as np
import glob
import cv2

import torch
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self,dataset,dataset_path, transform=None):
        # Image dataset paths
        images_path = dataset_path + '/IMG/'
        self.image_filenames_original = images_path + dataset['img_name']
        self.image_filenames_original = self.image_filenames_original.values.tolist()
        self.labels_original = dataset['steering'].values.tolist()
        self.num_images= len(self.image_filenames_original)
        self.image_width = 256
        self.image_height = 256
        self.transforms = transform

    def __getitem__(self,index): # returns a specific x,y of the datasets
        # Get the image
        image = Image.open(self.image_filenames_original[index])
        label = self.labels_original[index]
        image = self.transforms(image)

        return image , label
       
    def __len__(self):
        return self.num_images
