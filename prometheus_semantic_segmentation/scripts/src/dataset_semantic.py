#!/usr/bin/python3

# Data manipulation
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class DatasetSemantic(torch.utils.data.Dataset):
    def __init__(self,dataset,augmentation=True ,transforms=None):
        # Image dataset paths
        self.augmentation = augmentation
        self.images_original = []
        self.image_label = []
        for image_set in dataset:
            self.images_original.append(image_set[0])
            self.image_label.append(image_set[1])

        self.num_images= len(self.images_original)
        self.transforms = transforms

    def __getitem__(self,index): # returns a specific x,y of the datasets
        # Get the image
        image = Image.open(self.images_original[index]).convert('RGB')

        mask = Image.open(self.image_label[index])

        mask = mask.resize((112, 112), Image.NEAREST)

        # Convert to tensor
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)
        mask = np.array(mask, dtype=np.uint8)
 
        return image , mask
       
    def __len__(self):
        return self.num_images
