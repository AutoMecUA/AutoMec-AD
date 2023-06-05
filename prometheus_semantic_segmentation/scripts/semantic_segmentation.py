#!/usr/bin/python3

# Imports 
from functools import partial
import time
from typing import Any
import cv2
from matplotlib import pyplot as plt

import numpy as np
import rospy
import yaml
import os
from colorama import Fore, Style
import torch
from torchvision import transforms
from sensor_msgs.msg._Image import Image
from PIL import Image as IMG
from cv_bridge.core import CvBridge
from collections import namedtuple

#  custom imports
from models.deeplabv3_resnet50 import DeepLabv3
from models.segnet import SegNet
from models.segnetV2 import SegNetV2
from models.unet import UNet
from src.utils import LoadModel

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


def imgRgbCallback(message, config):

    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "passthrough")

    config['img_rgb'] = cv2.cvtColor(config['img_rgb'], cv2.COLOR_BGR2RGB)

    config["begin_img"] = True


# Main code
def main():
    ########################################
    # Initialization                       #
    ########################################
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'background'            ,  0 ,      0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'crosswalk'             ,  1 ,      1 , 'void'            , 0       , False        , True         , (255,255,255) ),
        Label(  'driveable'             ,  2 ,      2 , 'void'            , 0       , False        , True         , (128, 64,128) ),
        Label(  'driveable_alt'         ,  3 ,      3 , 'void'            , 0       , False        , True         , (244, 35,232) ),
        Label(  'parking'               ,  4 ,      4 , 'void'            , 0       , False        , True         , (250,170,160) ),
    ]

    mask = np.array([label.color for label in labels if label.trainId >= -1 and label.trainId <= 19] ,dtype='uint8')
            
    config: dict[str, Any] = dict(vel=None, img_rgb=None,
                                  bridge=None, begin_img=None)
    
    # Defining starting values
    config["begin_img"] = False
    config["bridge"] = CvBridge()

    win_name = 'Semantic Segmentation'
    cv2.namedWindow(winname=win_name,flags=cv2.WINDOW_NORMAL)

    rospy.init_node('semantic_segmentation', anonymous=False)
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_front_camera/rgb/image_raw')
    model_name = rospy.get_param('/model_semantic_name', '')

    # General Path
    automec_path=os.environ.get('AUTOMEC_DATASETS')
    path = f'{automec_path}/models/{model_name}/{model_name}.pkl'

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    # Retrieving info from yaml
    with open(f'{automec_path}/models/{model_name}/config.yaml') as file:
        info_loaded = yaml.load(file, Loader=yaml.FullLoader)

    rospy.loginfo('Using model: %s', path)
    print(info_loaded['model']['ml_arch']['name'])
    config['model'] = eval(info_loaded['model']['ml_arch']['name'])
    config['model'] = LoadModel(path,config['model'],device)
    config['model'].eval()
    rgb_mean = info_loaded['model']['rgb_mean']
    rgb_std = info_loaded['model']['rgb_std']
    image_size = info_loaded['model']['image_size']

    # Pre-processing
    config['transforms'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std)
    ])
    
    imgRgbCallback_part = partial(imgRgbCallback, config=config)

    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)

    # Frames per second
    rate = rospy.Rate(30)

    ###########################
    # Obtains the mask        #
    ###########################
    while not rospy.is_shutdown():
        # Checks if there is an image in the buffer
        if config["begin_img"] is False:
            continue
        
        image= config["img_rgb"]
        # Obtain segmented iamage
        image = config['transforms'](image)
        image = image.unsqueeze(0)
        image = image.to(device, dtype=torch.float)
        mask_predicted = config['model'](image)
        if info_loaded['model']['ml_arch']['name'] == 'DeepLabV3':
            mask_predicted_output = mask_predicted['out'][0]
        else:
            mask_predicted_output = mask_predicted[0]
        mask_predicted_output = mask_predicted_output.argmax(0)
        mask_predicted_output = mask_predicted_output.byte().cpu().numpy()
        mask_color = mask[mask_predicted_output].astype(np.uint8)

        cv2.imshow(win_name, cv2.cvtColor( mask_color, cv2.COLOR_RGB2BGR) )
        key = cv2.waitKey(1)
        # Publish angle
        rate.sleep()


if __name__ == '__main__':
    main()