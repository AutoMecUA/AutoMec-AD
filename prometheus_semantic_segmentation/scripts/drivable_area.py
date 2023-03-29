#!/usr/bin/python3

# Imports 
from functools import partial
import time
from typing import Any
import cv2

import numpy as np
import rospy
import yaml
import os
from colorama import Fore, Style
import torch
from torchvision import transforms
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
from collections import namedtuple

#  custom imports
from models.deeplabv3 import createDeepLabv3
from models.yolop import yolop


def imgRgbCallback(message, config):

    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "passthrough")

    config["begin_img"] = True

def preProcess(img, img_width=597, img_height=91):  
    img = cv2.resize(img, (img_width, img_height))  

    return img 


# Main code
def main():
    ########################################
    # Initialization                       #
    ########################################            
    config: dict[str, Any] = dict(vel=None, img_rgb=None,
                                  bridge=None, begin_img=None)
    
    PIL_to_Tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    tensor_to_pil_image = transforms.ToPILImage()

    # Defining starting values
    config["begin_img"] = False
    config["bridge"] = CvBridge()

    win_name = 'Semantic Segmentation'
    cv2.namedWindow(winname=win_name,flags=cv2.WINDOW_NORMAL)

    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_front_camera/rgb/image_raw')
    model_name = rospy.get_param('/model_semantic_name', 'segmentation')

    rospy.init_node('semantic_segmentation', anonymous=False)

    # General Path
    automec_path=os.environ.get('AUTOMEC_DATASETS')
    path = f'{automec_path}/models/{model_name}/{model_name}.pkl'

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    # Retrieving info from yaml
    with open(f'{automec_path}/models/{model_name}/{model_name}.yaml') as file:
        info_loaded = yaml.load(file, Loader=yaml.FullLoader)

    rospy.loginfo('Using model: %s', path)
    config['model'] = yolop()
    config['model'].to(device)
    config['model'].eval()
    
    imgRgbCallback_part = partial(imgRgbCallback, config=config)

    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)

    # Frames per second
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        if config["begin_img"] is False:
            continue
        preivous_time = time.time()
        # Obtain segmented image
        resized_img = preProcess(config["img_rgb"])
        image = np.array(resized_img)
        image = PIL_to_Tensor(image)
        image = image.unsqueeze(0)
        image = image.to(device, dtype=torch.float)
        det_out, da_seg_out,ll_seg_out = config['model'](image)
        
        _, da_seg_mask = torch.max(da_seg_out, 1)
        da_seg_mask = da_seg_mask.cpu().int().squeeze().numpy()
        print(da_seg_mask.shape)
        da_seg_mask = da_seg_mask.astype(np.uint8)
        da_seg_mask = 255*da_seg_mask

        print(f'FPS: {1/(time.time()-preivous_time)}')
        cv2.imshow(win_name, da_seg_mask)
        key = cv2.waitKey(1)
        # Publish angle
        rate.sleep()


if __name__ == '__main__':
    main()