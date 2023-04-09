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
from PIL import Image as IMG
from cv_bridge.core import CvBridge
from collections import namedtuple

#  custom imports
from models.deeplabv3 import createDeepLabv3
from models.deeplabv3_v1 import deeplabv3_v1
from models.deeplabv3_resnet50 import createDeepLabv3_resnet50
from models.yolop import yolop
from models.espnetv2_bdd100k_driveable import Model
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

    #config['img_rgb'] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    config["begin_img"] = True

def preProcess(img, img_width=112, img_height=112):  
    img = cv2.resize(img, (img_width, img_height))  

    return img 


# Main code
def main():
    ########################################
    # Initialization                       #
    ########################################
    labels = [
            #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
            Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
            Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
            Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
            Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
            Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
            Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
            Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
            Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
            Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
            Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
            Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
            Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
            Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
            Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
            Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
            Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
            Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
            Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
            Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
            Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
            Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
            Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
            Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
            Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
            Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
            Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
            Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
            Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]
            
    config: dict[str, Any] = dict(vel=None, img_rgb=None,
                                  bridge=None, begin_img=None)
    
    PIL_to_Tensor = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    print(info_loaded['model']['ml_arch']['name'])
    config['model'] = eval(info_loaded['model']['ml_arch']['name'])
    config['model'] = LoadModel(path,config['model'],device)
    config['model'].eval()

    mask = np.array([label.color for label in labels], dtype = np.uint8)

    #config['model'] = Model()
    
    imgRgbCallback_part = partial(imgRgbCallback, config=config)

    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)

    # Frames per second
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        if config["begin_img"] is False:
            continue
        preivous_time = time.time()
        # Obtain segmented iamage
        config['img_rgb'] = IMG.open('/media/andre/Andre/Automec/datasets/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png')
        config['img_rgb'] = np.asarray(config['img_rgb'])
        resized_img = preProcess(config["img_rgb"])
        image = PIL_to_Tensor(resized_img)
        image = image.unsqueeze(0)
        image = image.to(device, dtype=torch.float)
        mask_predicted = config['model'](image)
        mask_predicted_output = mask_predicted['out']

        mask_predicted_output = mask_predicted_output.detach().cpu()

        mask_predicted_output = tensor_to_pil_image(mask_predicted_output[0])

        #mask_predicted_array = np.array(mask_predicted_output)
        # print(mask_predicted_array.shape)

        # mask_color = np.zeros([112, 112, 3], dtype=np.uint8)
        # for i in range(mask_color.shape[0]):
        #     for j in range(mask_color.shape[1]):
        #         for label in labels:
        #             if np.asarray(mask_predicted_array)[i,j] == label.id:
        #                 mask_color[i,j,:] = (label.color[2], label.color[1], label.color[0])

        print(f'FPS: {1/(time.time()-preivous_time)}')
        cv2.imshow(win_name, cv2.cvtColor(mask[mask_predicted_output] , cv2.COLOR_RGB2BGR) )
        #cv2.imshow(win_name, np.array(mask_predicted_output))
        key = cv2.waitKey(1)
        # Publish angle
        rate.sleep()


if __name__ == '__main__':
    main()