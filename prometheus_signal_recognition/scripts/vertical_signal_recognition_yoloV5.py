#!/usr/bin/env python3

"""
    I need to say which image to label
    Implement the model (best.pt)
    Display the image with the label
"""

# Imports
import cv2
import torch
import rospy
import argparse
import numpy as np
from typing import Any
from pathlib import Path
from functools import partial
from cv_bridge.core import CvBridge
from sensor_msgs.msg import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

# My imports
from src.models.common import DetectMultiBackend
from src.utils_yolo.general import non_max_suppression, scale_coords, check_img_size
from src.utils_yolo.utils_yolo import time_sync, letterbox, Annotator, colors


def imgRgbCallback(message, config):
    """Callback for changing the image.
    Args:
        message (Image): ROS Image message.
        config (dict): Dictionary with the configuration. 
    """

    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "bgr8")

    config["begin_img"] = True


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'src/data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.1, #0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        name='exp',  # save results to project/name
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        image=None,
        ):
    
    # Load model
    #device = select_device(device)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #cuda: 0 index of gpu
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    names, pt = model.names, model.pt

    imgsz = check_img_size(imgsz)  # check image size

    im0s = image  # dataset

    im = image.transpose((2, 0, 1))   # dataset
    
    im = letterbox(image, imgsz, stride=32, auto=True)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    cv2.imshow('image 2', im.transpose(1,2,0))  
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
 
    # Inference
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    # Process predictions
    im0, _ = im0s.copy(), getattr(image, 'frame', 0)
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    imc = im0.copy() if save_crop else im0  # for save_crop
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

    for det in pred:
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    # Stream results
    im0 = annotator.result()
    cv2.imshow('image', im0)   
   
    key = cv2.waitKey(1)
    if key == ord('q'):
        rospy.loginfo('Leter "q" pressed, exiting the program')
        cv2.destroyAllWindows()
        rospy.signal_shutdown("Manual shutdown")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    opt = parser.parse_args()
    return opt


def main(opt):
    # Defining starting values
    config: dict[str, Any] = dict(img_rgb=None, bridge=None, begin_img=False)

    config["begin_img"] = False
    config["bridge"] = CvBridge()

    # Init Node
    rospy.init_node('vertical_signal_recognition', anonymous=False)

    # Getting parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_right_camera/image_raw')

    imgRgbCallback_part = partial(imgRgbCallback, config = config)

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)

    # Frames per second
    rate = rospy.Rate(30)

    ############################
    # Main loop                #
    ############################
    while not rospy.is_shutdown():
        # If there is no image, do nothing
        if config["begin_img"] is False:
            continue

        ############################
        # Predicts the steering    #
        ############################
        image = config['img_rgb']
    
        opt.source = image
        run(**vars(opt),image=image)
       
        rate.sleep()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)