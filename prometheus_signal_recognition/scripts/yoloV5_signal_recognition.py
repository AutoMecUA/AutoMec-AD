#!/usr/bin/env python3

"""
    I need to say which image to label
    Implement the model (best.pt)
    Display the image with the label
"""

# Imports
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

# My imports
from src.models.common import DetectMultiBackend
from src.utils_yolo.general import (LOGGER, check_file, check_img_size, check_imshow, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from src.utils_yolo.utils_yolo import (IMG_FORMATS, LoadImages, Annotator, colors, save_one_box,
                           select_device, time_sync)

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'src/data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
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
        ):

    source = str(source)

    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    names, pt = model.names, model.pt

    imgsz = check_img_size(imgsz)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, auto=pt)
    bs = 1  # batch_size
  
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    path, im, im0s = dataset

    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3
    
    # Process predictions
    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

    p = Path(p)  # to Path
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
                #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    
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
    cv2.imwrite('catkin_ws/src/AutoMec-AD/prometheus_signal_recognition/scripts/runs/detect/exp22/Sign3.jpeg', im0)
    window_name = 'image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 300, 400)
    cv2.imshow(window_name,im0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT, help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
   


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)