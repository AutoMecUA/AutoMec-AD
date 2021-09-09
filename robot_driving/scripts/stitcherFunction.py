import numpy as np
import cv2
import numpy as np
#import matplotlib.pyplot as plt


# Recieves ImageLeft, Image Right as input and dimesions to resize it , default = (640,480)
# Example:
#  left = cv2.imread('Left.png',cv2.IMREAD_COLOR)
# right = cv2.imread('Right.png',cv2.IMREAD_COLOR)
# dim = (640, 480)

def stitching(left,right,dim=(640,480)):

    # Nota. as vezes funciona melhor com o resize
    left = cv2.resize(left,dim,interpolation = cv2.INTER_AREA)
    right = cv2.resize(right,dim,interpolation = cv2.INTER_AREA)


    images = []

    images.append(left)
    images.append(right)

    stitcher = cv2.Stitcher.create()
    ret,pano = stitcher.stitch(images)

    if ret == cv2.STITCHER_OK:
        cv2.imshow('Panorama',pano)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print('Error during stitching')

