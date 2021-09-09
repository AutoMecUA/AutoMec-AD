import cv2
import numpy as np
#import matplotlib.pyplot as plt

dim=(680,480)

left = cv2.imread('Left.png',cv2.IMREAD_COLOR)
right = cv2.imread('Right.png',cv2.IMREAD_COLOR)


# Nota. funciona melhor com o resize ( dimensão aleatoria, mudar depois)
#left = cv2.resize(left,dim,interpolation = cv2.INTER_AREA)
#right = cv2.resize(right,dim,interpolation = cv2.INTER_AREA)


#img = cv2.imread('dumb.jpg')
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


# cv2.imshow('image',left)
# cv2.waitKey(0)
# cv2.destroyAllWindows()