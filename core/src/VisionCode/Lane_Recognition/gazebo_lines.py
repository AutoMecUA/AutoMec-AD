#!/usr/bin/env python3

# Import the required libraries
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge


def canny_edge_detector(image):
    BGR_min = (220, 150, 50)
    BGR_max = (240, 200, 150)
#     BGR_min = (50, 150,220)
#     BGR_max = (150, 200, 240)
    pintar_parede = cv2.inRange(image, BGR_min, BGR_max)

    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    th, canny = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canny, 8, cv2.CV_32S)

    #label = []
    n = 0
    max_area = 0
    label = 0
    for i in range(1, num_labels):
        height=stats[i][cv2.CC_STAT_HEIGHT]
        area = stats[i][cv2.CC_STAT_AREA]

        # if area > 200 and height>30:
        #     label.insert(n, i)
        #     n += 1

        if area > max_area and height>30:
            max_area = area
            label = i

    #Só apanhar Maiores partes branca
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(canny, connectivity=8)
    img2 = np.zeros(output.shape)
    img2[output == label] = 255

    #for i in range(0,len(label)):
        #img2[output == label[i]] = 255

    return pintar_parede

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    alpha=0
    #Define vertices
    polygons = np.array([[(0, height), (width, height), (round(width/2), round((height/2)*alpha))]])
    mask = np.zeros_like(image)

    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(mask, image)
    return masked_image

def Image_GET(image):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Câmara Robot", cv_image)

    mask_canny = canny_edge_detector(cv_image)
    cropped_image = region_of_interest(mask_canny)
    #cv2.imshow("Corte Triangular", cropped_image)
    cv2.imshow("Corte Triangular", mask_canny)
    #combo_image = cv2.addWeighted(cv_image, 0.8, line_image, 1, 1)
    #cv2.imshow("results", combo_image)


    #image_interest = region_of_interest(mask_canny)

    #cv2.imshow("Region of interest in blue", image_interest)


    #Linha horizontal
    altura_imagem = cropped_image.shape[0]      #altura=480
    largura_imagem=cropped_image.shape[1]
    vector1x=[]
    vector1y=[]

    #Search for white points in lines
    for t in range (1, 440, 1):
        for n in range(0, largura_imagem - 1):
           if cropped_image[t, n]==255:
                vector1x.append(n)
                vector1y.append(t)

    img2= np.zeros(cropped_image.shape)

    intervalo=0
    #Show image with only lines
    for n in range(1,len(vector1x)):
            img2[vector1y[n],vector1x[n]]=255




        # # Lado esquerdo
        # if n < round(largura_imagem / 2) and cropped_image[round(altura_imagem / 2) - 199, n] == 255:
        #     print("Obstáculo à esquerda")
        #
        # if n > round(largura_imagem / 2) and cropped_image[round(altura_imagem / 2) - 199, n] == 255:
        #     print("Obstáculo à direita")
    cv2.imshow("results- Cut", img2)

    cv2.waitKey(1)

def main():

    rospy.init_node('Robot_Send', anonymous=True)
    rospy.Subscriber("/robot/camera/rgb/image_raw", Image, Image_GET)
    rospy.spin()

if __name__ == '__main__':
    main()
