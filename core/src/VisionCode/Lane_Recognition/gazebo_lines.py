#!/usr/bin/env python3

# Import the required libraries
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from sklearn.model_selection import GridSearchCV


def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    print([[x1, y1, x2, y2]])
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):

    left_fit = []
    right_fit = []
    if lines is None:
        return None

    for line in lines:
        #2D->1D
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)

        try:
            slope= parameters[0]
            intercept=parameters[1]
        except TypeError:
            slope = 0
            intercept = 0

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))



    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    return np.array((left_line, right_line))

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def canny_edge_detector(image):
    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    th, canny = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canny, 8, cv2.CV_32S)

    label = []
    n = 0
    for i in range(1, num_labels):
        area = stats[i][cv2.CC_STAT_AREA]
        height=stats[i][cv2.CC_STAT_HEIGHT]

        if area > 200 and height>30:
            label.insert(n, i)
            n += 1

    #Só apanhar Maiores partes branca
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(canny, connectivity=8)
    img2 = np.zeros(output.shape)
    for i in range(0,len(label)):
        img2[output == label[i]] = 255

    return img2

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

    #cropped_image = region_of_interest(mask_canny)
    cropped_image = region_of_interest(mask_canny)
    cv2.imshow("results- Cut", cropped_image)
    #lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=20,maxLineGap=200)

    #averaged_lines = average_slope_intercept(cropped_image, lines)

    #line_image = display_lines(cv_image, averaged_lines)
    #combo_image = cv2.addWeighted(cv_image, 0.8, line_image, 1, 1)
    #cv2.imshow("results", combo_image)


    #image_interest = region_of_interest(mask_canny)

    #cv2.imshow("Region of interest in blue", image_interest)


    # Linha horizontal
    # for n in range(0, largura_imagem - 1):
    #     # img2[round(altura_imagem/2-200),n]=255
    #
    #     # Lado esquerdo
    #     if n < round(largura_imagem / 2) and img2[round(altura_imagem / 2) - 199, n] == 255:
    #         print("Obstáculo à esquerda")
    #
    #     if n > round(largura_imagem / 2) and img2[round(altura_imagem / 2) - 199, n] == 255:
    #         print("Obstáculo à direita")
    #
    # # Linha vertical
    # for n in range(0, altura_imagem - 1):
    #     img2[n, round(largura_imagem / 2)] = 255
    #
    # cv2.imshow("Region of interest in blue", img2)
    #
    cv2.waitKey(1)

def main():

    rospy.init_node('Robot_Send', anonymous=True)
    rospy.Subscriber("/robot/camera/rgb/image_raw", Image, Image_GET)
    rospy.spin()

if __name__ == '__main__':
    main()
