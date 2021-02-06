#!/usr/bin/env python3

# Import the required libraries
import cv2
import numpy as np


def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
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

    # Reduce noise from the image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def main():
    image_rgb = cv2.imread('/home/sf/Desktop/AutoMEC/catkin_ws/src/AutoMec-AD/robot_description/ImageGazebo/image1.png', 1)
    mask_canny=canny_edge_detector(image_rgb)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_canny, 8, cv2.CV_32S)

    label=[]
    n=0
    for i in range(1, num_labels):
        height = stats[i][cv2.CC_STAT_HEIGHT]
        if height > 170:
            label.insert(n,i)
            n+=1

    # Só apanhar Maiores partes branca
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_canny, connectivity=8)
    img2 = np.zeros(output.shape)
    for i in range(0,len(label)):
        img2[output == label[i]] = 255

    lines = cv2.HoughLinesP(mask_canny,2, np.pi / 180, 100,np.array([]), minLineLength=500,maxLineGap=20000)
    averaged_lines = average_slope_intercept(img2, lines)
    line_image = display_lines(image_rgb, averaged_lines)
    combo_image = cv2.addWeighted(image_rgb, 0.8, line_image, 1, 1)
    # cv2.imshow("results", combo_image)



    cv2.imshow("Region of interest in blue",combo_image)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
