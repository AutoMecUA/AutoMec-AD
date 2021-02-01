# TODO: check https://data-flair.training/blogs/road-lane-line-detection/
# TODO: check https://divyanshushekhar.com/lane-detection-opencv-python/

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import utils
import numpy as np
import math


def draw_lines(line_img, lines):
    """
    Used by website but code was not shown
    :param line_img:
    :param lines:
    :return:
    """
    pass


def method1(crop: bool = False):
    """
    As in https://towardsdatascience.com/line-detection-make-an-autonomous-car-see-road-lines-e3ed984952c
    Status: incoherent code, video code not sampled
    :return: Opens resulting image of method 1

    Reading images
    Grey Scale
    Gaussian Blur
    Canny Edge Detection
    """

    # Reading images
    img_path = utils.get_abs_path('../images/img4.jpeg')
    img = mpimg.imread(img_path)

    # Grey scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    kernel_size = 5  # Blur level: >5 returns error
    gauss_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

    # Canny edge detection
    low_threshold, high_threshold = [200, 300]
    canny_img = cv2.Canny(gauss_img, low_threshold, high_threshold)

    if not crop:
        # Show image
        plt.imshow(canny_img)
        plt.show()
    else:
        # MASK A REGION OF INTEREST
        # Setting the corners of the trapezium
        vertices = np.array([[(0, img_line.shape[0]), (img_line.shape[1], img_line.shape[0]), (400, 260),
                              (600, 260)]])  # make a blank/white image
        mask = np.zeros_like(img)
        mask_channels = (255,) * img.shape[2]  # Fill the area of interest with 0 and 255 these
        # which lie outside of it, thoughout all color channels
        cv2.fillPoly(mask, vertices, mask_channels)  # Keep only the pixels with 0 value of the canny_img
        masked_img = cv2.bitwise_and(canny_img, mask)

        # Hough lines detector
        lines = cv2.HoughLinesP(img, rho=1, theta=math.pi / 180,
                                threshold=15, lines=np.array([]),
                                minLineLength=30,
                                maxLineGap=40)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        # Find the road lines
        draw_lines(line_img, lines)

        # ...
        out_img = cv2.addWeighted(img, 0.9, img_lines, 1.0, 0.0)

        # Show image
        plt.imshow(out_img)
        plt.show()


def method2():
    """
     from https://valueml.com/autonomous-lane-detection-for-self-driving-cars-in-python/
     Status: bug, video code not sampled
    :return:
    """
    # Loading lane image
    img_path = utils.get_abs_path("../images/img4.jpeg")
    image_c = cv2.imread(img_path)

    # Gray Scale
    image_g = cv2.cvtColor(image_c, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    image_blur = cv2.GaussianBlur(image_g, (7, 7), 0)

    # Canny filters
    threshold_low = 10
    threshold_high = 200
    image_canny = cv2.Canny(image_blur, threshold_low, threshold_high)

    # Masking
    vertices = np.array([(20, 950), (350, 650), (650, 650), (1000, 950)])
    mask = np.zeros_like(image_g)
    cv2.fillPoly(mask, np.int32([vertices]), 255)

    # Canny + Masking
    masked_image = cv2.bitwise_and(image_canny, mask)

    # HOUGH LINES DETECTION AND DRAW FUNCTION
    rho = 2
    theta = np.pi / 180
    threshold = 40
    min_line_len = 100
    max_line_gap = 50
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_image = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), [255, 0, 255], 35)

    # Identification
    a = 1
    b = 1
    c = 1
    image = cv2.addWeighted(image_c, a, line_image, b, c)

    plt.figure()
    plt.imshow(image)
    # plt.show()


if __name__ == '__main__':
    # method3('../images/img4.jpeg')
    ...
