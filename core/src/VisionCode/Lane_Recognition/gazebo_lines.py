#!/usr/bin/env python3

# References:
# [1] - https://note.nkmk.me/en/python-numpy-opencv-image-binarization/

# Import the required libraries
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge


def canny_alternate(image):

    #For discarding colors
    for i, row in enumerate(image):
        for j, pixel in enumerate(row):
            r, g, b = image[i][j]
            # How much can pixels deviate from black/white color
            deviation = 55
            minim, maxim = deviation, 255 - deviation
            # For discarding colored pixels (road is not colored)
            # if any of r, g or b is outside the spectrum [0, 55] U [215, 255]
            if any([minim < color < maxim for color in (r, g, b)]):
                # Paint black
                image[i][j] = (0, 0, 0)

    # RGB to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    th = 30  # reasonably working (best value?)

    # Binary image
    ret, bin_img = cv2.threshold(gray_image, th, 255, cv2.THRESH_BINARY)

    # TODO (obsolete?) Perform white pixel clusters removal: goal is to remove unwanted regions
    #   - if this task is achieved, region_of_interest becomes obsolete

    return bin_img


def canny_edge_detector(image):
    BGR_min = (220, 150, 50)
    BGR_max = (240, 200, 150)
    # BGR_min = (50, 150,220)
    # BGR_max = (150, 200, 240)

    pintar_parede = cv2.inRange(image, BGR_min, BGR_max)

    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    th, canny = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canny, 8, cv2.CV_32S)

    # label = []
    n = 0
    max_area = 0
    label = 0
    for i in range(1, num_labels):
        height = stats[i][cv2.CC_STAT_HEIGHT]
        area = stats[i][cv2.CC_STAT_AREA]

        # if area > 200 and height>30:
        #     label.insert(n, i)
        #     n += 1

        if area > max_area and height > 30:
            max_area = area
            label = i

    # Só apanhar Maiores partes branca
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(canny, connectivity=8)
    img2 = np.zeros(output.shape)
    img2[output == label] = 255

    # for i in range(0,len(label)):
    # img2[output == label[i]] = 255

    return pintar_parede


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    alpha = 0
    # Define vertices
    polygons = np.array([[(0, height), (width, height), (round(width / 2), round((height / 2) * alpha))]])
    mask = np.zeros_like(image)

    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(mask, image)
    return masked_image


def get_coeffs(bin_image, degree: int) -> list:
    """

    :param bin_image: pixels' value must either be 0 or 255 (in gray scale)
    :param degree:
    :return:
    """

    height = bin_image.shape[0]
    vector = {"x": list(), "y": list()}
    x, y = vector["x"], vector["y"]

    for i, row in enumerate(bin_image):
        # cropped_image[i] == row
        for j, pixel in enumerate(row):
            # row[j] == pixel
            if pixel == 255:
                x.append(j)
                y.append(height - i)  # Top row should be largest and bottom row == 0

    # Robustness check
    assert len(y) == len(x), "Error: Vectors x and y are not of same length!"

    return np.polyfit(x=x, y=y, deg=degree)


def unify_line(bin_image, side: str = "", average: bool = True):
    """

    :param average: pixel averaging? Yes if set True
    :param bin_image: image of the road: expected to be of binary form
    :param side: left, right or center (obsolete for now)
    :return: image with only one of the lane lines visible

    Algorithm (pseudocode):
    for row in image:
      for x, pixel in enum(row):  # x == column number of pixel
          if pixel is white:
              add x to list; set row[x] (pixel) black
      row[list.average] = white
      flush list

    """

    assert side in ["right", "left"]  # side is left or right

    whites: list = list()
    vertical_line: float = 0.55  # line on the right/left of which pixels are not accounted for
    image_width = bin_image.shape[1]

    def check_line(col: int):
        """

        :param col: Column number of a given pixel
        :return:True if pixel is to the right and side=="right",
                    also if pixel is to the left and side=="left".
                False otherwise
        """
        right: bool = col > vertical_line * image_width  # if true, pixel is to the right of line
        if right and side == "right":
            return True
        elif not right and side == "left":
            return True
        else:
            return False

    # Tentar colocar apenas a maior área
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_image, 8, cv2.CV_32S)
    # max_area = 0
    # label = 0
    #
    # for i in range(1, num_labels):
    #     area = stats[i][cv2.CC_STAT_AREA]
    #
    #     if area > max_area:
    #         max_area = area
    #         label=num_labels
    #
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bin_image, connectivity=8)
    # img2 = np.zeros(output.shape)
    # img2[output == label] = 255

    for row in bin_image:
        for x, pixel in enumerate(row):
            if check_line(x):
                if average:  #
                    if pixel == 255:  # pixel is white
                        whites.append(x)
                        row[x] = 0  # set black
            else:
                # Also set black because it's not an interest region
                row[x] = 0
        # ...
        if average:
            try:
                av: int = int(np.average(whites))
            except ValueError:
                whites = list()
                continue
            row[av] = 255
            whites = list()  # flush

    return bin_image


def quadratic_image(coeffs: list,
                    width: int, height: int):

    image = np.zeros((height, width), np.uint8)

    for x in range(width):
        y = sum(
            [ak * x ** (len(coeffs) - i) for i, ak in enumerate(coeffs)]
        )  # a0x**n + a1x**n-1 + ... + an -> poly equation
        row = height - y
        if 0 <= row < height:
            image[row][x] = 255

    return image


def Image_GET(image):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Camara Robot", cv_image)
    altura_imagem, largura_imagem = cv_image.shape[:2]  # altura=480

    # Binary image
    alt_img = canny_alternate(cv_image)
    cv2.imshow("binarized image (test)", alt_img)

    # Select one road line
    alt_img = unify_line(alt_img, side="right", average=False)
    cv2.imshow("Test single curve", alt_img)

    # Get a, b and c such as ax2 + bx + c is the best fit to given image
    coeffs: list = get_coeffs(alt_img, degree=2)

    # Draw a graph with the estimated curve
    image_curve = quadratic_image(coeffs=coeffs, width=largura_imagem,
                                  height=altura_imagem)  # de-comment when bin_mask_canny is good

    cv2.imshow("Quadratic regression result", image_curve)

    cv2.waitKey(1)


def main():
    rospy.init_node('Robot_Send', anonymous=True)
    rospy.Subscriber('/robot/camera/rgb/image_raw', Image, Image_GET)
    rospy.spin()


if __name__ == '__main__':
    main()
