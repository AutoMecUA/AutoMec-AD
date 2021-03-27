#!/usr/bin/env python3

# References:
# [1] - https://note.nkmk.me/en/python-numpy-opencv-image-binarization/
from geometry_msgs.msg import Twist

global cv_image
global see_image
global get_out
# Import the required libraries
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge


def canny_alternate(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    th = 100 # (to keep the wall out of the binary image, 100 does the work)

    _, bin_img = cv2.threshold(gray_image, th, 255, cv2.THRESH_BINARY)

    return bin_img


def get_coeffs(bin_image) -> list:
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

    # Here the codes searchs for the best corresondent degree
    degree = 0
    get_out = False

    while True:
        coeffs = np.polyfit(x=x, y=y, deg=degree)
        for i in range(0, len(coeffs)):
            if abs(coeffs[0]) < 1:
                get_out = True
                break

        if get_out:
            break

        radius_stats(coeffs=coeffs,
                     exes=x)
        degree += 1

    return np.polyfit(x=x, y=y, deg=degree-1) if len(x) != 0 else None


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

    if coeffs is None:
        print("No white pixels found!")
        return None

    image = np.zeros((height, width), np.uint8)
    for x in range(width):
        y = sum(
            [ak * x ** (len(coeffs) - i - 1) for i, ak in enumerate(coeffs)]
        )  # a0x**n + a1x**n-1 + ... + an -> poly equation
        row = int(height - y)
        if 0 <= row < height:
            image[row][x] = 255

    return image


def Image_GET(image):
    global cv_image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    global see_image
    see_image=True


def biggest_area(image):
    max_area = 0
    label = 0
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    for i in range(1, nb_components):
        area = stats[i][cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            label = i
    img2 = np.zeros(output.shape)
    img2[output == label] = 255

    return img2


def radius_stats(coeffs: list, exes: list):

    if len(coeffs) == 3:
        return

    radiuses: list = [radius_2poly(*coeffs, result) for result in exes]

    print(f"Average: {np.average(radiuses)}",
          f"Standard deviation {np.std(radiuses)}",
          sep="; ", end=".")


def radius_2poly(a: float, b: float,
                 c: float, x: float) -> float:
    """
    return: return radius of a given 2nd degree polynomial curve
    """

    y: float = a*x**2 + b*x + c

    numerator: float = 1 + (2*a*y + b) ** 2

    numerator = numerator ** (3/2)

    result = numerator / abs(2*a)

    return result


def main():
    global cv_image
    rospy.init_node('Robot_Send', anonymous=True)
    rospy.Subscriber('/robot/camera/rgb/image_raw', Image, Image_GET)
    velocity_publisher = rospy.Publisher('/robot/cmd_vel', Twist, queue_size=10)
    global see_image
    see_image=False
    while not rospy.is_shutdown():
        if see_image==False:
            continue
        cv2.imshow("Camara Robot", cv_image)
        altura_imagem, largura_imagem = cv_image.shape[:2]  # altura=480

        # Binary image
        alt_img = canny_alternate(cv_image)
        cv2.imshow("binarized image (test)", alt_img)

        #Biggest area - trial fase
        #alt_img=biggest_area(alt_img)

        # Select one road line
        alt_img = unify_line(alt_img, side="right", average=False)
        cv2.imshow("Test single curve", alt_img)

        # Get a, b and c such as ax2 + bx + c is the best fit to given image
        coeffs: list = get_coeffs(alt_img)


        #Velocity parameters:
        vel_msg = Twist()
        #vel_msg.linear.x = 0.05  # change constant value to change linear velocity
        #vel_msg.linear.y = 0
        #vel_msg.linear.z = 0
        # Angular velocity in the z-axis.
        #vel_msg.angular.x = 0
        #vel_msg.angular.y = 0

        # Debug
        print(f"coeffs: {coeffs}")
        if len(coeffs)==2:
            print("Declive da Reta: " + str(coeffs[1]))
            vel_msg.linear.x = 0.05
            if coeffs[1]>0:
                vel_msg.angular.z = -0.2
            else:
                vel_msg.angular.z=0.2
        elif len(coeffs) == 1:
            print("Reta Horizontal: " + str(coeffs[0]))
            vel_msg.angular.z=0
            vel_msg.linear.x = 0.05
        else:
            print("Grau Superior:" + str(len(coeffs)))

        image_curve = quadratic_image(coeffs=coeffs, width=largura_imagem,
                                      height=altura_imagem)  # de-comment when bin_mask_canny is good

        if image_curve is not None:
            cv2.imshow("Polyfit regression result", image_curve)

        print("Teste1")




        velocity_publisher.publish(vel_msg)
        cv2.waitKey(1)

        rospy.Rate(1).sleep()

if __name__ == '__main__':
    main()
