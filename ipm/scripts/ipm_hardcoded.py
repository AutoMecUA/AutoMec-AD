import os

import cv2

from lib import ipm_class_ros

if __name__ == "__main__":
    height, width = 480, 680
    K = [[563.62112445, 0., 340.5], [0., 563.62112445, 240.5], [0., 0., 1.]]
    pose = {'X': 0.0, 'Y': 0.0, 'Z': 0.547, 'r': 0.0, 'p': 0.6, 'y': 0.0}
    img_rgb = cv2.imread(f"{os.path.dirname(os.path.abspath(__file__))}/images/saved_image.jpg")
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    ipm_instance = ipm_class_ros.IPM(height=height, width=width, K=K, pose=pose)

    output_image = ipm_instance.calculate_output_image(gray)

