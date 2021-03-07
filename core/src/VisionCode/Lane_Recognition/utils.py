# File for useful functions in the context of lane recognition

import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def get_gray_image_histogram(image):
    """
    As shown in https://docs.opencv.org/master/d8/dbc/tutorial_histogram_calculation.html
    :param image:
    :return: shows histogram of image (mapping of colors and their frequencies)
    """

    hist_size = 256
    hist = cv.calcHist([image], [0], None, [256], [0, 255])

    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / hist_size))
    hist_image = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    for i in range(1, hist_size):
        cv.line(hist_image, (bin_w * (i - 1), hist_h - int(hist[i - 1])),
                (bin_w * (i), hist_h - int(hist[i])),
                (255, 0, 0), thickness=2)

    cv.imshow('calcHist Demo', hist_image)
    cv.waitKey(1)


def show_image(img_path):
    """

    :param img_path: Image's absolute path: /.../image.png
    :return: Window showing the requested image. Press any key to close it
    """

    img = cv.imread(filename=img_path)

    show_image_temp(img=img)


def show_image_temp(img):
    """

    :param img:
    :return:
    """
    # Create a visualization window
    # CV_WINDOW_AUTOSIZE : window size will depend on image size
    cv.namedWindow("Display window", cv.WINDOW_AUTOSIZE)

    # Show the image
    cv.imshow("Display window", img)

    # Wait
    cv.waitKey(0)

    # Destroy the window -- might be omitted
    cv.destroyWindow("Display window")


def get_abs_path(rel_path: str, is_dir: bool = False):
    """

    :param rel_path: relative path
    :param is_dir:
    :return:
    """

    path = os.path.realpath(rel_path)

    assert os.path.exists(path), "File does not exist in specified path!"
    assert not is_dir or os.path.isdir(path), "Expected directory, got file"

    return path


def draw_quadratic(a: float, b: float, c: float,
                   title: str = None, legend: str = None):

    width, height = 640, 480

    x = np.arange(0, width, 1)  # [0, 1, 2, ..., width - 1]
    y = a * x**2 + b * x + c  # quadratic

    plt.ylim(0, height)  # ymin and ymax
    plt.plot(x, y, 'b')  # blue line

    # Describing the graph
    if title is not None:
        plt.title(title)
    if legend is not None:
        plt.legend(legend)

    plt.show()


if __name__ == '__main__':
    # For individual testing of the modules
    # draw_quadratic(-0.001, 1, 200,
    #                title="quadratic", legend="legend")
    gray_image = cv.cvtColor(
        cv.imread(get_abs_path(rel_path="../images/img1.jpeg")),
        cv.COLOR_RGB2GRAY
    )
    get_gray_image_histogram(gray_image)
    cv.waitKey(0)
