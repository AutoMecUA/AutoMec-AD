# File for useful functions in the context of lane recognition

import cv2 as cv
import os


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


if __name__ == '__main__':
    # Testing, maybe...
    ...
