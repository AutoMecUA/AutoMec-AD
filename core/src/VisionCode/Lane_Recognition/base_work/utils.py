import os
import cv2

# References
# https://techtutorialsx.com/2020/12/13/python-opencv-add-slider-to-window/


def get_abs_path(rel_path):
    return os.path.realpath(rel_path)


def on_change(value):
    print(value)


def show_image(img_path: str = str(), image=None, title="Display window"):
    """

    :param title:
    :param image:
    :param img_path: Image's absolute path: /.../image.png
    :return: Window showing the requested image. Press any key to close it
    """

    if image is not None:
        img = image
    elif img_path != str():
        img = cv2.imread(filename=img_path)
    else:
        raise AssertionError("No image provided!")

    # Create a visualization window
    # CV_WINDOW_AUTOSIZE : window size will depend on image size
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    # Show the image
    cv2.imshow(title, img)

    # Wait
    cv2.waitKey(0)

    # Destroy the window -- might be omitted
    cv2.destroyWindow(title)
