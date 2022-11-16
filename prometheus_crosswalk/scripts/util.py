import cv2
import numpy as np


def _blurred_brightness_rate(img: np.ndarray, threshold: int) -> float:
    """Counts the rate of the pixels that surpass the threshold provided

    :param img: Image
    :param threshold: integer value to be compared with the pixel values
    :return:
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image. Is this any use? TODO Delete if not
    img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

    return (img > threshold).sum() / img.size
