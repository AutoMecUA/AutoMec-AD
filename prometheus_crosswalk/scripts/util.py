import cv2
import numpy as np

def _blurred_brightness_rate(img: np.ndarray, threshold: int) -> float:
    """Counts the rate of the pixels that surpass the threshold provided

    :param img: Image
    :param threshold: integer value to be compared with the pixel values
    :return:
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _,h = img.shape
    img = img[int(h/2):int(3*h/4),:]

    return((img > threshold).sum() / img.size)