import cv2
import numpy as np

# Tune these constants - Base the tuning upon the characteristics of an image with vs without crosswalk
BINARY_THRESHOLD = 128
MIN_WHITENESS = 0.05
MAX_WHITENESS = 0.15


def crosswalk_sureness(frame: np.ndarray) -> float:
    """Detects the probability of a crosswalk being in front of a car

    :param frame: Image of the camera of the robot
    :return: How sure the function is that there is a crosswalk in front of the car based on the frame image
    """

    actual_whiteness = _image_brightness_rate(frame, threshold=BINARY_THRESHOLD)

    # Threshold for maximum whiteness - because an image whill never be 100% white
    actual_whiteness = min(actual_whiteness, MAX_WHITENESS)
    # Threshold for minimum whiteness - because an image whill never be 100% black
    actual_whiteness = max(actual_whiteness, MIN_WHITENESS)

    # Return a normalized value: ranging from [0,1] instead of [MIN_WH, MAX_WH]
    normalized = (actual_whiteness - MIN_WHITENESS) / (MAX_WHITENESS - MIN_WHITENESS)

    assert 0.0 <= normalized <= 1.0, f"Invalid sureness value '{normalized}'!"

    return normalized


def _image_brightness_rate(img: np.ndarray, threshold: int) -> float:
    """Counts the rate of the pixels that surpass the threshold provided

    :param img: Image
    :param threshold: integer value to be compared with the pixel values
    :return:
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image
    img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

    return (img > threshold).sum() / img.size
