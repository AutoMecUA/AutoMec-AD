import numpy as np

from util import _blurred_brightness_rate

# Tune these constants - Base the tuning upon the characteristics of an image with vs without crosswalk
BINARY_THRESHOLD = 128
MIN_WHITENESS = 0.05
MAX_WHITENESS = 0.15


def basic_sureness(frame: np.ndarray) -> float:
    """Calculates the probability based solely on the (normalized) count of bright pixels

    Probability, that is, that there is a crosswalk right in the front of the car whose camera recorded the frame

    :param frame: Image of the camera of the robot
    :return: How sure the function is that there is a crosswalk in front of the car based on the frame image
    """

    actual_whiteness = _blurred_brightness_rate(frame, threshold=BINARY_THRESHOLD)

    # Threshold for maximum whiteness - an image whill never be 100% white
    actual_whiteness = min(actual_whiteness, MAX_WHITENESS)
    # Threshold for minimum whiteness - an image whill never be 100% black
    actual_whiteness = max(actual_whiteness, MIN_WHITENESS)

    # Return a normalized value: ranging from [0,1] instead of [MIN_WH, MAX_WH]
    normalized = (actual_whiteness - MIN_WHITENESS) / (MAX_WHITENESS - MIN_WHITENESS)

    assert 0.0 <= normalized <= 1.0, f"Invalid sureness value '{normalized}'!"

    return normalized
