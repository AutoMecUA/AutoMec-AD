import cv2
import numpy as np
from PIL.ImageSequence import Iterator

# TODO tune these constants - Base the tuning upon the characteristics of an image with vs without crosswalk
BINARY_THRESHOLD = 128
MIN_WHITENESS = 0.2
MAX_WHITENESS = 0.7


def crosswalk_sureness(frame: np.ndarray) -> float:
    """Detects the probability of a crosswalk being in front of a car

    :param frame: Image of the camera of the robot
    :return: How sure the function is that there is a crosswalk in front of the car based on the frame image
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, frame = cv2.threshold(frame, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # We don't need the image contours nor any other small features
    frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

    actual_whiteness = frame.sum() / (frame.size * 255)

    # Threshold for maximum whiteness - because an image whill never be 100% white
    actual_whiteness = min(actual_whiteness, MAX_WHITENESS)
    # Threshold for minimum whiteness - because an image whill never be 100% black
    actual_whiteness = max(actual_whiteness, MIN_WHITENESS)

    return actual_whiteness


def read_video(video_file_path: str) -> Iterator:
    """

    :param video_file_path: Path to the video file
    :return: Frames of the image
    """

    video = cv2.VideoCapture(video_file_path)

    while video.grab():
        retval, frame = video.read()
        assert retval, "Error code reading video frames!"
        yield frame

    video.release()


def main():
    for frame in read_video(video_file_path="../crosswalk.mkv"):
        sureness = crosswalk_sureness(frame)
        print(f" {sureness}")


if __name__ == '__main__':
    main()
