import logging
import random
import time

import cv2
from PIL.ImageSequence import Iterator

try:
    from prometheus_crosswalk.src import lib
except ImportError:
    import lib

# Toggle debug mode - features random frame skip
DEBUG_MODE: bool = False
FRAME_SKIP_RATE = 0.99
VIDEO_FILE_PATH = "../crosswalk_cut.mp4"


def read_video(video_file_path: str) -> Iterator:
    """

    :param video_file_path: Path to the video file
    :return: Frames of the image
    """

    video = cv2.VideoCapture(video_file_path)

    while video.grab():
        retval, frame = video.retrieve()

        # Skip frames: good for debugging
        if DEBUG_MODE and random.random() < FRAME_SKIP_RATE:
            logging.log(level=logging.DEBUG, msg="Random skipping...")
            continue

        if not retval:
            logging.log(level=logging.WARNING, msg="Error reading a video frame!")
            continue
        yield frame

    video.release()


def stateless_test(func: callable):
    for frame in read_video(video_file_path=VIDEO_FILE_PATH):
        start_time = time.time()
        sureness = func(frame)
        frames_per_second = 1 / (time.time() - start_time)

        print(f"Current frame - {int(frames_per_second)} fps: ", end="")
        if sureness > 0.5:
            print(f"IS crosswalk ({int(sureness*100)}% sure)")
        else:
            print(f"NOT crosswalk (only {int(sureness * 100)}% sure)")

        cv2.namedWindow(winname="Frame", flags=cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    stateless_test(lib.crosswalk_sureness)
