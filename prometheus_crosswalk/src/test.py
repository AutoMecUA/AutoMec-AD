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
    fps_average, fps_max, fps_min = 0, 0, float("inf")

    for frame_count, frame in enumerate(iterable=read_video(video_file_path=VIDEO_FILE_PATH), start=1):

        start_time = time.time()
        sureness = func(frame)
        frames_per_second = 1 / (time.time() - start_time)

        # Calculate stats
        fps_average = (fps_average * (frame_count - 1) + frames_per_second) / frame_count
        fps_min = min(fps_min, frames_per_second)
        fps_max = max(fps_max, frames_per_second)

        print_frame_stats(frames_per_second, sureness)

        cv2.namedWindow(winname="Frame", flags=cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            # Cast stats to int
            fps_min, fps_max, fps_average = int(fps_min), int(fps_max), int(fps_average)

            print(f"fps statistics: min/max/avg: {fps_min}/{fps_max}/{fps_average}")
            exit(0)


def print_frame_stats(frames_per_second, sureness):
    """Displays the statistics regarding a frame's processing

    :param frames_per_second:
    :param sureness:
    :return:
    """

    print(f"Current frame - {int(frames_per_second)} fps: ", end="")
    if sureness > 0.5:
        print(f"IS crosswalk ({int(sureness * 100)}% sure)")
    else:
        print(f"NOT crosswalk (only {int(sureness * 100)}% sure)")


if __name__ == '__main__':
    stateless_test(lib.crosswalk_sureness)
