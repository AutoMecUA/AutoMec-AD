"""Tests out stateless or stateful (soon) algorithms

"""

import logging
import random
import time
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL.ImageSequence import Iterator

import stateless_lib

WHITE = (255, 255, 255)

# Toggle debug mode - features random frame skip
TOGGLE_GRAPH: bool = True  # Show a graph at the end of the run
DEBUG_MODE: bool = False
FRAME_SKIP_RATE = 0.99
VIDEO_FILE_PATH = "../crosswalk_cut.mp4"

KILO = 1024
MEGA = 2**20


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


def show_line_graph(data: np.ndarray, title: str = ""):
    plt.plot(data)

    plt.title(title)

    plt.show()


def cv2_put_text(img: np.ndarray, text: str) -> np.ndarray:
    """Provides a simple version of the cv2.putText function

    :param img:
    :param text:
    :return:
    """

    # Define defaults
    coordinates = (100, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.60
    color = WHITE
    thickness = 2
    linetype = cv2.LINE_AA
    # TODO move the above variables to function parameters (with defaults, of course)

    return cv2.putText(img, text, coordinates, font, font_scale, color, thickness, linetype)


def stateless_test(func: callable):
    """Test crosswalk

    :param func:
    :return:
    """

    fps_average, fps_max, fps_min = 0, 0, float("inf")

    white_count_tracker = np.zeros((10 * KILO,))

    for frame_count, frame in enumerate(iterable=read_video(video_file_path=VIDEO_FILE_PATH), start=1):
        frame: np.ndarray  # just a type hint

        start_time = time.time()
        sureness = func(frame)
        frames_per_second: float = 1 / (time.time() - start_time)

        # Add the number of white pixels to the counting list
        threshold__sum = (frame > stateless_lib.BINARY_THRESHOLD).sum()
        white_count_tracker[frame_count] = threshold__sum / frame.size

        # Calculate fps statistics
        fps_average = (fps_average * (frame_count - 1) + frames_per_second) / frame_count
        fps_min = min(fps_min, frames_per_second)
        fps_max = max(fps_max, frames_per_second)

        cv2.namedWindow(winname="Frame", flags=cv2.WINDOW_KEEPRATIO)
        frame = cv2_put_text(frame, text=frame_stats(frames_per_second, sureness))

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    # Cast stats to int
    fps_min, fps_max, fps_average = int(fps_min), int(fps_max), int(fps_average)

    print(f"fps statistics: min/max/avg: {fps_min}/{fps_max}/{fps_average}")

    if TOGGLE_GRAPH:
        show_line_graph(data=np.trim_zeros(white_count_tracker), title="Probability of bright pixels in each frame")
    exit(0)


def frame_stats(frames_per_second, sureness, do_print: bool = False) -> Union[str, None]:
    """Displays the statistics regarding a frame's processing

    :param do_print:
    :param frames_per_second:
    :param sureness:
    :return:
    """

    frames_per_second = str(int(frames_per_second)).zfill(4)
    sureness_text = str(int(sureness * 100)).zfill(3)

    res = f"Current frame - {frames_per_second} fps: "
    if sureness > 0.5:
        res += f" IS crosswalk ({sureness_text}% sure)"
    else:
        res += f"NOT crosswalk ({sureness_text}% sure)"

    if do_print:
        print(res)
    return res

def main():
    stateless_test(stateless_lib.basic_sureness)


if __name__ == '__main__':
    main()
