import cv2
import utils
import numpy as np

# https://learnopencv.com/getting-started-opencv-cuda-module/

# For installing opencv with cuda support:
# https://gist.github.com/raulqf/a3caa97db3f8760af33266a1475d0e5e
# https://danielhavir.github.io/notes/install-opencv/


def processing(image: np.ndarray):

    # Distance of each color level to 0 or 255
    borders_length = 50

    # Process each color channel (r, g or b)
    for i, row in enumerate(image):
        for j, pixel in enumerate(row):
            r, g, b = pixel
            # How much can pixels deviate from black/white color
            min, max = borders_length, 255 - borders_length
            # For discarding colored pixels (road is not colored)
            # if any of r, g or b is outside the spectrum [0, 10] U [245, 255]
            if any([min < color < max for color in (r, g, b)]):
                # Paint black
                image[i][j] = (0, 0, 0)

    return image


def gpu_process(im_path: str):

    # Transfer data from CPU to GPU
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    src = cv2.cuda_GpuMat()
    src.upload(img)  # Error here: no cuda support

    # Image Processing
    clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    dst = clahe.apply(src, cv2.cuda_Stream.Null())
    # Goal would be to process image independently of which device it is (CPU or GPU) ...
    #   ... however it may not be possible
    # processing(image=..., device="GPU")

    # Transfer back to CPU
    result = dst.download()
    cv2.imshow("result", result)
    cv2.waitKey(0)


if __name__ == '__main__':
    gpu_process(
        utils.get_abs_path("lane_test/left_curve.jpg")
    )
