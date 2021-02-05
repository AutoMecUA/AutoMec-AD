import cv2
import numpy as np
import utils


# References
# https://www.learnpython.org/en/Multiple_Function_Arguments
# https://medium.com/@yogeshojha/self-driving-cars-beginners-guide-to-computer-vision-finding-simple-lane-lines-using-python-a4977015e232


def masking(image, polygon):
    """

    :param image: Must be greyscaled
    :param polygon: list of points
    :return: masked image
    """
    # Masking region of interest
    height, width = image.shape[0:2]
    for i, point in enumerate(polygon):
        # 1 - y because image matrix is processed from top to down
        polygon[i] = point[0] * (width - 1), (1 - point[1]) * (height - 1)

    polygon = np.array(polygon, np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygon], 255)
    cropped_image = cv2.bitwise_and(image, mask)

    return cropped_image


def hough_p(original_img, cropped_img):
    # Hough's transform
    rho = 2
    theta = np.pi / 180
    threshold = 100  # Minimum votes to be recognized as a line
    lines = cv2.HoughLinesP(cropped_img, rho, theta, threshold, np.array([]), minLineLength=40, maxLineGap=5)

    # Mid code:
    slopes = []

    line_image = np.zeros_like(original_img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            if x2 != x1:
                slopes.append((y2-y1) / (x2-x1))
            else:
                # Big number
                slopes.append(100_000)
    else:
        raise AssertionError("lines == None")

    av_slope = np.average(slopes)

    return cv2.addWeighted(original_img, 0.8, line_image, 1, 1), av_slope


def lane_detector(img_path, **parameters) -> np.ndarray:
    """
    Status: working, but needs better masking
    :return: Image with lines drawn over lane separators

    :keyword Arguments:
        * *blur_level* (``int``) --
            level of blurring:
                Depends on blur_funct followingly described:
                    - if it's average blur function, any int works
                    - if it's median blur function, any odd int above 1 works
                    - else if it's gaussian blur function, any positive odd int is acceptable
        * *blur_funct* (``funct``) --
            choose from available opencv blur functions
        * *thresholds* (``tuple``) --
            format (threshold1, threshold2)
            arguments explained in cv2::Canny
        * *polygon* (``array``)
            format [point1, point2, point3]
            Since in most cases the roads shape like a triangle,
                our mask will be this triangle,
                defined by fraction values.
                E.g.: (.5, .5) means the point at the center of the image
        ...
    """
    # ---- Parameters: get in kwargs if it's there, if not set as default ----

    # Blurring
    if "blur_level" not in parameters.keys() or \
            parameters['blur_level'] not in [3, 5, 7]:
        blur_level = 5
    else:
        blur_level = parameters['blur_level']
    blur = (blur_level, blur_level)

    if "blur_funct" not in parameters.keys():
        blur_funct = cv2.GaussianBlur
    else:
        blur_funct = parameters['blur_funct']

    # Canny edge detection
    thresholds = (50, 150)  # Default value
    if "thresholds" in parameters:
        assert all([0 <= elem < 256 for elem in parameters['thresholds']]),\
            "Invalid threshold values!"
        thresholds = parameters['thresholds']

    # Masking
    polygon = [(0, 0), (0, .4), (.5, .78), (.99, .4), (.99, 0)]
    if "polygon" in parameters:
        assert all([len(tupl) == 2 for tupl in polygon]), "Wrong polygon data struct"
        polygon = parameters["polygon"]

    # ---- Pipeline stages of image processing ----
    # Loading the image
    lane_image = cv2.imread(utils.get_abs_path(img_path))

    # Converting into grayscale
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    # Reduce Noise and Smoothen Image
    if blur_funct == cv2.GaussianBlur:
        blur = blur_funct(
            src=gray, ksize=blur, sigmaX=0
        )
    else:
        blur = blur_funct(
            src=gray, ksize=blur[0], sigmaX=0
        )

    # Edge detection (canny)
    canny_image = cv2.Canny(
        image=blur, threshold1=thresholds[0], threshold2=thresholds[1]
    )

    # Mask region of interest
    cropped_image = masking(canny_image, polygon)

    # Hough Transform
    combo_image, slope = hough_p(original_img=lane_image, cropped_img=cropped_image)

    # Direction values
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)
    print(f"Slope: {slope}", f"Angle (rad): {round(np.pi/2 - angle_rad, 2)}*pi",
          f"Angle (degrees) {round(90.0 - float(angle_deg), 2)}º", sep="\n", end=".")
    utils.show_image(image=combo_image, title=img_path.split('/')[-1])

    return combo_image


if __name__ == '__main__':

    for i in range(1, 16):
        im_path = f"../../images/img{i}.jpeg"
        print(f"\nImage {i}:")
        try:
            lane_detector(img_path=im_path,
                          blur_level=7,
                          blur_funct=cv2.GaussianBlur,
                          thresholds=(50, 150),
                          polygon=[(0, .1), (.5, .55), (1.0, .1)]
                          )
        except AssertionError:
            print(f"No lines found in images/img{i}.jpeg")
