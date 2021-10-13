#!/usr/bin/env python3

# Imports
import cv2
import numpy as np
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
import pathlib

global img_rbg
global bridge
global begin_img

# not used by now
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255

    return img


def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img

    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    begin_img = True


def main():

    # PARAMETERS__________________________________________________________________

    # Import Parameters
    scale_import = 0.1  # The scale of the first image, related to the imported one.
    N_red = 2  # Number of piramidizations to apply to each image.
    factor_red = 0.8

    # Font Parameters
    subtitle_offset = -10
    subtitle_2_offset = -10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 255)
    font_thickness = 2

    # Line Parameters
    line_thickness = 3

    # Detection Parameters

    scale_cap = 0.4
    detection_threshold = 0.85

    # Initial velocity
    vel = 0

    # ______________________________________________________________________________

    # Images to import and Images Info
    dict_images = {
        'pForward': {'title': 'Follow Straight Ahead', 'type': 'Panel', 'color': 'green', 'images': {}},
        'pStop': {'title': 'Stop', 'type': 'Panel', 'color': 'red', 'images': {}},
        'pLeft': {'title': 'Left', 'type': 'Panel', 'color': 'green', 'images': {}},
        'pRight': {'title': 'Right', 'type': 'Panel', 'color': 'green', 'images': {}},
        'pParking': {'title': 'Parking', 'type': 'Panel', 'color': 'yellow', 'images': {}},
        'pChess': {'title': 'Chess', 'type': 'Panel', 'color': 'red', 'images': {}}
    }

    # Colors dictionary
    dict_colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'yellow': (0, 255, 255)}

    # Global variables
    global img_rbg
    global bridge
    global begin_img
    begin_img = False
    velbool = False


    # Init Node
    rospy.init_node('ml_driving', anonymous=False)

    # Get parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera2/rgb/image_raw')
    signal_cmd_topic = rospy.get_param('~signal_cmd_topic', '/signal_vel')
    
    # Create publishers
    pubbool = rospy.Publisher(signal_cmd_topic, Bool, queue_size=10)

    # ______________________________________________________________________________
    
    path = str(pathlib.Path(__file__).parent.absolute())

    # Images Importation and Resizing
    Counter_Nr_Images = 0
    for name in dict_images.keys():

        # Key and Value for the Zero and Tilt Images
        images_key = '0'
        tilt1_key = 'd'
        tilt2_key = 'l'
        tilt3_key = 'r'
        images_value = cv2.imread(path + '/' + name + '.png', cv2.IMREAD_GRAYSCALE)

        # Determination of required dimensions for the Zero Image
        width = int(images_value.shape[1] * scale_import)
        height = int(images_value.shape[0] * scale_import)
        dim = (width, height)

        # Resizing the Zero Image
        images_value = cv2.resize(images_value, dim)

        # Updating the dictionary with the Key and Value of the Zero Image
        dict_images[name]['images'][images_key] = images_value

        # Define the fraction to transform
        frac = 1 / 8

        # Locate points of the signal which you want to transform
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([[width * frac, 0], [width * (1 - frac), 0], [0, height], [width, height]])
        pts3 = np.float32(
            [[width * frac, 0], [width * (1 - frac), height * frac], [0, height], [width, height * (1 - frac)]])
        pts4 = np.float32(
            [[width * frac, height * frac], [width * (1 - frac), 0], [0, height * (1 - frac)], [width, height]])

        # Transform the original images into a tilted one
        matrix1 = cv2.getPerspectiveTransform(pts1, pts2)
        matrix2 = cv2.getPerspectiveTransform(pts1, pts3)
        matrix3 = cv2.getPerspectiveTransform(pts1, pts4)
        tilt1 = cv2.warpPerspective(images_value, matrix1, dim)
        tilt2 = cv2.warpPerspective(images_value, matrix2, dim)
        tilt3 = cv2.warpPerspective(images_value, matrix3, dim)

        # Updating the dictionary with the Key and Value of the tilted Image
        dict_images[name]['images'][tilt1_key] = tilt1
        dict_images[name]['images'][tilt2_key] = tilt2
        dict_images[name]['images'][tilt3_key] = tilt3

        Counter_Nr_Images += 4

        # Piramidization of the Zero and Tilt Image, creating smaller versions of it
        for n in range(N_red):
            # Defining the keys
            images_key = str(2 * n - 1)
            tilt1_keypyr = tilt1_key + "." + str(2 * n + 1)
            tilt2_keypyr = tilt2_key + "." + str(2 * n + 1)
            tilt3_keypyr = tilt3_key + "." + str(2 * n + 1)
            images_keyh = str(2 * n)
            tilt1_keypyrh = tilt1_key + "." + str(2 * n + 2)
            tilt2_keypyrh = tilt2_key + "." + str(2 * n + 2)
            tilt3_keypyrh = tilt3_key + "." + str(2 * n + 2)

            # Creating another lair of piramidization, assuming dimensions stay the same between signals
            width = int(images_value.shape[1] * 3 / 4)
            height = int(images_value.shape[0] * 3 / 4)
            dim = (width, height)
            images_valueh = cv2.resize(images_value, dim)
            tilt1h = cv2.resize(tilt1, dim)
            tilt2h = cv2.resize(tilt2, dim)
            tilt3h = cv2.resize(tilt3, dim)

            # Pyramidization
            images_value = cv2.pyrDown(images_value)
            tilt1 = cv2.pyrDown(tilt1)
            tilt2 = cv2.pyrDown(tilt2)
            tilt3 = cv2.pyrDown(tilt3)

            # Updating the dictionary with the Key and Value
            dict_images[name]['images'][images_key] = images_value
            dict_images[name]['images'][tilt1_keypyr] = tilt1
            dict_images[name]['images'][tilt2_keypyr] = tilt2
            dict_images[name]['images'][tilt3_keypyr] = tilt3
            dict_images[name]['images'][images_keyh] = images_valueh
            dict_images[name]['images'][tilt1_keypyrh] = tilt1h
            dict_images[name]['images'][tilt2_keypyrh] = tilt2h
            dict_images[name]['images'][tilt3_keypyrh] = tilt3h
            Counter_Nr_Images += 8

    # Number of Images Created
    print("Number of images: " + str(Counter_Nr_Images))

    for name in dict_images.keys():
        for key in dict_images[name]['images']:
            dict_images[name]['images'][key] = cv2.GaussianBlur(dict_images[name]['images'][key], (3, 3), 0)

    # ______________________________________________________________________________

    # Create an object of the CvBridge class
    bridge = CvBridge()


    # Subscribe and publish topics (only after CvBridge)
    rospy.Subscriber(image_raw_topic,
                     Image, message_RGB_ReceivedCallback)


    rate = rospy.Rate(10)

    while not rospy.is_shutdown():

        if begin_img == False:
            continue

        #resized_ = preProcess(img_rbg)

        width_frame = img_rbg.shape[1]
        height_frame = img_rbg.shape[0]
        #default_dim = (width_frame, height_frame)
        reduced_dim = (int(width_frame * scale_cap), int(height_frame * scale_cap))
        frame = cv2.resize(img_rbg, reduced_dim)

        # Converting to a grayscale frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = 0
        loc = 0
        max_res = 0
        max_loc = 0
        max_name = ''
        max_key = ''


        # For each image:
        for name in dict_images.keys():
            for key in dict_images[name]['images']:

                matrix_res = cv2.matchTemplate(gray_frame, dict_images[name]['images'][key], cv2.TM_CCOEFF_NORMED)
                res = np.max(matrix_res)
                loc = np.where(matrix_res == res)

                if res > max_res:
                    max_res = res
                    max_loc = loc

                    max_name = name
                    max_key = key


        if max_res > detection_threshold:

            max_width = int(dict_images[max_name]['images'][max_key].shape[1] / scale_cap)
            max_height = int(dict_images[max_name]['images'][max_key].shape[0] / scale_cap)
            max_dim = (max_width, max_height)

            for pt in zip(*max_loc[::-1]):
                pt = tuple(int(pti / scale_cap) for pti in pt)
                cv2.rectangle(frame, pt, (pt[0] + max_width, pt[1] + max_height),
                            dict_colors.get(dict_images[max_name]['color']), line_thickness)
                text = 'Detected: ' + max_name + ' ' + max_key + ' > ' + dict_images[max_name]['type'] + ': ' + \
                    dict_images[max_name]['title']
                
                #print(text)

                origin = (pt[0], pt[1] + subtitle_offset)
                origin_2 = (0, height_frame + subtitle_2_offset)
                # Using cv2.putText() method
                subtitle = cv2.putText(img_rbg, str(max_name) + '_' + str(max_key) + ' ' + str(round(max_res, 2)), origin,
                                    font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                subtitle_2 = cv2.putText(img_rbg, text, origin_2, font, font_scale, font_color, font_thickness,
                                        cv2.LINE_AA)

            # Defining and publishing the velocity of the car in regards to the signal seen
            if max_name == "pForward":
                velbool = True
            elif max_name == "pStop":
                velbool = False

        pubbool.publish(velbool)

        # Show image
        cv2.imshow("Frame", img_rbg)
        key = cv2.waitKey(1)

        rate.sleep()

if __name__ == '__main__':
    main()