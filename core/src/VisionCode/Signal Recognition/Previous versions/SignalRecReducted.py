
import cv2
import numpy as np

# PARAMETERS__________________________________________________________________

# Import Parameters
scale_import = 0.4  # The scale of the first image, related to the imported one.
N_red = 3  # Number of piramidizations to apply to each image.
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
detection_threshold = 0.7

# ______________________________________________________________________________

# Images to import and Images Info
dict_images = {
    'pForward': {'title': 'Follow Straight Ahead', 'type': 'Panel', 'color': 'green', 'images': {}},
    'pStop': {'title': 'Stop', 'type': 'Panel', 'color': 'red', 'images': {}}
}

# Colors dictionary
dict_colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'yellow': (0, 255, 255)}

# Images Importation and Resizing
Counter_Nr_Images = 0
for name in dict_images.keys():

    # Key and Value for the Zero Image
    images_key = '0'
    images_value = cv2.imread(name + '.png', cv2.IMREAD_GRAYSCALE)

    # Determination of required dimensions for the Zero Image
    width = int(images_value.shape[1] * scale_import)
    height = int(images_value.shape[0] * scale_import)
    dim = (width, height)

    # Resizing the Zero Image
    images_value = cv2.resize(images_value, dim)

    # Updating the dictionary with the Key and Value of the Zero Image
    dict_images[name]['images'][images_key] = images_value

    Counter_Nr_Images += 1

    # Piramidization of the Zero Image, creating smaller versions of it
    for n in range(N_red):
        images_key = str(n + 1)
        images_value = cv2.pyrDown(images_value)

        # Updating the dictionary with the Key and Value
        dict_images[name]['images'][images_key] = images_value

        Counter_Nr_Images += 1


# Number of Images Created
print("Number of images: " + str(Counter_Nr_Images))

for name in dict_images.keys():
    for key in dict_images[name]['images']:
        dict_images[name]['images'][key] = cv2.GaussianBlur(dict_images[name]['images'][key], (3, 3), 0)
        cv2.imshow(name + ' ' + key, dict_images[name]['images'][key])


# VIDEO CAPTURE AND PROCESSING

# Start of video capture
cap = cv2.VideoCapture(0)

while True:

    # Reading and resizing one frame
    _, def_frame = cap.read()
    width_frame = def_frame.shape[1]
    height_frame = def_frame.shape[0]
    default_dim = (width_frame, height_frame)
    reduced_dim = (int(width_frame * scale_cap), int(height_frame * scale_cap))
    frame = cv2.resize(def_frame, reduced_dim)

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

    max_width = int(dict_images[max_name]['images'][max_key].shape[1] / scale_cap)
    max_height = int(dict_images[max_name]['images'][max_key].shape[0] / scale_cap)
    max_dim = (max_width, max_height)

    if max_res > detection_threshold:

        for pt in zip(*max_loc[::-1]):
            pt = tuple(int(pti / scale_cap) for pti in pt)
            cv2.rectangle(def_frame, pt, (pt[0] + max_width, pt[1] + max_height),
                          dict_colors.get(dict_images[max_name]['color']), line_thickness)
            text = 'Detected: ' + max_name + ' ' + max_key + ' > ' + dict_images[max_name]['type'] + ': ' + \
                   dict_images[max_name]['title']
            print(text)

            origin = (pt[0], pt[1] + subtitle_offset)
            origin_2 = (0, height_frame + subtitle_2_offset)
            # Using cv2.putText() method
            subtitle = cv2.putText(def_frame, str(max_name) + '_' + str(max_key) + ' ' + str(round(max_res, 2)), origin,
                                   font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            subtitle_2 = cv2.putText(def_frame, text, origin_2, font, font_scale, font_color, font_thickness,
                                     cv2.LINE_AA)

    # Show the frame
    '''frame = cv2.resize(frame, default_dim)'''
    cv2.imshow("Working Frame", frame)
    cv2.imshow("Frame", def_frame)

    key = cv2.waitKey(1)

    if key == 27:  # Press "ESC"
        break  # End While cycle

cap.release()  # Stops Video Capture
cv2.destroyAllWindows()  # Closes all windows
