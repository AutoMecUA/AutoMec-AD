import time
import cv2
import numpy as np

# PARAMETERS__________________________________________________________________

# Import Parameters
scale_import = 0.4 # The scale of the first image, related to the imported one.
N_pir = 3          # Number of piramidizations to apply to each image.

# Font Parameters
subtitle_offset = -10
subtitle_2_offset = -10
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 0, 255)
font_thickness = 2

# Line Parameters
line_thickness = 3

#
scale_cap = 0.4
detection_threshold = 0.8

# Tracking Parameters
tracking_activation_threshold = 10
tracking_threshold = 0.95
tracking_sizes = [0.9, 1]

#______________________________________________________________________________

# Images to import and Images Info
dict_images = {
    'pRight': {'title': 'Follow to the Right', 'type': 'Panel', 'color': 'yellow', 'images': {}},
    'pLeft': {'title': 'Follow to the Left', 'type': 'Panel', 'color': 'yellow', 'images': {}},
    'pEnd': {'title': 'End of trial', 'type': 'Panel', 'color': 'yellow', 'images': {}},
    'pPark': {'title': 'Follow to parking area', 'type': 'Panel', 'color': 'yellow', 'images': {}},
    'pStop': {'title': 'Stop', 'type': 'Panel', 'color': 'yellow', 'images': {}},
    's01': {'title': 'Other Danger', 'type': 'WARNING Sign', 'color': 'red', 'images': {}},
    's11': {'title': 'Depression', 'type': 'WARNING Sign', 'color': 'red', 'images': {}},
    's21': {'title': 'Animals', 'type': 'WARNING Sign', 'color': 'red', 'images': {}},
    's31': {'title': 'Narrow Road', 'type': 'WARNING Sign', 'color': 'red', 'images': {}},
    's02': {'title': 'Parking Lot', 'type': 'INFORMATION Sign', 'color': 'green', 'images': {}},
    's12': {'title': 'Maximum Recommended Speed: 60', 'type': 'INFORMATION Sign', 'color': 'green', 'images': {}},
    's22': {'title': 'Hospital', 'type': 'INFORMATION Sign', 'color': 'green', 'images': {}},
    's32': {'title': 'Zebra Crossing', 'type': 'INFORMATION Sign', 'color': 'green', 'images': {}},
    's03': {'title': 'Follow to the Left', 'type': 'MANDATORY Sign', 'color': 'blue', 'images': {}},
    's13': {'title': 'Turn the Headlights On', 'type': 'MANDATORY Sign', 'color': 'blue', 'images': {}},
    's23': {'title': 'Roundabout', 'type': 'MANDATORY Sign', 'color': 'blue', 'images': {}},
    's33': {'title': 'Bus Lane', 'type': 'MANDATORY Sign', 'color': 'blue', 'images': {}}
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

    # Show the Zero Image
    cv2.imshow(name + ' ' + images_key, dict_images[name]['images'][images_key])

    Counter_Nr_Images += 1

    # Piramidization of the Zero Image, creating smaller versions of it
    for n in range(N_pir):
        images_key = str(n+1)
        images_value = cv2.pyrDown(images_value)

        # Updating the dictionary with the Key and Value
        dict_images[name]['images'][images_key] = images_value

        # Show the Image
        cv2.imshow(name + ' ' + images_key, dict_images[name]['images'][images_key])

        Counter_Nr_Images += 1

# Number of Images Created
print("Number of images: " + str(Counter_Nr_Images))

# TIMING FUNCTIONS CREATION
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
     print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
     print("Toc: start time not set")

def tocs():
    if 'startTime_for_tictoc' in globals():
        return str(time.time() - startTime_for_tictoc)
    else:
        print("Toc: start time not set")
    return None

# VIDEO CAPTURE AND PROCESSING

# Start of video capture

tracking_activation = 0
tracking_mode = 0
p_max_name = '_'
res = 0
loc = 0

max_loc = 0
max_name = ''
max_key = ''

cap = cv2.VideoCapture(0)

while True:
    # Timer Starts
    tic()

    max_res = 0

    # Reading and resizing one frame
    _, def_frame = cap.read()
    width_frame = def_frame.shape[1]
    height_frame = def_frame.shape[0]
    default_dim = (width_frame, height_frame)
    reduced_dim = (int(width_frame * scale_cap), int(height_frame * scale_cap))
    frame = cv2.resize(def_frame, reduced_dim)

    # Converting to a grayscale frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # For each image:
    if tracking_mode == 0:
        for name in dict_images.keys():
            for key in dict_images[name]['images']:
                if key != '0':
                    matrix_res = cv2.matchTemplate(gray_frame, dict_images[name]['images'][key], cv2.TM_CCOEFF_NORMED)
                    res = np.max(matrix_res)
                    loc = np.where(matrix_res == res)

                if res > max_res:
                    max_res = res
                    max_loc = loc

                    max_name = name
                    max_key = key

        max_width = int(dict_images[max_name]['images'][max_key].shape[1]/scale_cap)
        max_height = int(dict_images[max_name]['images'][max_key].shape[0]/scale_cap)
        max_dim = (max_width, max_height)

    elif tracking_mode == 1:
        for i in range(len(tracking_sizes)):
            width = int(tracking_img.shape[1] * tracking_sizes[i])
            height = int(tracking_img.shape[0] * tracking_sizes[i])
            dim = (width, height)

            matrix_res_tracking = cv2.matchTemplate(gray_frame, cv2.resize(tracking_img, dim), cv2.TM_CCOEFF_NORMED)
            res = np.max(matrix_res_tracking)
            loc = np.where(matrix_res_tracking == res)

            if res > max_res:
                max_res = np.max(matrix_res_tracking)
                max_loc = np.where(matrix_res_tracking == max_res)
                tracking_idx = i

        max_width = int(tracking_img.shape[1] * tracking_sizes[i] / scale_cap)
        max_height = int(tracking_img.shape[0] * tracking_sizes[i] / scale_cap)
        max_dim = (max_width, max_height)

    if ((max_res > detection_threshold)&(tracking_mode == 0))|((max_res > tracking_threshold)&(tracking_mode == 1)):
        for pt in zip(*max_loc[::-1]):

            pt = tuple(int(pti/scale_cap) for pti in pt)
            cv2.rectangle(def_frame, pt, (pt[0] + max_width, pt[1] + max_height), dict_colors.get(dict_images[max_name]['color']), line_thickness)
            text = 'Detected: ' + max_name + '_' + max_key + ' > ' + dict_images[max_name]['type'] + ': ' + dict_images[max_name]['title']
            print(text)

            origin = (pt[0], pt[1] + subtitle_offset)
            origin_2 = (0, height_frame + subtitle_2_offset)

            # Using cv2.putText() method
            subtitle = cv2.putText(def_frame, str(max_name) + '_' + str(max_key) + ' ' + str(round(max_res, 2)), origin, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            subtitle_2 = cv2.putText(def_frame, text, origin_2, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            if tracking_mode == 1:
                origin_3 = (0, height_frame + 4 * subtitle_2_offset)
                subtitle_3 = cv2.putText(def_frame, 'TRACKING', origin_3, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            tracking_img = gray_frame[int(pt[1]*scale_cap):int((pt[1] + max_height)*scale_cap), int(pt[0]*scale_cap):int((pt[0] + max_width)*scale_cap)]
            cv2.imshow("tracking_img", tracking_img)

        if max_name == p_max_name:
            tracking_activation += 1
        else:
            tracking_activation = 0

        p_max_name = max_name

        if tracking_activation == tracking_activation_threshold:
            tracking_activation = 0
            tracking_mode = 1

    elif tracking_mode == 1:
        tracking_mode = 0

    # Show the frame
    '''frame = cv2.resize(frame, default_dim)'''
    cv2.imshow("Working Frame", frame)
    cv2.imshow("Frame", def_frame)
    print(str(tracking_mode))

    key = cv2.waitKey(1)

    if key == 27:           # Press "ESC"
        break               # End While cycle
    toc()

cap.release()               # Stops Video Capture
cv2.destroyAllWindows()     # Closes all windows
