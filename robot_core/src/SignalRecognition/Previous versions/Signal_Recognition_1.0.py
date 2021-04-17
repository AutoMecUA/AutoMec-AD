import time
import cv2
import numpy as np

# PARAMETERS__________________________________________________________________

# Import Parameters
scale_import = 0.4 # The scale of the first image, related to the imported one.
N_pir = 3          # Number of piramidizations to apply to each image.

# Font Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 0, 0)
font_thickness = 1

# Line Parameters
line_thickness = 3

#
scale_cap = 0.4
detection_threshold = 0.75

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
    width = int(images_value.shape[1] * (scale_import))
    height = int(images_value.shape[0] * (scale_import))
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
cap = cv2.VideoCapture(0)

while True:
    # Timer Starts
    tic()

    # Reading and resizing one frame
    _, frame = cap.read()
    width_frame = frame.shape[1]
    height_frame = frame.shape[0]
    default_dim = (width_frame, height_frame)
    reduced_dim = (int(width_frame * scale_cap), int(height_frame * scale_cap))
    frame = cv2.resize(frame, reduced_dim)

    # Converting to a grayscale frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = 0
    max_res = 0
    max_loc = 0
    max_name = ''
    max_key = ''

    # For each image:
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

    max_width = int(dict_images[max_name]['images'][max_key].shape[1])
    max_height = int(dict_images[max_name]['images'][max_key].shape[0])
    max_dim = (max_width, max_height)


    if max_res > detection_threshold:

        for pt in zip(*max_loc[::-1]):

            cv2.rectangle(frame, pt, (pt[0] + max_width, pt[1] + max_height), dict_colors.get(dict_images[max_name]['color']), line_thickness)
            print('Detected: ' + max_name + ' ' + max_key + ' -> ' + dict_images[max_name]['type'] + ': ' + dict_images[max_name]['title'])

            origin = (pt[0], pt[1])
            # Using cv2.putText() method
            image = cv2.putText(frame, str(max_res) + ' ' + str(max_name) + ' ' + str(max_key), origin, font, font_scale, font_color, font_thickness, cv2.LINE_AA)


    # Show the frame
    frame = cv2.resize(frame, default_dim)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:           # Press "ESC"
        break               # End While cycle
    toc()

cap.release()               # Stops Video Capture
cv2.destroyAllWindows()     # Closes all windows
