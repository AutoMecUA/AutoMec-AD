import numpy as np
import cv2
index_camera=0
cap = cv2.VideoCapture(index_camera)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
