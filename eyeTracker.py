# https://www.youtube.com/watch?v=VWUgkcX_KoY&list=PL6Yc5OUgcoTlvHb5OfFLUJ90ofBuoU5g8&index=1

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    cv2.imshow("Eye Capture", frame)

    # If the "s" key is pressed, stop
    key = cv2.waitKey(1)
    if key == ord("s"):
        break


cap.release()
cv2.destroyAllWindows()