# https://www.youtube.com/watch?v=VWUgkcX_KoY&list=PL6Yc5OUgcoTlvHb5OfFLUJ90ofBuoU5g8&index=1

import cv2
import numpy as np

# dlib is a machine learning library mostly used for computer vision
# It supports up to 68 points on your face (check image for reference)
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Returns the midpoint of two points
def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:

    _, frame = cap.read()

    # Grayscale image to save computational power
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # This holds the rectangle points for where the face is in the frame, top left to bottom right
    # [[x1, y1], [x2, y2]]
    faces = detector(gray)

    for face in faces:

        landmarks = predictor(gray, face)

        # These are the horizontal points on the left eye
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)

        center_top_point = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom_point = midpoint(landmarks.part(41), landmarks.part(40))


    cv2.imshow("Eye Capture", frame)

    # If the "s" key is pressed, stop
    key = cv2.waitKey(1)
    if key == ord("s"):
        break


cap.release()
cv2.destroyAllWindows()