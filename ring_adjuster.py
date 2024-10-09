import cv2 as cv
import numpy as np
import requests
import json
import time

# Initialize VideoCapture and GUI window
cap = cv.VideoCapture("http://192.168.1.10:8000/video_feed")
cv.namedWindow("Frame")

# Initialize parameters for center and ring sizes
center_x = 251
center_y = 287
ring_10 = 12
ring_9 = 36
ring_8 = 60
ring_7 = 84
ring_6 = 108
ring_5 = 132
ring_4 = 156
ring_3 = 180
ring_2 = 204
ring_1 = 228

# Function to draw rings on the canvas
def drawRings(canvas):
    cv.circle(canvas, (center_x, center_y), 1, (0, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_10, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_9, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_8, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_7, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_6, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_5, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_4, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_3, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_2, (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), ring_1, (255, 0, 255), 2)
    return canvas

# Empty function for trackbar event
def nothing(_):
    pass

# Create trackbars for dynamic adjustments
cv.createTrackbar('Center X', 'Frame', 251, 400, nothing)
cv.createTrackbar('Center Y', 'Frame', 287, 400, nothing)
cv.createTrackbar('Ring 10', 'Frame', 12, 400, nothing)
cv.createTrackbar('Ring 9', 'Frame', 36, 300, nothing)
cv.createTrackbar('Ring 8', 'Frame', 60, 300, nothing)
cv.createTrackbar('Ring 7', 'Frame', 84, 300, nothing)
cv.createTrackbar('Ring 6', 'Frame', 108, 300, nothing)
cv.createTrackbar('Ring 5', 'Frame', 132, 300, nothing)
cv.createTrackbar('Ring 4', 'Frame', 156, 300, nothing)
cv.createTrackbar('Ring 3', 'Frame', 180, 300, nothing)
cv.createTrackbar('Ring 2', 'Frame', 204, 300, nothing)
cv.createTrackbar('Ring 1', 'Frame', 228, 300, nothing)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error with camera")
        break

    # Perform perspective correction on the frame (if needed)
    points_src = np.array([[0, 49], [78, 578],[637, 49], [558, 578] ])
    points_dst = np.float32([[0, 0], [0, 580], [500, 0], [500, 580]])
    matrix, _ = cv.findHomography(points_src, points_dst)
    image_out = cv.warpPerspective(frame, matrix, (500, 580))
    frame = image_out

    # Read trackbar values for center and ring sizes
    center_x = cv.getTrackbarPos('Center X', 'Frame')
    center_y = cv.getTrackbarPos('Center Y', 'Frame')
    ring_10 = cv.getTrackbarPos('Ring 10', 'Frame')
    ring_9 = cv.getTrackbarPos('Ring 9', 'Frame')
    ring_8 = cv.getTrackbarPos('Ring 8', 'Frame')
    ring_7 = cv.getTrackbarPos('Ring 7', 'Frame')
    ring_6 = cv.getTrackbarPos('Ring 6', 'Frame')
    ring_5 = cv.getTrackbarPos('Ring 5', 'Frame')
    ring_4 = cv.getTrackbarPos('Ring 4', 'Frame')
    ring_3 = cv.getTrackbarPos('Ring 3', 'Frame')
    ring_2 = cv.getTrackbarPos('Ring 2', 'Frame')
    ring_1 = cv.getTrackbarPos('Ring 1', 'Frame')

    # Draw rings on the frame
    frame = drawRings(frame)

    # Display the frame with rings
    cv.imshow("Frame", frame)

    # Exit condition (press 'q' to quit)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release resources and close windows
cap.release()
cv.destroyAllWindows()
