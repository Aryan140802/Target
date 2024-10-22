import cv2 as cv
import numpy as np
import cv2.aruco as aruco

cap = cv.VideoCapture('http://192.168.1.10:8000/video_feed')
cv.namedWindow("Frame")

# calibration_data = np.load('calibration_params.npz')
# mtx = calibration_data['mtx']
# dist = calibration_data['dist']

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

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


def getArucoCenters(corners):
    centers = []
    for marker in corners:
        x_sum = 0
        y_sum = 0
        for x, y in marker[0]:
            x_sum += x
            y_sum += y
        center = (int(x_sum // 4), int(y_sum // 4))
        centers.append(center)
    return centers


def addToDict(centers, ids):
    center_dict = {}
    for i in range(len(centers)):
        center_dict[ids[i][0]] = centers[i]
    return center_dict


def correctPerspective(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(image=gray_frame, dictionary=aruco_dict, parameters=parameters)
    markers_found = False
    if ids is not None and len(ids) == 4:
        centers = getArucoCenters(corners)
        center_dict = addToDict(centers, ids)
        points_src = np.array([center_dict[0], center_dict[3], center_dict[1], center_dict[2]])
        points_dst = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])

        matrix, _ = cv.findHomography(points_src, points_dst)
        image_out = cv.warpPerspective(frame, matrix, (500, 500))
        frame = image_out
        markers_found = True

    return frame, markers_found


def drawRings(canvas):
    cv.circle(canvas, (center_x, center_y), (1), (0, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_10), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_9), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_8), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_7), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_6), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_5), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_4), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_3), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_2), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_1), (255, 0, 255), 2)

    return canvas


def nothing(_):
    pass


# Trackbars for adjusting the center and rings dynamically
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

     # Undistort the captured frame
    # frame = cv.undistort(frame, mtx, dist, None)

    # Process frame to detect markers and correct perspective
    processed_frame, target_detected = correctPerspective(frame)

    if target_detected:
        # Get the trackbar positions to dynamically adjust the rings
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

        # Draw the rings on the processed frame
        processed_frame = drawRings(processed_frame)

    # Combine original and processed frame side by side
    # combined_frame = np.hstack((frame, processed_frame))

    # Display the combined frame
    cv.imshow("Frame", processed_frame)

    key = cv.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
