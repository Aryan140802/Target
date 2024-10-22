import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
import requests
import json
import time


# cap = cv.VideoCapture(1)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

URL = 'http://127.0.0.1:5000/api/starter'


# Initialize the video feed

def selected_ip():
    try:
        response = requests.get('http://127.0.0.1:5000/api/selected_ip')  # Flask endpoint
        if response.status_code == 200:
            return response.json().get('selected_ip')
        else:
            print("Error fetching selected IP")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

selected_ip =  selected_ip()  # Fetch selected IP from the Flask app

if selected_ip is None:
    raise ValueError("No selected IP provided. Please ensure you have selected the device in the app.")

# Use the selected IP in the video stream
cap = cv.VideoCapture(f"http://{selected_ip}:8000/video_feed")  # Use dynamic IP address


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
        points_dst = np.float32([[0, 0], [0, 580], [500, 0], [500, 580]])

        matrix, _ = cv.findHomography(points_src, points_dst)
        image_out = cv.warpPerspective(frame, matrix, (500, 580))
        frame = image_out
        markers_found = True

    return frame, markers_found


def sendData(image):
    _, img_encoded = cv.imencode('.jpg', image)

    files = {'image': img_encoded.tobytes()}

    response = requests.post(URL, files=files)

    return response


ret, frame = cap.read()

corrected_image, target_detected = correctPerspective(frame)

try:
    if target_detected:
        response = sendData(corrected_image)
        print(f"Server response: {response.status_code}, {response.text}")
except Exception as e:
    print("The error is: ", e)

cap.release()
cv.destroyAllWindows()
print("Starter script exited")