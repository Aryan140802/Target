import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
import requests
import json


# calibration_data = np.load('calibration_params.npz')
# mtx = calibration_data['mtx']
# dist = calibration_data['dist']


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
score = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
angles = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
score_sum = 0
URL = 'http://127.0.0.1:5000/api/score'


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
        points_dst = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])

        matrix, _ = cv.findHomography(points_src, points_dst)
        image_out = cv.warpPerspective(frame, matrix, (500, 500))
        frame = image_out
        markers_found = True

    return frame, markers_found


def cleanCircles(concentricCircles):
    cleanedCircles = []
    for x, y, r in concentricCircles:
        if 230 < x < 270:
            cleanedCircles.append((x, y, r))

    tempRadius = {}

    for x, y, r in cleanedCircles:
        tempRadius[round(r, -1)] = (x, y, r)

    print(tempRadius)

    return tempRadius


def merge_tuples(tuples, threshold=5):
    tuples.sort(key=lambda x: x[2])

    merged = []

    current_group = [tuples[0]]

    for i in range(1, len(tuples)):
        current_tuple = tuples[i]
        last_tuple = current_group[-1]

        if abs(current_tuple[2] - last_tuple[2]) <= threshold:
            current_group.append(current_tuple)
        else:
            merged.append(merge_group(current_group))
            current_group = [current_tuple]

    merged.append(merge_group(current_group))

    return merged


def merge_group(group):
    avg_x = sum(t[0] for t in group) / len(group)
    avg_y = sum(t[1] for t in group) / len(group)
    avg_r = sum(t[2] for t in group) / len(group)
    return (int(avg_x), int(avg_y), int(avg_r))


def detectRings(frame, canvas):
    gray_cropped_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_cropped_frame, (5, 5), 0)
    canny = cv.Canny(blurred, 50, 100)

    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    concentricCircles = []
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
        ((x, y), radius) = cv.minEnclosingCircle(contour)

        concentricCircles.append((int(x), int(y), int(radius)))

    cleanedCircles = merge_tuples(concentricCircles, 10)
    for x, y, radius in cleanedCircles:
        if 4 < radius < 300 and 245 < x < 255 and 285 < y < 290:
            cv.circle(canvas, (x, y), (radius), (0, 0, 0), 2)
            cv.circle(canvas, (x, y), 2, (0, 255, 255), 3)

    return canvas


def detectWhiteRingBullets(frame, canvas):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(frame, 5)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    mask = th1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=2)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    min_area = 0
    max_area = 400

    bullets = []

    for contour in contours:
        area = cv.contourArea(contour)
        if min_area <= area <= max_area:
            approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
            ((x, y), radius) = cv.minEnclosingCircle(contour)
            if (230 > calculateDistance(int(x), int(y)) > 80):
                bullets.append((int(x), int(y)))
                print(area)
                cv.circle(canvas, (int(x), int(y)), (int(radius)), (0, 0, 255), -1)

    return canvas, bullets


def detectBlackRingBullets(frame, canvas):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(frame, 5)
    img = cv.bitwise_not(img)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    mask = th1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=2)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    min_area = 50
    max_area = 250

    bullets = []

    for contour in contours:
        area = cv.contourArea(contour)
        # print(area)
        if min_area <= area <= max_area:
            approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
            ((x, y), radius) = cv.minEnclosingCircle(contour)
            if (calculateDistance(int(x), int(y)) <= 80):
                bullets.append((int(x), int(y)))
                cv.circle(canvas, (int(x), int(y)), (int(radius)), (255, 0, 0), -1)

    return canvas, bullets


def drawRings(canvas, center_x=251, center_y=287):
    cv.circle(canvas, (center_x, center_y), (12), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (36), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (60), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (84), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (108), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (132), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (156), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (180), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (204), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (228), (255, 0, 255), 2)
    # cv.circle(canvas, (center_x, center_y), (250), (255, 0, 255), 2)

    return canvas


def calculateDistance(x1, y1, x2=251, y2=287):
    radius = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return radius


def displayScore(score, canvas):
    cv.putText(canvas, f'Score: {score}', (200, 40), cv.FONT_HERSHEY_PLAIN, 3, (80, 127, 255), 3)
    return canvas


def sendData(image, angles):
    _, img_encoded = cv.imencode('.jpg', image)
    files = {'image': img_encoded.tobytes(), 'angles': (None, json.dumps(angles), 'application/json')}
    return requests.post(URL, files=files)


def calculateAngle(x, y):
    delta_x = (x - 251)
    delta_y = (y - 287)
    if delta_x == 0:
        return -90
    else:
        # slope_hole = delta_y/delta_x
        # angle = round(math.atan(abs(slope_hole-slope_base/(1+ slope_hole*slope_base))) * 180 / math.pi)
        angle = round(math.atan2(delta_y, delta_x) * 180 / math.pi)
        return angle


def updateScore(bullets):
    global score_sum
    for x, y in bullets:
        dist = calculateDistance(x, y)
        angle = calculateAngle(x, y)
        if 0 <= dist <= 12:
            score[10].append((x, y))
            score_sum += 10
            angles[10].append(angle)
        elif 12 < dist <= 36:
            score[9].append((x, y))
            score_sum += 9
            angles[9].append(angle)
        elif 36 < dist <= 60:
            score[8].append((x, y))
            score_sum += 8
            angles[8].append(angle)
        elif 60 < dist <= 84:
            score[7].append((x, y))
            score_sum += 7
            angles[7].append(angle)
        elif 84 < dist <= 108:
            score[6].append((x, y))
            score_sum += 6
            angles[6].append(angle)
        elif 108 < dist <= 132:
            score[5].append((x, y))
            score_sum += 5
            angles[5].append(angle)
        elif 132 < dist <= 156:
            score[4].append((x, y))
            score_sum += 4
            angles[4].append(angle)
        elif 156 < dist <= 180:
            score[3].append((x, y))
            score_sum += 3
            angles[3].append(angle)
        elif 180 < dist <= 204:
            score[2].append((x, y))
            score_sum += 2
            angles[2].append(angle)
        elif 204 < dist <= 228:
            score[1].append((x, y))
            score_sum += 1
            angles[1].append(angle)


ret, frame = cap.read()

# frame = cv.undistort (frame, mtx, dist, None)
corrected_image, target_detected = correctPerspective(frame)
output_frame = corrected_image.copy()

if target_detected:
    # output_frame = drawRings(output_frame)
    output_frame, white_ring_bullets = detectWhiteRingBullets(corrected_image, output_frame)
    output_frame, black_ring_bulllets = detectBlackRingBullets(corrected_image, output_frame)
    updateScore(white_ring_bullets)
    updateScore(black_ring_bulllets)
    print(angles)
    new_score = 0
    score_dict = {"bullets": [], "total_score": 0}

    try:
        # response = sendData(output_frame, score_dict)
        response = sendData(output_frame, angles)
        print(f"Server response: {response.status_code}, {response.text}")
        pass
    except Exception as e:
        print("The error is: ", e)

    output_frame = displayScore(score_sum, output_frame)

# cv.imshow('frame', output_frame)
#
# cv.waitKey(0)

cap.release()
cv.destroyAllWindows()