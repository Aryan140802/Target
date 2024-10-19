import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
import requests
import json
import time
from starter_script import VideoFeed  # Import the VideoFeed class


calibration_data = np.load('calibration_params.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
prev_frame = None
fps_limit = 1
start_time = time.time()
score = {10:[],9:[],8:[],7:[],6:[],5:[],4:[],3:[],2:[],1:[]}
angles = {10:[],9:[],8:[],7:[],6:[],5:[],4:[],3:[],2:[],1:[]}
score_sum = 0
URL = 'http://127.0.0.1:5000/api/score'





# Initialize the video feed
video_feed = VideoFeed()




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


def getBullets(th1, output_frame, draw=True):
    contours = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    bullets = []
    for contour in contours:
        # print(contour)
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
        ((x, y), radius) = cv.minEnclosingCircle(contour)
        bullets.append((int(x), int(y)))
        cv.circle(output_frame, (int(x), int(y)), (int(radius)), (0, 0, 255), -1)

    return bullets


def calculateDistance(x1, y1, x2=251, y2=287):
    radius = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    # print(x1,y1,x2,y2,radius)
    return radius


def updateScore(bullets):
    global score_sum, score, angles

    for x, y in bullets:
        dist = calculateDistance(x, y)
        angle = calculateAngle(x, y)
        if 0 <= dist <= 12:
            score[10].append((x, y))
            score_sum += 10
            angles['10'].append(angle)
        elif 12 < dist <= 36:
            score[9].append((x, y))
            score_sum += 9
            angles['9'].append(angle)
        elif 36 < dist <= 60:
            score[8].append((x, y))
            score_sum += 8
            angles['8'].append(angle)
        elif 60 < dist <= 84:
            score[7].append((x, y))
            score_sum += 7
            angles['7'].append(angle)
        elif 84 < dist <= 108:
            score[6].append((x, y))
            score_sum += 6
            angles['6'].append(angle)
        elif 108 < dist <= 132:
            score[5].append((x, y))
            score_sum += 5
            angles['5'].append(angle)
        elif 132 < dist <= 156:
            score[4].append((x, y))
            score_sum += 4
            angles['4'].append(angle)
        elif 156 < dist <= 180:
            score[3].append((x, y))
            score_sum += 3
            angles['3'].append(angle)
        elif 180 < dist <= 204:
            score[2].append((x, y))
            score_sum += 2
            angles['2'].append(angle)
        elif 204 < dist <= 228:
            score[1].append((x, y))
            score_sum += 1
            angles['1'].append(angle)


def drawFrame(frame):
    i = 1
    for points in score.keys():
        frame = cv.putText(frame, f"{points}:{len(score[points])}", (0, 20 * i), cv.FONT_HERSHEY_COMPLEX, 0.5,
                           (0, 0, 255), 2)
        i += 1


def sendData(image, angles):
    _, img_encoded = cv.imencode('.jpg', image)
    # files = {'score': (None, json.dumps(score), 'application/json'),'image': img_encoded.tobytes(),
    #          'angles': (None, json.dumps(angles), 'application/json')}
    files = {'image': img_encoded.tobytes(),
             'angles': (None, json.dumps(angles), 'application/json')}
    # data = {'score': score}

    # response = requests.post(URL, files=files, json=score)
    response = requests.post(URL, files=files)

    return response


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


def get_current_score():
    try:
        response = requests.get('http://127.0.0.1:5000/api/data')
        if response.status_code == 200:

            return response.json().get('angles')
        else:
            print("Error fetching score")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


angles = get_current_score()
print(type(angles))
if angles is None:
    angles = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
for val in angles.keys():
    score_sum += int(val) * len(angles[val])

# print(angles)
while True:
    ret, frame = video_feed.read()
    frame = cv.undistort(frame, mtx, dist, None)

    curr_time = time.time()

    if not ret:
        print("Error with Webcam")
        break

    if ((curr_time - start_time)) > fps_limit:
        # if True:

        # ret = True
        # frame = cv.imread("./targetWithHole.jpg")
        corrected_image, target_detected = correctPerspective(frame)
        frame = corrected_image.copy()

        frame = cv.GaussianBlur(frame, (5, 5), 0)

        # hcont = cv.hconcat([output_frame,cv.cvtColor(mask1,cv.COLOR_GRAY2BGR),cv.cvtColor(mask2,cv.COLOR_GRAY2BGR)])

        # cv.imshow('frame', hcont)

        output_frame = frame.copy()
        if target_detected:
            if prev_frame is not None:
                diff_frame = 255 - cv.absdiff(frame, prev_frame)
                # output_frame = diff_frame
                grayscale_frame = cv.cvtColor(diff_frame, cv.COLOR_BGR2GRAY)
                ret2, th1 = cv.threshold(grayscale_frame, 170, 255, cv.THRESH_BINARY)
                th1 = cv.bitwise_not(th1)
                kernel = np.ones((3, 3), np.uint8)

                th1 = cv.dilate(th1, kernel, iterations=3)
                # th1 = cv.erode(th1, kernel, iterations=5)

                # output_frame = th1
                bullets = getBullets(th1, output_frame)
                prev_score_sum = score_sum
                updateScore(bullets)
                # print(score)
                # drawFrame(output_frame)
                if prev_score_sum != score_sum:
                    try:
                        response = sendData(output_frame, angles)
                        print(f"Server response: {response.status_code}, {response.text}")

                    except Exception as e:
                        print("The error is: ", e)

                # print("frame done")

                output_frame = cv.hconcat([output_frame, cv.cvtColor(grayscale_frame, cv.COLOR_GRAY2BGR),
                                           cv.cvtColor(th1, cv.COLOR_GRAY2BGR)])

            prev_frame = frame

        # cv.imshow('frame', output_frame)
        start_time = time.time()

    key = cv.waitKey(1)

    if key == ord('q'):
        break


video_feed.cleanup()
cv.destroyAllWindows()