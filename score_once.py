import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
import requests
import json
from starter_script import VideoFeed  # Import the VideoFeed class


score = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
angles = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
score_sum = 0
URL = 'http://127.0.0.1:5000/api/score'

center_x = 252
center_y = 257

ring_10x = 8
ring_10 = 18
ring_9 = 38
ring_8 = 61
ring_7 = 81
ring_6 = 107
ring_5 = 135
ring_4 = 165
ring_3 = 188
ring_2 = 222
ring_1 = 253


# Initialize the video feed
video_feed = VideoFeed()

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


def drawRings(canvas, center_x=250, center_y=250):
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


def calculateDistance(x1, y1, x2=251, y2=252):
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
    delta_x = (x - center_x)
    delta_y = (y - center_y)
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
        if 0 <= dist <= ring_10:
            score[10].append((x, y))
            score_sum += 10
            angles[10].append(angle)
        elif ring_10 < dist <= ring_9:
            score[9].append((x, y))
            score_sum += 9
            angles[9].append(angle)
        elif ring_9 < dist <= ring_8:
            score[8].append((x, y))
            score_sum += 8
            angles[8].append(angle)
        elif ring_8 < dist <= ring_7:
            score[7].append((x, y))
            score_sum += 7
            angles[7].append(angle)
        elif ring_7 < dist <= ring_6:
            score[6].append((x, y))
            score_sum += 6
            angles[6].append(angle)
        elif ring_6 < dist <= ring_5:
            score[5].append((x, y))
            score_sum += 5
            angles[5].append(angle)
        elif ring_5 < dist <= ring_4:
            score[4].append((x, y))
            score_sum += 4
            angles[4].append(angle)
        elif ring_4 < dist <= ring_3:
            score[3].append((x, y))
            score_sum += 3
            angles[3].append(angle)
        elif ring_3 < dist <= ring_2:
            score[2].append((x, y))
            score_sum += 2
            angles[2].append(angle)
        elif ring_2 < dist <= ring_1:
            score[1].append((x, y))
            score_sum += 1
            angles[1].append(angle)


def correctFisheye(frame1):
    height, width, _ = frame.shape

    fish_eye = {
        'focal': [433, 1500],
        'cx': [376, 1500], 'cy': [408, 600],
        'k1': [30, 100], 'k2': [100, 100]
    }
    wrap_points = {
        'x1': 19, 'y1': 116,
        'x2': 84, 'y2': 677,
        'x3': 675, 'y3': 693,
        'x4': 789, 'y4': 103
    }
    # global focal, cx, cy, k1, k2
    focal = fish_eye['focal'][0] - 60
    cx = fish_eye['cx'][0] - 60
    cy = fish_eye['cy'][0] - 60
    k1 = (fish_eye['k1'][0] - 60)/100
    k2 = (fish_eye['k2'][0] - 60)/100
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0, 0, 1]], dtype=np.float32)  # Ensure matrix is of type float32

    D = np.array([k1, k2, 0, 0], dtype=np.float32)  # Ensure the distortion coefficients are also float32
    new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=1)
    undistorted_image = cv.fisheye.undistortImage(frame1, K, D=D, Knew=new_K)
    points = np.array([
        [wrap_points['x1'], wrap_points['y1']],
        [wrap_points['x2'], wrap_points['y2']],
        [wrap_points['x3'], wrap_points['y3']],
        [wrap_points['x4'], wrap_points['y4']]
    ], dtype=np.int32)
    # cv.polylines(undistorted_image, [points], True, (0, 255, 0), 3)
    # cv.imshow('Fisheye Correction', undistorted_image)
    return undistorted_image

ret, frame = video_feed.read()

frame = correctFisheye(frame)

points_src = np.array([[0, 47],[71,551],[646,77],[557,578]])
points_dst = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])

matrix, _ = cv.findHomography(points_src, points_dst)
image_out = cv.warpPerspective(frame, matrix, (500, 500))
corrected_image = image_out

# corrected_image, target_detected = correctPerspective(frame)
output_frame = corrected_image.copy()

# if target_detected:
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

cv.imshow('frame', output_frame)

cv.waitKey(0)

video_feed.cleanup()
cv.destroyAllWindows()