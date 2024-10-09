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

center_x = 258
center_y = 268

ring_10x = 8
ring_10 = 18
ring_9 = 44
ring_8 = 72
ring_7 = 97
ring_6 = 122
ring_5 = 151
ring_4 = 179
ring_3 = 211
ring_2 = 240
ring_1 = 263


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
    print(score)
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
            print(f"Score for Hole 10: {score[10]} (Total: {score_sum})")  # Print current score for hole 10
            angles[10].append(angle)
        elif ring_10 < dist <= ring_9:
            score[9].append((x, y))
            score_sum += 9
            print(f"Score for Hole 9: {score[9]} (Total: {score_sum})")  # Print current score for hole 9
            angles[9].append(angle)
        elif ring_9 < dist <= ring_8:
            score[8].append((x, y))
            score_sum += 8
            print(f"Score for Hole 8: {score[8]} (Total: {score_sum})")
            angles[8].append(angle)
        elif ring_8 < dist <= ring_7:
            score[7].append((x, y))
            score_sum += 7
            print(f"Score for Hole 7: {score[7]} (Total: {score_sum})")
            angles[7].append(angle)
        elif ring_7 < dist <= ring_6:
            score[6].append((x, y))
            score_sum += 6
            print(f"Score for Hole 6: {score[6]} (Total: {score_sum})")
            angles[6].append(angle)
        elif ring_6 < dist <= ring_5:
            score[5].append((x, y))
            score_sum += 5
            print(f"Score for Hole 5: {score[5]} (Total: {score_sum})")
            angles[5].append(angle)
        elif ring_5 < dist <= ring_4:
            score[4].append((x, y))
            score_sum += 4
            print(f"Score for Hole 4: {score[4]} (Total: {score_sum})")
            angles[4].append(angle)
        elif ring_4 < dist <= ring_3:
            score[3].append((x, y))
            score_sum += 3
            print(f"Score for Hole 3: {score[3]} (Total: {score_sum})")
            angles[3].append(angle)
        elif ring_3 < dist <= ring_2:
            score[2].append((x, y))
            score_sum += 2
            print(f"Score for Hole 2: {score[2]} (Total: {score_sum})")
            angles[2].append(angle)
        elif ring_2 < dist <= ring_1:
            score[1].append((x, y))
            score_sum += 1
            print(f"Score for Hole 1: {score[1]} (Total: {score_sum})")
            angles[1].append(angle)





ret, frame = video_feed.read()

points_src = np.array([[9, 62], [9, 569], [619, 89], [548, 622]])
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

# cv.waitKey(0)
video_feed.cleanup()
cv.destroyAllWindows()