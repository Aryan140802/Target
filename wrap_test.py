"""

Press   :   Function
  P     :   prints wrapped points
  O     :   Shows Original Image
  W     :   Shows Wrapped Image
  V     :   Shows TrackBar For Wrapping points
  R     :   Shows Rings on Image
  B     :   Shows Binary Threshold Image
  S     :   Shows Sharpened Image
  Q     :   Quits Entire Program

NOTE : If a window is already on, pressing its corresponding Button will Close it
NOTE : Sharpened Image ('S') will only show up if Binary Image is active.
"""

# TODO: Add Multi Threading

import sys

import cv2
import numpy as np

output_pts = np.float32([[0, 0], [0, 500], [500, 500], [500, 0]])

show_Rings = False
show_Wrapped = False
show_wrapPoints = False
show_Params = False
sharpening = False
show_OG = True

Thresh = 127
k_size = 3
lim = 0

wrap_points = {
    'x1': 0, 'y1': 136,
    'x2': 93, 'y2': 766,
    'x3': 634, 'y3': 748,
    'x4': 770, 'y4': 143
}

rings = {
    'center_x': 247, 'center_y': 250,
    'ring_11':0,
    'ring_10': 0, 'ring_9': 0, 'ring_8': 46,
    'ring_7': 74, 'ring_6': 99, 'ring_5': 127, 'ring_4': 153,
    'ring_3': 170, 'ring_2': 204, 'ring_1': 227,
}


params = {'alpha': 1.5, 'beta': -0.5, 'gamma': 0}

key_actions = {
    ord('W'): 'show_Wrapped',
    ord('O'): 'show_OG',
    ord('V'): 'show_wrapPoints',
    ord('R'): 'show_Rings',
    ord('B'): 'show_Params',
    ord('S'): 'sharpening'
}


def manage_window(window_name, should_show, show_func=None, close_func=None, draw_func=None):

    if should_show:
        if draw_func:
            draw_func()
        if show_func and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            show_func()
    else:
        if close_func:
            close_func(window_name)
        else:
            cv2.destroyWindow(window_name)


def closeWindow(name: str):
    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(name)


def startWrapPoints():
    cv2.namedWindow("WrapPoints")
    cv2.resizeWindow("WrapPoints", (400, 300))

    for point, initial_val in wrap_points.items():
        cv2.createTrackbar(point, 'WrapPoints', initial_val, lim,
                           lambda val, key=point: update_point(val, key, wrap_points))


def startParams():
    cv2.namedWindow("Params")
    cv2.resizeWindow("Params", (400, 500))
    for key1, initial_val in params.items():
        cv2.createTrackbar(key1, 'Params', int(initial_val*2+20), 40,
                           lambda val, key=key1: update_point(((val / 2) - 10), key, params))

    cv2.createTrackbar("Thresh", "Params", Thresh, 255, lambda x: globals().update(Thresh=x))
    cv2.createTrackbar("k_size", "Params", k_size, 255, lambda x: globals().update(k_size=(x + (0 if x % 2 else 1))))


def imageProcessor(frame1, sharp: bool):

    frame1 = frame1.copy()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise and improve detection
    gray_blurred = cv2.GaussianBlur(gray, (k_size, k_size), 2)

    output_image = gray_blurred
    if sharp:
        output_image = sharpened_image = cv2.addWeighted(gray, params['alpha'], gray_blurred, params['beta'], params['gamma'])
        cv2.imshow("Sharpened Image", sharpened_image)
    else:
        closeWindow("Sharpened Image")

    _, binary_image = cv2.threshold(output_image, Thresh, 255, cv2.THRESH_BINARY)

    cv2.imshow("Binary Image", binary_image)


def startRings():
    cv2.namedWindow("Ring_Points")
    cv2.resizeWindow("Ring_Points", (400, 850))
    for key1, initial_val in rings.items():
        cv2.createTrackbar(key1, 'Ring_Points', initial_val, 400, lambda val, key=key1: update_point(val, key, rings))


def drawRings(canvas):
    center_x, center_y = rings['center_x'], rings['center_y']
    cv2.circle(canvas, (center_x, center_y), 1, (0, 0, 255), 2)

    for i in range(1, 12):
        cv2.circle(canvas, (center_x, center_y), rings["ring_" + str(i)], (255, 0, 255), 2)

    cv2.imshow("Rings", canvas)


def update_point(val, key, cus_dict):
    cus_dict[key] = val


def printAll():
    print(f"wrap Points: {wrap_points}")
    print(rings)
    print(params)
    print(f"Threshold: {Thresh}")
    print(f"Kernal Size: {k_size}")


if __name__ == "__main__":
    cap = cv2.VideoCapture("http://192.168.1.4:8000/video_feed")
    _, frame = cap.read()
    lim = max(frame.shape[0], frame.shape[1])

    while True:

        _, frame = cap.read()

        points = np.array([
            [wrap_points['x1'], wrap_points['y1']],
            [wrap_points['x2'], wrap_points['y2']],
            [wrap_points['x3'], wrap_points['y3']],
            [wrap_points['x4'], wrap_points['y4']]
        ], dtype=np.int32)
        cv2.polylines(frame, [points], True, (0, 255, 0), 3)

        input_pts = np.float32(points)
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        frameW = cv2.warpPerspective(frame, M, (500, 500))

        k = cv2.waitKey(1)

        # cv2.circle(frameW, (250,250), 50,(0, 0, 0), 2)

        if k == ord('P'):
            print(points)
        elif k in key_actions:
            globals()[key_actions[k]] = not globals()[key_actions[k]]
        elif k == ord('Q'):
            printAll()
            cv2.destroyAllWindows()
            break

        manage_window("WrapPoints", show_wrapPoints, startWrapPoints, closeWindow)
        manage_window("Wrapped Image", show_Wrapped, None, closeWindow, lambda: cv2.imshow("Wrapped Image", frameW))
        manage_window("Original Image", show_OG, None, closeWindow, lambda: cv2.imshow("Original Image", frame))
        manage_window("Ring_Points", show_Rings, startRings, closeWindow, lambda: drawRings(frameW))
        manage_window("Params", show_Params, startParams, closeWindow, lambda: imageProcessor(frameW, sharpening))

        if not show_Rings:
            closeWindow("Rings")

        if not show_Params:
            closeWindow("Binary Image")
            closeWindow("Sharpened Image")
    sys.exit()
