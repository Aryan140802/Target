import cv2 as cv
import numpy as np
import requests


class VideoFeed:
    def __init__(self, flask_ip="127.0.0.1", flask_port=5000, video_port=8000):
        self.flask_ip = flask_ip
        self.flask_port = flask_port
        self.video_port = video_port
        self.selected_ip = self.get_selected_ip()

        if not self.selected_ip:
            raise ValueError("No selected IP provided. Please ensure you have selected the device in the app.")

        # Initialize video capture with the selected IP
        self.cap = cv.VideoCapture(f"http://{self.selected_ip}:{self.video_port}/video_feed")

        # URL to send frames for processing
        self.api_url = f'http://{self.flask_ip}:{self.flask_port}/api/starter'

    def get_selected_ip(self):
        try:
            response = requests.get(f'http://{self.flask_ip}:{self.flask_port}/api/selected_ip')  # Correct Flask endpoint
            if response.status_code == 200:
                ip = response.json().get('selected_ip')
                if ip:
                    return ip
                else:
                    print("No IP address selected.")
                    return None
            else:
                print(f"Error fetching selected IP: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def send_data(self, image):
        _, img_encoded = cv.imencode('.jpg', image)
        files = {'image': img_encoded.tobytes()}

        try:
            response = requests.post(self.api_url, files=files)
            print(f"Server response: {response.status_code}, {response.text}")
            return response
        except Exception as e:
            print(f"Error sending data: {e}")
            return None

    def process_frame(self):
        ret, frame = self.cap.read()

        if ret:
            points_src = np.array([[9, 83], [9, 590], [563, 77], [578, 587]])
            points_dst = np.float32([[0, 0], [0, 580], [500, 0], [500, 580]])

            matrix, _ = cv.findHomography(points_src, points_dst)
            frame_out = cv.warpPerspective(frame, matrix, (500, 580))

            return frame_out
        else:
            print("Failed to capture frame")
            return None

    def stream_and_send(self):
        while True:
            frame = self.process_frame()

            if frame is not None:
                self.send_data(frame)
                # Uncomment this for debugging (to see the frame locally)
                cv.imshow("Processed Frame", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv.destroyAllWindows()
        print("Video feed stopped.")

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame



# This can be imported and used in another script
if __name__ == "__main__":
    video_feed = VideoFeed()
    video_feed.stream_and_send()