from flask import Flask, render_template, Response, request
import dlib
import math
import cv2
import csv
from datetime import datetime
import easyocr
import os
import time

app = Flask(__name__)

# Global variables
video = None
metrics = 8.8
speed_limit = 90
log_data = []
camera_running = True

# Initialize car cascade
carCascade = cv2.CascadeClassifier("C:/Users/adwet/Downloads/VEHICLE-SPEED-TRACKING-main/VEHICLE-SPEED-TRACKING-main/myhaar.xml") 

# Initialize easyOCR reader globally
reader = easyocr.Reader(['en'], gpu=False)

def initialize_video(selection, video_file=0):
    global video
    if selection == 'Camera':
        video = cv2.VideoCapture(0)
    elif selection == 'Video' and video_file:
        video_file = os.path.abspath(video_file)
        video = cv2.VideoCapture(video_file)
    return video.isOpened()

def log_data_to_csv(speed, number_plate):
    if speed > 0 and number_plate:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = [now, f"{speed} km/h", number_plate]
        log_data.append(log_entry)
        with open('vehicle_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)

def estimate_speed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = metrics
    d_meters = d_pixels / ppm
    fps = 10
    speed = d_meters * fps * 3.6
    return speed

def detect_number_plate_easyocr(image, speed):
    results = reader.readtext(image)
    if results:
        number_plate = results[0][1].strip()
        log_data_to_csv(speed, number_plate)
        return number_plate
    return ""

def track_multiple_objects():
    global video, metrics, speed_limit, log_data, camera_running
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    while True:
        if not camera_running:  # Check if the camera is stopped
            print("Camera stopped. Waiting to resume...")
            time.sleep(0.5)  # Sleep to prevent excessive CPU usage
            continue

        if video is None or not video.isOpened():
            print("Video source is not available.")
            break

        rc, image = video.read()
        if not rc:
            print("Failed to read from video source.")
            break

        image = cv2.resize(image, (1280, 720))
        resultImage = image.copy()

        frameCounter += 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                x_bar, y_bar = x + 0.5 * w, y + 0.5 * h

                matchCarID = None
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    t_x, t_y = int(trackedPosition.left()), int(trackedPosition.top())
                    t_w, t_h = int(trackedPosition.width()), int(trackedPosition.height())
                    t_x_bar, t_y_bar = t_x + 0.5 * t_w, t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h))):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
            t_x, t_y, t_w, t_h = int(trackedPosition.left()), int(trackedPosition.top()), int(trackedPosition.width()), int(trackedPosition.height())
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        for i in carLocation1.keys():
            [x1, y1, w1, h1] = carLocation1[i]
            [x2, y2, w2, h2] = carLocation2[i]
            carLocation1[i] = [x2, y2, w2, h2]

            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                if speed[i] is None or (speed[i] == 0 and y1 >= 275 and y1 <= 285):
                    speed[i] = estimate_speed([x1, y1, w1, h1], [x2, y2, w2, h2])

                # Update speed display continuously if speed is greater than 0
                if speed[i] is not None and speed[i] > 0:
                    cv2.putText(resultImage, str(int(speed[i])) + " km/h", (int(x1 + w1 / 2), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', resultImage)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input', methods=['POST'])
def input_settings():
    global metrics, speed_limit
    metrics = float(request.form['ppm'])
    speed_limit = float(request.form['max_speed'])
    return render_template('selection.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global video
    input_method = request.form['input_method']
    video_file = request.files['video_file'] if 'video_file' in request.files else None

    if input_method == 'Video' and video_file:
        video_path = os.path.join("uploads", video_file.filename)
        video_file.save(video_path)
        video_path = os.path.abspath(video_path)
        success = initialize_video(input_method, video_path)
    else:
        success = initialize_video(input_method)

    if not success:
        return render_template('selection.html', error="Video source is not available.")
    
    return render_template('tracking.html', logged_data=log_data)

@app.route('/video_feed')
def video_feed():
    return Response(track_multiple_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera_running
    camera_running = False
    return {'status': 'stopped'}

@app.route('/resume_camera')
def resume_camera():
    global camera_running
    camera_running = True
    return {'status': 'running'}

@app.route('/log_data')
def log_data_view():
    return {'data': log_data}

if __name__ == '__main__':
    if not os.path.exists('vehicle_data.csv'):
        with open('vehicle_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["DateTime", "Speed (km/h)", "Number Plate"])

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)