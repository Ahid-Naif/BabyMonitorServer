from flask import Flask, Response, render_template, jsonify
import cv2
import datetime
import os
import threading
import webbrowser
from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from PIL import Image, ImageDraw
import numpy as np
import time
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Change parameter to specify a different camera

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording_started = None

recordings_folder = 'static'

# Create the recordings folder if it doesn't exist
if not os.path.exists(recordings_folder):
    os.makedirs(recordings_folder)

# Lock for synchronization
lock = threading.Lock()

# Face detection and landmark models
face_detection_model = FaceDetection()
face_landmark_model = FaceLandmark()

# Variables to keep track of the covered state and time
global covered_state, face_covered, last_covered_time, display_enabled, body_exist
covered_state = False
face_covered = False  # covered flag
last_covered_time = 0
display_enabled = True 
body_exist = True

def start_recording():
    global out, recording_started
    recording_started = datetime.datetime.now()
    recording_filename = os.path.join(recordings_folder, f'video_{recording_started.strftime("%Y-%m-%d_%H-%M-%S")}.mp4')  # Change the file extension to .mp4
    out = cv2.VideoWriter(recording_filename, -1, 20.0, (640, 480))

def stop_recording():
    global out, recording_started
    if out is not None:
        out.release()
    recording_started = None

# Function to draw all landmarks as points on the image
def draw_all_landmarks(image, landmarks):
    if display_enabled:
        draw = ImageDraw.Draw(image)
        for landmark in landmarks:
            x = int(landmark.x * image.width)
            y = int(landmark.y * image.height)
            purple_color = (128, 0, 128)  # (R, G, B)
            draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], fill=purple_color, outline=purple_color)

def printStatus(img):
    if display_enabled and body_exist:
        if face_covered:
            text = "Danger"
            text_color = (0, 0, 255)  # Red color in BGR
        else:
            text = "Safe"
            text_color = (0, 255, 0)  # Green color in BGR
    
        # Convert PIL Image back to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Position of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (10, frame.shape[0] - 10)  # Position at bottom left corner
        font_scale = 1
        font_thickness = 2
        # Put the text on the frame
        cv2.putText(frame, text, bottom_left_corner, font, font_scale, text_color, font_thickness)
    else:
        # Convert PIL Image back to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    return frame

# Function to create a rectangular mask over a facial region (mouth or nose)
def mask_region_rect(image, landmarks, region_indices):
    xs = [landmarks[idx].x for idx in region_indices if idx < len(landmarks)]
    ys = [landmarks[idx].y for idx in region_indices if idx < len(landmarks)]
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_x = int(min_x * image.width)
    max_x = int(max_x * image.width)
    min_y = int(min_y * image.height)
    max_y = int(max_y * image.height)
    mask = Image.new('L', (image.width, image.height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([min_x, min_y, max_x, max_y], fill=255)
    return mask

# Function to calculate the skin presence ratio in a masked region
def calculate_skin_ratio(image, mask):
    if mask is None:
        return 0
    mask_np = np.array(mask)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    region = cv2.bitwise_and(cv_image, cv_image, mask=mask_np)
    ycrcb_image = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    skin_mask = cv2.inRange(ycrcb_image, min_YCrCb, max_YCrCb)    
    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = cv2.countNonZero(mask_np)
    skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
    return skin_ratio

# Indices for the mouth and nose landmarks
mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0]
nose_indices = [2, 98, 327, 331, 297]  # Replace with your model's indices

def is_there_face(detections):
    return (len(detections) > 0)

# Generate function that captures camera frames, processes them, and streams
def generate():
    global covered_state, face_covered, last_covered_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if it's time to start a new recording
        if recording_started is None or (datetime.datetime.now() - recording_started).seconds >= 20:
            stop_recording()
            start_recording()

        # Use lock to synchronize access to out
        with lock:
            out.write(frame)
            
        # Perform face detection and landmark drawing
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = face_detection_model(img)
        if is_there_face(detections) and body_exist:
            for detection in detections:
                roi = face_detection_to_roi(detection, img.size)
                landmarks = face_landmark_model(img, roi)
                draw_all_landmarks(img, landmarks)

                mouth_mask = mask_region_rect(img, landmarks, mouth_indices)
                nose_mask = mask_region_rect(img, landmarks, nose_indices)

                mouth_skin_ratio = calculate_skin_ratio(img, mouth_mask) if mouth_mask else 0
                nose_skin_ratio = calculate_skin_ratio(img, nose_mask) if nose_mask else 0

                mouth_covered = mouth_skin_ratio < 0.6
                nose_covered = nose_skin_ratio < 0.6

                if mouth_covered or nose_covered:
                    if not covered_state:
                        # If not previously covered, start the timer
                        last_covered_time = time.time()
                        covered_state = True
                    else:
                        # If already covered, check if 1 second has passed
                        if time.time() - last_covered_time > 0.5:
                            if mouth_covered or nose_covered:
                                face_covered = True
                            else:
                                covered_state = False
                else:
                    face_covered = False
        else:
            if not body_exist:
                if not covered_state:
                    # If not previously covered, start the timer
                    last_covered_time = time.time()
                    covered_state = True
                else:
                    # If already covered, check if 1 second has passed
                    if time.time() - last_covered_time > 0.5:
                        if not body_exist:
                            face_covered = True
                        else:
                            covered_state = False
            else:
                face_covered = False   

        frame = printStatus(img)
        # Encode the frame as JPEG for streaming
        ret, jpeg = cv2.imencode('.jpeg', frame)
        if not ret:
            break

        # Convert JPEG to bytes and yield it for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def get_recordings_list():
    return os.listdir(recordings_folder)

@app.route("/play_video/<filename>")
def play_video(filename):
    return render_template('play_video.html', filename=filename)

@app.route('/stream_and_recordings')
def stream_and_recordings():
    recordings = get_recordings_list()
    return render_template('stream_and_recordings.html', recordings=recordings)

@app.route('/api/stream_and_recordings')
def api_stream_and_recordings():
    recordings = get_recordings_list()
    return jsonify(recordings)

@app.route('/api/delete_all_recordings', methods=['DELETE'])
def delete_all_recordings():
    try:
        # Loop through all the files in the recordings folder and delete them
        for filename in os.listdir(recordings_folder):
            file_path = os.path.join(recordings_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        return jsonify({'message': 'All recordings deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-danger', methods=['GET'])
def check_value():
    return jsonify(face_covered), 200

if __name__ == '__main__':
    start_recording()
    webbrowser.open('http://localhost:5000/stream_and_recordings')  # Open the page in the default browser
    app.run(host='0.0.0.0', port=5000, debug=True)
