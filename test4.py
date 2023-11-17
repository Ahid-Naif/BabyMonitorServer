from flask import Flask, Response, render_template
import cv2
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize video capture with the first camera device
cap = cv2.VideoCapture(0)

# Directory for saving recordings
recordings_folder = 'static'
if not os.path.exists(recordings_folder):
    os.makedirs(recordings_folder)

def get_recordings_list():
    """Retrieves the list of recording files."""
    return os.listdir(recordings_folder)

def generate():
    while True:
        success, frame = cap.read()  # Read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate frame output with boundary

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    recordings = get_recordings_list()
    return render_template('stream_and_recordings.html', recordings=recordings)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
