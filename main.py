from flask import Flask, Response, render_template, send_file
import cv2
import datetime
import os
import threading
import webbrowser

app = Flask(__name__, static_folder='static')

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Change the parameter to specify a different camera (0 for default)

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

def generate():
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

        # Encode the frame as JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
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

if __name__ == '__main__':
    start_recording()
    webbrowser.open('http://localhost:5000/stream_and_recordings')  # Open the page in the default browser
    app.run(host='0.0.0.0', port=5000, debug=True)
