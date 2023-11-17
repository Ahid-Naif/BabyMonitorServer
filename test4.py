from flask import Flask, Response
import cv2

app = Flask(__name__)

# Initialize video capture with the first camera device
cap = cv2.VideoCapture(0)

def generate_frames():
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
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Return a simple HTML page with an embedded video feed
    return """
    <html>
    <head>
    <title>Raspberry Pi Camera Stream</title>
    </head>
    <body>
    <h1>Raspberry Pi Camera Stream</h1>
    <img src="/video_feed" />
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
