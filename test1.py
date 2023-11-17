from flask import Flask, Response, render_template, jsonify, request
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

# Flask app setup with CORS (Cross-Origin Resource Sharing) enabled
app = Flask(__name__, static_folder='static')
CORS(app)

# Video capture setup with OpenCV
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

# Setup for video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
out = None  # VideoWriter object
recording_started = None  # Track start time of recording
recorded_video_duration = 20 # each recorded video will be 20 seconds long

# Directory for saving recordings
recordings_folder = 'static'
if not os.path.exists(recordings_folder):
    os.makedirs(recordings_folder)

# Threading lock for synchronization
lock = threading.Lock()

# Face detection and landmark models initialization
face_detection_model = FaceDetection()
face_landmark_model = FaceLandmark()

# Global variables for state management
global covered_state, face_covered, last_covered_time, display_enabled, body_exist
covered_state = False  # Track if face is covered
face_covered = False   # Flag for face covered state
last_covered_time = 0  # Last time face was covered
display_enabled = True  # Control display of landmarks and status
body_exist = True  # Track if a body exists in the frame

def start_recording():
    """Starts video recording."""
    global out, recording_started
    recording_started = datetime.datetime.now()
    recording_filename = os.path.join(recordings_folder, f'video_{recording_started.strftime("%Y-%m-%d_%H-%M-%S")}.mp4')
    # cv2.VideoWriter is used to create a video writer object for recording.
    # It takes several parameters:

    # recording_filename: The filename where the video will be saved.
    # -1: This specifies the codec to be used. A value of -1 prompts the user to select the codec from a list of available codecs on their machine.
    # 20.0: This is the frames per second (FPS) rate at which the video is recorded. 
    # (640, 480): This tuple specifies the resolution of the video, i.e., width and height in pixels.
    out = cv2.VideoWriter(recording_filename, -1, 20.0, (640, 480))

def stop_recording():
    """Stops the current video recording."""
    global out, recording_started
    if out is not None:
        out.release()
    recording_started = None

def draw_all_landmarks(image, landmarks):
    """Draws facial landmarks on the image."""
    if display_enabled:
        draw = ImageDraw.Draw(image)
        purple_color = (128, 0, 128)  # Purple color in RGB
        for landmark in landmarks:
            x = int(landmark.x * image.width)
            y = int(landmark.y * image.height)
            # draw.ellipse() is a method from the PIL library used to draw an ellipse shape.
            # It takes several parameters:

            # [(x - 1, y - 1), (x + 1, y + 1)]: These are coordinates for the bounding box of the ellipse.
            #    - (x - 1, y - 1) is the top left corner of the bounding box.
            #    - (x + 1, y + 1) is the bottom right corner of the bounding box.
            #    - Here, 'x' and 'y' are the coordinates of the landmark. The ellipse is drawn with a width and height of 2 pixels (hence -1 and +1).

            # fill=purple_color: This sets the fill color of the ellipse. In this case, it's set to a purple color defined earlier.

            # outline=purple_color: This sets the color of the ellipse's outline. It's also set to the same purple color.
            draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], fill=purple_color, outline=purple_color)

def printStatus(img):
    """Prints 'Danger' or 'Safe' on the image based on the face_covered state."""
    if display_enabled and body_exist:
        if face_covered:
            text, text_color = ("Danger", (0, 0, 255))
        else: 
            text, text_color = ("Safe", (0, 255, 0))
        # Convert the image from RGB to BGR color space.
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (10, frame.shape[0] - 10)
        font_scale, font_thickness = 1, 2
        # cv2.putText is used to add text to an image in OpenCV.
        # It takes several parameters:

        # frame: The image (numpy array) on which you want to draw the text.
        # text: The actual text string that will be drawn on the image.
        # bottom_left_corner: The position where the text should start. This is the bottom-left corner of the text in pixels.
        # font: The font type for the text. cv2.FONT_HERSHEY_SIMPLEX is a classic, simple font.
        # font_scale: The scale factor that is multiplied by the base font size.
        # text_color: The color of the text, defined in BGR (Blue, Green, Red) format.
        # font_thickness: The thickness of the lines used to draw the text.
        cv2.putText(frame, text, bottom_left_corner, font, font_scale, text_color, font_thickness)
    else:
        # Convert the image from RGB to BGR color space.
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame

def mask_region_rect(image, landmarks, region_indices):
    """
    Creates a mask over a specified facial region based on given landmarks.

    Parameters:
    image (PIL.Image): The image on which the landmarks are detected.
    landmarks (list): A list of landmarks detected on the face.
    region_indices (list): Indices of the landmarks that define the region to be masked.

    Returns:
    PIL.Image: A mask image where the specified region is white (255) and the rest is black (0).
    """

    # Extract the x and y coordinates of the specified landmarks
    xs = [landmarks[idx].x for idx in region_indices if idx < len(landmarks)]
    ys = [landmarks[idx].y for idx in region_indices if idx < len(landmarks)]

    # If no coordinates are found, return None
    if not xs or not ys:
        return None

    # Create a new blank image for the mask with the same dimensions as the original image
    # 'L' mode means it's an 8-bit grayscale image
    mask = Image.new('L', (image.width, image.height), 0)

    # Create a drawing context for the mask
    draw = ImageDraw.Draw(mask)

    # Draw a rectangle on the mask image using the landmark coordinates
    # The rectangle covers the region defined by the specified landmarks
    draw.rectangle([int(min(xs) * image.width), int(min(ys) * image.height), 
                    int(max(xs) * image.width), int(max(ys) * image.height)], fill=255)

    # Return the mask image
    return mask

def calculate_skin_ratio(image, mask):
    """
    Calculates the skin presence ratio in a masked region of an image.

    Parameters:
    image (PIL.Image): The image in which the skin presence needs to be calculated.
    mask (PIL.Image): A binary mask where the region of interest is white (255) and the rest is black (0).

    Returns:
    float: The ratio of skin-colored pixels to the total number of pixels in the masked region.
    """
    
    # Return 0 if mask is None, indicating no region to calculate ratio on.
    if mask is None:
        return 0

    # Convert the mask from a PIL image to a NumPy array for processing with OpenCV.
    mask_np = np.array(mask)

    # Convert the image from PIL's RGB format to OpenCV's BGR format.
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply the mask to the image. This operation performs a bitwise 'and' between
    # the image and itself, but only in the areas where the mask is white.
    region = cv2.bitwise_and(cv_image, cv_image, mask=mask_np)

    # Convert the masked region from BGR to YCrCb color space.
    # YCrCb is often used in skin detection as it separates luminance from chrominance,
    # making it less sensitive to lighting changes.
    ycrcb_image = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)

    # Define the minimum and maximum YCrCb values for skin color.
    min_YCrCb, max_YCrCb = np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8)

    # Create a binary mask where skin-colored pixels are white and others are black.
    skin_mask = cv2.inRange(ycrcb_image, min_YCrCb, max_YCrCb)

    # Calculate the ratio of skin pixels to total pixels in the region of interest.
    # cv2.countNonZero counts the number of non-zero (white) pixels in the skin_mask.
    # cv2.countNonZero on mask_np counts the number of pixels in the region of interest.
    if cv2.countNonZero(mask_np) > 0:
        skin_ratio = cv2.countNonZero(skin_mask) / cv2.countNonZero(mask_np)
    else:
        skin_ratio = 0
    return skin_ratio

# Indices for the mouth and nose landmarks
mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0]
nose_indices = [2, 98, 327, 331, 297]

def is_there_face(detections):
    """Checks if there are any faces detected."""
    return len(detections) > 0

def generate():
    """Generator function for video streaming."""
    global covered_state, face_covered, last_covered_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if recording_started is None or (datetime.datetime.now() - recording_started).seconds >= recorded_video_duration:
            stop_recording()
            start_recording()

        with lock:
            out.write(frame)
            
        # Convert the image from BGR to RGB color space.
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = face_detection_model(img)
        
        # Convert the image from RGB to BGR color space.
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpeg', frame)
        if not ret:
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Flask routes for different functionalities
@app.route('/video_feed')
def video_feed():
    """Route for video feed."""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def get_recordings_list():
    """Retrieves the list of recording files."""
    return os.listdir(recordings_folder)

@app.route("/play_video/<filename>")
def play_video(filename):
    """Route to play a specific video."""
    return render_template('play_video.html', filename=filename)

@app.route('/stream_and_recordings')
def stream_and_recordings():
    """Route for stream and recordings page."""
    recordings = get_recordings_list()
    return render_template('stream_and_recordings.html', recordings=recordings)

@app.route('/api/stream_and_recordings')
def api_stream_and_recordings():
    """API route to get list of recordings."""
    recordings = get_recordings_list()
    return jsonify(recordings)

@app.route('/api/delete_all_recordings', methods=['DELETE'])
def delete_all_recordings():
    """API route to delete all recordings."""
    try:
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

@app.route('/api/set-display-enabled', methods=['POST'])
def set_boolean():
    global display_enabled
    data = request.json
    display_enabled = data.get('value', False)
    return jsonify({'success': True}), 200


if __name__ == '__main__':
    webbrowser.open('http://localhost:5000/stream_and_recordings')  # Open the page in the default browser
    app.run(host='0.0.0.0', port=5000, debug=True)
    