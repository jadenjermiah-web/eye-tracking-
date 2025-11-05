from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Gaze data storage
gaze_history = deque(maxlen=50)
current_gaze = {"x": 0, "y": 0, "direction": "center"}

def get_eye_region(landmarks, eye_points, frame_shape):
    """Extract eye region coordinates"""
    points = np.array([(landmarks[p].x * frame_shape[1], 
                       landmarks[p].y * frame_shape[0]) for p in eye_points], dtype=np.int32)
    return points

def get_iris_center(landmarks, iris_points, frame_shape):
    """Calculate iris center"""
    iris_points_coords = [(landmarks[p].x * frame_shape[1], 
                          landmarks[p].y * frame_shape[0]) for p in iris_points]
    center_x = sum(p[0] for p in iris_points_coords) / len(iris_points_coords)
    center_y = sum(p[1] for p in iris_points_coords) / len(iris_points_coords)
    return int(center_x), int(center_y)

def estimate_gaze(left_iris, right_iris, left_eye, right_eye):
    """Estimate gaze direction based on iris position relative to eye"""
    # Calculate relative position of iris in eye
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    
    left_offset = (left_iris[0] - left_eye_center[0], left_iris[1] - left_eye_center[1])
    right_offset = (right_iris[0] - right_eye_center[0], right_iris[1] - right_eye_center[1])
    
    # Average offset
    avg_x = (left_offset[0] + right_offset[0]) / 2
    avg_y = (left_offset[1] + right_offset[1]) / 2
    
    # Normalize to -1 to 1 range (approximate)
    eye_width = np.linalg.norm(left_eye[3] - left_eye[0])
    norm_x = avg_x / (eye_width * 0.3) if eye_width > 0 else 0
    norm_y = avg_y / (eye_width * 0.3) if eye_width > 0 else 0
    
    # Clamp values
    norm_x = max(-1, min(1, norm_x))
    norm_y = max(-1, min(1, norm_y))
    
    # Determine direction
    direction = "center"
    if abs(norm_x) > 0.3 or abs(norm_y) > 0.3:
        if abs(norm_x) > abs(norm_y):
            direction = "right" if norm_x > 0 else "left"
        else:
            direction = "down" if norm_y > 0 else "up"
    
    return norm_x, norm_y, direction

def generate_frames():
    """Generate video frames with eye tracking"""
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # Get eye regions
            left_eye = get_eye_region(landmarks, LEFT_EYE, frame.shape)
            right_eye = get_eye_region(landmarks, RIGHT_EYE, frame.shape)
            
            # Get iris centers
            left_iris = get_iris_center(landmarks, LEFT_IRIS, frame.shape)
            right_iris = get_iris_center(landmarks, RIGHT_IRIS, frame.shape)
            
            # Draw eye contours
            cv2.polylines(frame, [left_eye], True, (100, 100, 100), 1)
            cv2.polylines(frame, [right_eye], True, (100, 100, 100), 1)
            
            # Draw iris centers
            cv2.circle(frame, left_iris, 3, (50, 50, 50), -1)
            cv2.circle(frame, right_iris, 3, (50, 50, 50), -1)
            
            # Estimate gaze
            gaze_x, gaze_y, direction = estimate_gaze(
                left_iris, right_iris, left_eye, right_eye
            )
            
            # Update global gaze data
            global current_gaze
            current_gaze = {
                "x": float(gaze_x),
                "y": float(gaze_y),
                "direction": direction,
                "timestamp": time.time()
            }
            gaze_history.append((gaze_x, gaze_y))
            
            # Draw gaze indicator
            center_x, center_y = w // 2, 50
            gaze_point_x = int(center_x + gaze_x * 100)
            gaze_point_y = int(center_y + gaze_y * 50)
            cv2.circle(frame, (gaze_point_x, gaze_point_y), 8, (80, 80, 80), -1)
            cv2.circle(frame, (center_x, center_y), 120, (200, 200, 200), 2)
            
            # Add text
            cv2.putText(frame, f"Gaze: {direction.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gaze_data')
def gaze_data():
    """Get current gaze data"""
    return jsonify(current_gaze)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)