import sqlite3
import time
import cv2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO
import os

# Initialize Flask app and SocketIO
app = Flask(__name__, template_folder='../templates')
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize YOLO model
model_path = r'C:\Users\Purna\OneDrive\PP\ultralytics\yolov8n.pt'  # Path to YOLOv8 model
model = YOLO(model_path)

# Database creation and connection
def create_db():
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY,
        timestamp REAL,
        object_name TEXT,
        x_min INTEGER,
        y_min INTEGER,
        x_max INTEGER,
        y_max INTEGER,
        position TEXT
    )
    ''')
    conn.commit()
    conn.close()

def insert_detection(object_name, timestamp, x_min, y_min, x_max, y_max, position):
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO detections (timestamp, object_name, x_min, y_min, x_max, y_max, position)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, object_name, x_min, y_min, x_max, y_max, position))
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Add authentication logic if required
        return redirect(url_for('classroom'))
    return render_template('login.html')

@app.route('/classroom', methods=['GET', 'POST'])
def classroom():
    if request.method == 'POST':
        return jsonify({"message": "Video stream started."})
    return render_template('classroom.html')

# Video Streaming Function
def generate_video_stream():
    video_path = r'C:\Users\Purna\Downloads\v1.mp4'  # Replace with your video path
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Run YOLOv8 detection
        results = model.predict(frame, save=False, conf=0.5)
        for result in results:
            frame = result.plot()
            for detection in result.boxes:
                class_id = int(detection.cls)
                class_name = model.names[class_id]
                if class_name in ["person", "mobile phone"]:
                    x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
                    position = f"({x_min}, {y_min})"
                    timestamp = time.time()
                    insert_detection(class_name, timestamp, x_min, y_min, x_max, y_max, position)

        # Convert frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video', methods=['POST'])
def start_video():
    data = request.get_json()
    room = data.get('classroom')

    if not room:
        return jsonify({"error": "Room number is required"}), 400

    # Log or use the room number for further processing
    print(f"Room selected: {room}")

    # Return success response to frontend
    return jsonify({"success": True})



# Main Entry Point
if __name__ == '__main__':
    create_db()
    socketio.run(app, host='0.0.0.0', port=5000)
