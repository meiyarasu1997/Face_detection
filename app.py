from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import os
import numpy as np

app = Flask(__name__)

# Function to get the encoding of an image
def get_face_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        return face_encoding
    except:
        return "face not found"

# Directory containing images of known individuals
known_faces_dir = "known_faces"

# Dictionary to store known face encodings and corresponding names
known_face_encodings = {}
known_face_names = []

# Populate known face encodings
for file_name in os.listdir(known_faces_dir):
    name = os.path.splitext(file_name)[0]
    file_path = os.path.join(known_faces_dir, file_name)
    face_encoding = get_face_encoding(file_path)
    known_face_encodings[name] = face_encoding
    known_face_names.append(name)

# Function to perform face recognition
# def recognize_face(frame):
#     # Find all face locations and face encodings in the current frame
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)

#     # Loop through each face found in the frame
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Check if the face matches any known faces
#         matches = face_recognition.compare_faces(known_face_encodings.values(), face_encoding)
#         name = "Unknown"

#         # Use the name of the first known face found
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]

#         # Draw a rectangle and label on the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

#     return frame

def recognize_face(frame):
    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Convert known_face_encodings values to a NumPy array
    known_face_encodings_array = np.array(list(known_face_encodings.values()))

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings_array, face_encoding)
        name = "Unknown"

        # Use the name of the first known face found
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle and label on the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    return frame
# Function to capture video frames
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            break
        else:
            frame = recognize_face(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
