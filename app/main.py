# app/main.py
import face_recognition
import cv2
import time
import datetime
import os
import glob
import numpy as np
from datetime import timezone, timedelta

import dlib
print("DLIB_USE_CUDA:", dlib.DLIB_USE_CUDA)

# Импортируем настройки из конфига
from app.config import get_settings
settings = get_settings()

# Initialize variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Configure paths and parameters
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PHOTOS_DIR = os.path.join(PROJECT_ROOT, settings.PHOTOS_DIR) 

# Load settings from config
CAMERA_INDEX = settings.CAMERA_INDEX
CAMERA_WIDTH = settings.CAMERA_WIDTH
CAMERA_HEIGHT = settings.CAMERA_HEIGHT
RESIZE_FACTOR = settings.RESIZE_FACTOR

FACE_RECOGNITION_MODEL = settings.FACE_RECOGNITION_MODEL
FACE_DISTANCE_THRESHOLD = settings.FACE_DISTANCE_THRESHOLD

# Parse font
FONT = getattr(cv2, settings.FONT)

# Get colors from config
COLOR_KNOWN = settings.color_known
COLOR_UNKNOWN = settings.color_unknown
COLOR_INFO = settings.color_info

# Localization
UNKNOWN_PERSON_LABEL = settings.UNKNOWN_PERSON_LABEL
TIMEZONE_OFFSET = settings.TIMEZONE_OFFSET
TIMEZONE_LABEL = settings.TIMEZONE_LABEL

# Define timezone
CUSTOM_TIMEZONE = timezone(timedelta(hours=TIMEZONE_OFFSET))

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    # Check if photos directory exists
    if not os.path.isdir(PHOTOS_DIR):
        raise FileNotFoundError(f"Photos directory '{PHOTOS_DIR}' not found")
    
    # Find all JPEG files in photos directory
    image_paths = glob.glob(os.path.join(PHOTOS_DIR, '*.jpg')) + glob.glob(os.path.join(PHOTOS_DIR, '*.jpeg'))
    if not image_paths:
        raise ValueError(f"No JPEG images found in {PHOTOS_DIR}")
    
    print(f"Loading {len(image_paths)} known faces...")
    
    # Load each image and create encodings
    for image_path in image_paths:
        # Extract base name without extension
        name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Get face encodings
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            # Add first found face encoding
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            print(f"Loaded face: {name}")
        else:
            print(f"Warning: No face found in {image_path}")
    
    return known_face_encodings, known_face_names

def capture_and_process():
    global prev_frame_time
    
    # Initialize video capture
    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    
    # Set camera properties for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    process_this_frame = True
    
    try:
        known_face_encodings, known_face_names = load_known_faces()
        print(f"Successfully loaded {len(known_face_encodings)} faces")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Mirror and resize frame
        frame = cv2.flip(frame, 1)
        
        # Process every other frame to save resources
        if process_this_frame:
            # Resize and convert to RGB
            small_frame = cv2.resize(frame, (0, 0), fx=1/RESIZE_FACTOR, fy=1/RESIZE_FACTOR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces and their encodings
            face_locations = face_recognition.face_locations(rgb_small_frame, model=FACE_RECOGNITION_MODEL)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Recognize faces
            face_names = []
            for face_encoding in face_encodings:
                # Use face_distance to get best match instead of first match
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    # Use a threshold to determine if the match is good enough
                    if face_distances[best_match_index] < FACE_DISTANCE_THRESHOLD:
                        name = known_face_names[best_match_index]
                    else:
                        name = UNKNOWN_PERSON_LABEL
                else:
                    name = UNKNOWN_PERSON_LABEL
                
                face_names.append(name)
        
        process_this_frame = not process_this_frame
        
        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale coordinates back to original size
            top *= RESIZE_FACTOR
            right *= RESIZE_FACTOR
            bottom *= RESIZE_FACTOR
            left *= RESIZE_FACTOR
            
            # Choose colors based on recognition status
            color = COLOR_KNOWN if name != UNKNOWN_PERSON_LABEL else COLOR_UNKNOWN
            text_color = (255, 255, 255)  # White text
            
            # Draw rectangle around face
            cv2.rectangle(
                frame, 
                (left, top), 
                (right, bottom), 
                color, 
                2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (left, bottom - 50),
                (right, bottom),
                color, cv2.FILLED
            )
            
            # Draw name label
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 16),
                FONT,
                1.0,
                text_color,
                2
            )
        
        # Calculate and display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (40, 50),
            FONT,
            1,
            COLOR_INFO,
            2
        )
                
        # Display current time with configured timezone
        current_time_utc = datetime.datetime.now(timezone.utc)
        current_time_local = current_time_utc.astimezone(CUSTOM_TIMEZONE)
        time_str = current_time_local.strftime("%H:%M:%S")
                
        cv2.putText(
            frame,
            f"{TIMEZONE_LABEL}: {time_str}",
            (40, 100),
            FONT,
            1,
            COLOR_INFO,
            2
        )
                
        # Show frame
        cv2.imshow("Video", frame)
        
        # Exit on ESC or window close
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_process()
