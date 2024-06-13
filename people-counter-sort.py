import os
import sys
import cv2
import numpy as np
import requests

# Tambahkan path ke direktori 'sort'
sys.path.append('C:/Program Files/Python312/model_ssd/ssd_env/Lib/site-packages/sort')

# Import kelas Sort dari modul sort
from sort import Sort

# Load SSD model and config file
model_path = "ssd_model/mobilenet_iter_73000.caffemodel"
config_path = "ssd_model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Open your laptop's camera
cap = cv2.VideoCapture(1)  # Use '0' for built-in webcam, '1' for external webcam
assert cap.isOpened(), "Error accessing webcam"

# Initialize variables
people_in = 0
people_out = 0
last_sent_people_in = 0
last_sent_people_out = 0

# Initialize SORT tracker
tracker = Sort()

# Function to send data to ESP32
def send_to_esp32(in_count, out_count, total_count):
    url = 'http://192.168.1.38/update_counts'  # Replace with the IP address of your ESP32
    payload = {
        'in': in_count,
        'out': out_count,
        'total': total_count
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f'Status Code: {response.status_code}, Response: {response.text}')
    except requests.exceptions.RequestException as e:
        print(f'Error sending data to ESP32: {e}')

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    line_position = height // 2  # Horizontal line in the middle of the frame

    # Prepare the frame for SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    dets = []

    # Iterate over each detected object
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Only consider "person" class
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                # Ensure the bounding box fits within the frame dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                # Append the detection to the list
                dets.append([x1, y1, x2, y2, confidence])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label and confidence score
                cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print dets structure
    print("Detected objects:", dets)

    # Update tracker with detections
    dets = np.array(dets)
    tracks = tracker.update(dets)

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Draw tracking ID
        cv2.putText(frame, f'ID {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check if the object crosses the line
        if (y1 < line_position < y2) or (y2 < line_position < y1):
            if y1 < line_position < y2:
                people_in += 1
            elif y2 < line_position < y1:
                people_out += 1

    # Draw the horizontal line
    cv2.line(frame, (0, line_position), (width, line_position), (0, 0, 255), 2)

    # Show counts
    total_people = people_in - people_out
    cv2.putText(frame, f'IN: {people_in}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'OUT: {people_out}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Total: {total_people}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Send data to ESP32 only if there's a change in people_in or people_out
    if people_in != last_sent_people_in or people_out != last_sent_people_out:
        send_to_esp32(people_in, people_out, total_people)
        last_sent_people_in = people_in
        last_sent_people_out = people_out

    # Show the frame with detections and counts
    cv2.imshow('SSD Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
