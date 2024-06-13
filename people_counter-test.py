import cv2
import numpy as np
import requests

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
next_person_id = 0
tracked_objects = {}
last_sent_people_in = 0
last_sent_people_out = 0

# Function to check if a person crosses the horizontal line
def is_crossing_line(prev_centroid, current_centroid, line_position):
    return (prev_centroid[1] <= line_position < current_centroid[1]) or (prev_centroid[1] >= line_position > current_centroid[1])

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

    current_centroids = []

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

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label and confidence score
                cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate centroid of the bounding box
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_centroids.append(centroid)

    # Update tracking information
    updated_tracked_objects = {}
    for centroid in current_centroids:
        min_distance = float("inf")
        assigned_id = None

        # Find the closest tracked object
        for person_id, data in tracked_objects.items():
            prev_centroid = data['centroid']
            distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            if distance < min_distance:
                min_distance = distance
                assigned_id = person_id

        if min_distance > 50:  # If no close tracked object is found, assign a new ID
            assigned_id = next_person_id
            next_person_id += 1

        updated_tracked_objects[assigned_id] = {'centroid': centroid, 'counted': tracked_objects.get(assigned_id, {'counted': False})['counted']}

        # Check if the object crosses the line
        if assigned_id in tracked_objects:
            prev_centroid = tracked_objects[assigned_id]['centroid']
            if not tracked_objects[assigned_id]['counted'] and is_crossing_line(prev_centroid, centroid, line_position):
                if prev_centroid[1] < line_position < centroid[1]:
                    people_in += 1
                elif prev_centroid[1] > line_position > centroid[1]:
                    people_out += 1
                updated_tracked_objects[assigned_id]['counted'] = True

    tracked_objects = updated_tracked_objects

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
