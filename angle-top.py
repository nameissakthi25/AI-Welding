import cv2
from ultralytics import YOLO
import math
import numpy as np

# Define the output video file path and settings
output_video_path = "output_tracking.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_rate = 30  # Adjust the desired frame rate
output_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (640, 480))  # Adjust the frame size as needed

# Load the YOLOv8 model
model = YOLO(r'Bottle\bottle-100\RTX-100.pt')

# Open the video file
cap = cv2.VideoCapture(0)

# Initialize variables to store the midpoints of the bounding boxes
cap_midpoint = None
plastic_bottle_midpoint = None

# Create a named window for displaying the video
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Reset midpoints
        cap_midpoint = None
        plastic_bottle_midpoint = None

        # Loop through the detected objects in the first class (results[0])
        if len(results[0].boxes.cls) > 1:
            if results[0].boxes.cls[0] == 0:
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy.cpu()[0])
                x3, y3, x4, y4 = map(int, results[0].boxes.xyxy.cpu()[1])
            elif results[0].boxes.cls[0] == 1:
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy.cpu()[1])
                x3, y3, x4, y4 = map(int, results[0].boxes.xyxy.cpu()[0])

            # Calculate the midpoint of the bounding box
            cap_midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            plastic_bottle_midpoint = ((x3 + x4) // 2, (y3 + y4) // 2)
            cv2.circle(frame, cap_midpoint, 10, (0, 0, 255), -1)
            cv2.circle(frame, plastic_bottle_midpoint, 10, (0, 0, 255), -1)

            # Calculate the angle between cap midpoint and plastic bottle midpoint
            delta_x = plastic_bottle_midpoint[0] - cap_midpoint[0]
            delta_y = plastic_bottle_midpoint[1] - cap_midpoint[1]
            angle = math.degrees(math.atan2(delta_y, delta_x))

            # Display the angle in the top-left corner of the window
            cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the output video
        output_video.write(frame)

        cv2.imshow("YOLOv8 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, close the display window, and release the output video writer
cap.release()
output_video.release()
cv2.destroyAllWindows()
