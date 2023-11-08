import cv2
from ultralytics import YOLO
import math

# Load the YOLOv8 model
model = YOLO(r'Bottle\bottle-100\b-cap-100-f.pt')

# Open the video file
cap = cv2.VideoCapture(0)

# Initialize variables to store the midpoints of the bounding boxes
cap_midpoint = None
plastic_bottle_midpoint = None

# Create a named window for displaying the video
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

# Loop through the video frames
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
            x1, y1, x2, y2 = map(int, results[0].boxes.xyxy.cpu()[0])
            x3, y3, x4, y4 = map(int, results[0].boxes.xyxy.cpu()[1])
            # Calculate the midpoint of the bounding box
            cap_midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            plastic_bottle_midpoint = ((x3 + x4) // 2, (y3 + y4) // 2)
            cv2.circle(frame, cap_midpoint, 10, (0, 0, 255), -1)
            cv2.circle(frame, plastic_bottle_midpoint, 10, (0, 0, 255), -1)
            
            # Calculate the midpoint between cap and bottle
            horizontal_midpoint = ((cap_midpoint[0] + plastic_bottle_midpoint[0]) // 2, 
                                   (cap_midpoint[1] + plastic_bottle_midpoint[1]) // 2)
            
            # Draw a horizontal line
            cv2.line(frame, cap_midpoint, plastic_bottle_midpoint, (0, 0, 255), 2)
            
            # Calculate the angle between the horizontal line and the X-axis
            delta_x = plastic_bottle_midpoint[0] - cap_midpoint[0]
            delta_y = plastic_bottle_midpoint[1] - cap_midpoint[1]
            angle = math.degrees(math.atan2(delta_y, delta_x))
            
            print(f"Angle between the horizontal line and X-axis: {angle} degrees")

        cv2.imshow("YOLOv8 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
