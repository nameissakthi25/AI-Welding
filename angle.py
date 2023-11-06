import cv2
from ultralytics import YOLO
import math

# Load the YOLOv8 model
model = YOLO(r'Bottle\bottle-100\b-cap-100-f.pt')

cap = cv2.VideoCapture(0)

# Initialize lists to store center points for each class
class1_center_points = []
class2_center_points = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Clear the center point lists for each class in each frame
        class1_center_points = []
        class2_center_points = []

        for det in results[0].boxes.xyxy:
            print(det)
            # Get the class label and bounding box coordinates
            x1, y1, x2, y2 = det[:4]

            # Calculate the center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            label = results[0].names
            print('Label',type(label.keys()))
            if len(label.keys())>1:
                # Class 1
                print('Truee')
                class1_center_points.append((center_x, center_y))
                class2_center_points.append((center_x, center_y))

        # Draw lines between center points if both classes are detected
        if len(class1_center_points) > 0 and len(class2_center_points) > 0:
            for center1 in class1_center_points:
                for center2 in class2_center_points:
                    cv2.line(frame, center1, center2, (0, 0, 255), 2)  # Red line

                    # Calculate the angle between the two points
                    delta_x = center2[0] - center1[0]
                    delta_y = center2[1] - center1[1]
                    angle = math.degrees(math.atan2(delta_y, delta_x))

                    # Print the angle
                    print(f"Angle between the two points: {angle} degrees")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
