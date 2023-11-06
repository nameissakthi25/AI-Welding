import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r'Bottle\bottle-100\b-cap-200-best.pt')


cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        for det in results[0].boxes.xyxy:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = det[:4]

            # Calculate the center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw a point at the center
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red point

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
