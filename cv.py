import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('best.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Run inference loop
while True:
    # Read frame from the camera
    success, frame = cap.read()

    if success:
        # Run YOLOv5 inference on the frame
        results = model(frame)

        # Check each Results object for detections
        for result in results:
            # Check if there are any detections
            if len(result) > 0:
                # Visualize the results on the frame
                annotated_frame = result.show()

                # Display the annotated frame
                cv2.imshow('YOLOv5 Inference', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if reading the frame fails
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
