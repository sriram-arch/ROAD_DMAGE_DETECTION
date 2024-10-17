import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Load the YOLO model with custom-trained weights for crack detection
model = YOLO('best.pt')

# Open a file dialog to allow the user to choose an image
def ask_image_from_user():
    root = Tk()
    root.withdraw()  # Hide the Tkinter root window
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]  # Allow only image files
    )
    return image_path

# Function to perform YOLO inference on the uploaded image
def detect_crack_on_image():
    while True:
        image_path = ask_image_from_user()  # Ask user to upload an image
        
        if image_path:
            # Read the selected image
            image = cv2.imread(image_path)
            
            # Check if the image was successfully loaded
            if image is None:
                print("Error: Could not read the image.")
                return

            # Run YOLO inference on the image
            results = model(image)

            # Visualize the results on the image
            annotated_image = results[0].plot()

            # Display detection results: bounding box with crack type
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])  # Get class index (integer)
                    confidence = float(box.conf[0])  # Get confidence score
                    crack_type = model.names[class_id]  # Get the crack type label
                    print(f"Detected crack type: {crack_type}, Confidence: {confidence:.2f}")

            # Display the annotated image with detection results
            cv2.imshow("YOLOv8 Crack Detection", annotated_image)

            # Wait for the 'q' key to close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Ask the user if they want to check another file or exit
        choice = input("Do you want to check another file? (y/n): ")
        if choice.lower() != 'y':
            break

# Function to perform YOLO inference using webcam
def detect_crack_with_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference - Webcam", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Error: Failed to read from camera.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Main menu to let the user choose between webcam or file upload, and ask for choices after detection
def main():
    while True:
        print("\nChoose an option for crack detection:")
        print("1. Use Webcam")
        print("2. Upload an Image File")
        print("3. Exit")
        
        choice = input("Enter 1, 2, or 3: ")

        if choice == '1':
            print("Opening webcam...")
            detect_crack_with_webcam()
            post_detection_choice()
        elif choice == '2':
            print("Please upload an image file.")
            detect_crack_on_image()
            post_detection_choice()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# Function to ask the user what they want to do next after a detection
def post_detection_choice():
    while True:
        print("\nWhat would you like to do next?")
        print("1. Switch to Webcam")
        print("2. Switch to File Upload")
        print("3. Exit")

        next_choice = input("Enter 1, 2, or 3: ")

        if next_choice == '1':
            detect_crack_with_webcam()
        elif next_choice == '2':
            detect_crack_on_image()
        elif next_choice == '3':
            print("Exiting the program.")
            exit()
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# Run the main function to start the program
if __name__ == "__main__":
    main()
