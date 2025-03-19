import cv2
import time

# Initialize the camera and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# List of names corresponding to IDs
name_list = ["", "Akash"]  # ID 1 = Akash

# Check if camera opens successfully
if not video.isOpened():
    print("Error: Could not open camera.")
    exit()

# Frame rate calculation
prev_time = time.time()

while True:
    ret, frame = video.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Frame rate calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y + h, x:x + w])

        # Confidence threshold
        if conf < 50 and 0 <= serial < len(name_list):
            name = name_list[serial]
            label = f"{name} (Conf: {round(conf, 2)})"
            color = (0, 255, 0)  # Green for recognized face
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red for unknown face

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw label background
        label_bg_color = (0, 0, 255) if label == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), label_bg_color, -1)

        # Display the label
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Add a title bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (255, 94, 77), -1)
    cv2.putText(frame, "FACE RECOGNITION", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Display FPS in the corner
    cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# Release the camera and close windows
video.release()
cv2.destroyAllWindows()
print("Face Recognition Completed..................")
