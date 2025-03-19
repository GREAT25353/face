import cv2

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter Your ID:")  # Provide ID for dataset
count = 0  # Initialize the count variable

if not video.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Correct color conversion
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f'datasets/User.{id}.{count}.jpg', gray[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if count > 500:  # Collect 500 images
        break

video.release()
cv2.destroyAllWindows()
print("Dataset collection done..................")
