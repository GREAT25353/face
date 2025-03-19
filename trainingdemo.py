import cv2
import numpy as np
from PIL import Image
import os

# Ensure you have the contrib version installed
recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "datasets"

def getImageId(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L')
        facenp = np.array(faceImage)

        # Extract ID safely
        filename = os.path.basename(imagePaths)
        Id = int(filename.split(".")[1])  # Extract ID from filename

        faces.append(facenp)
        ids.append(Id)

        # Display image while training
        cv2.imshow("Training", facenp)
        cv2.waitKey(1)

    return ids, faces

# Get IDs and faces properly
Ids, facedata = getImageId(path)

# Train and save
recognizer.train(facedata, np.array(Ids))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed.........................")
