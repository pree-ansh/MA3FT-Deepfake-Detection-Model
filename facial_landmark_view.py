import dlib
import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_facial_alignment(image_path, display=False):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the face detector and shape predictor
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Detect faces in the image
    faces = face_detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Predict facial landmarks
        landmarks = landmark_predictor(gray, face)

        # Convert landmarks to NumPy array
        landmarks_np = np.array([(landmark.x, landmark.y) for landmark in landmarks.parts()])

        # Draw landmarks on the face (optional)
        if display:
            image_copy = image.copy()
            for (x, y) in landmarks_np:
                cv2.circle(image_copy, (x, y), 2, (0, 255, 0), -1)
            plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
            plt.title('Facial Landmarks')
            plt.axis('off')
            plt.show(block=True)

        # Check for unnatural or inconsistent facial landmarks (custom logic)
        # Add your logic here to analyze the landmarks for alignment issues
        # For example, check distances between specific landmarks

    return landmarks_np if faces else None


image_path = 'sample.png'

# Detect facial alignment and optionally display landmarks
landmarks = detect_facial_alignment(image_path, display=True)

# Use the 'landmarks' variable for further analysis if needed
pass