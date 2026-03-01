import cv2
from utils import extract_landmarks

# Read the image
frame = cv2.imread("hands1.png")
if frame is None:
    print("Could not load hands1.png")
else:
    landmarks, _ = extract_landmarks(frame)
    if landmarks is not None:
        print("Landmarks shape:", landmarks.shape)
        print("First 15 features:", landmarks[:15])
        print("Max feature:", landmarks.max())
        print("Min feature:", landmarks.min())
        print("Sum of features:", landmarks.sum())
        from predictor import predict_category
        print("\nPredictions with Wrist-Relative Coordinates:")
        print(" Alphabets:", predict_category(landmarks, "alphabets"))
        print(" Numbers:", predict_category(landmarks, "numbers"))
        print(" Basic7:", predict_category(landmarks, "basic7"))
    else:
        print("No hands detected in hands1.png")
