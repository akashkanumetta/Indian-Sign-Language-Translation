import cv2
import numpy as np
from utils import extract_landmarks
from predictor import predict_category

img = cv2.imread("hands1.png")
flipped = cv2.flip(img, 1)

l_orig, _ = extract_landmarks(img)
l_flip, _ = extract_landmarks(flipped)

print("Original Image Predictions:")
print(" Basic7:", predict_category(l_orig, "basic7"))
print(" Numbers:", predict_category(l_orig, "numbers"))
print(" Alphabets:", predict_category(l_orig, "alphabets"))

print("\nFlipped Image Predictions:")
print(" Basic7:", predict_category(l_flip, "basic7"))
print(" Numbers:", predict_category(l_flip, "numbers"))
print(" Alphabets:", predict_category(l_flip, "alphabets"))
