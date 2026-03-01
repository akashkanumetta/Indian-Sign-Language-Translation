import requests
import cv2
import numpy as np

# Create a dummy image (black square 640x480)
# This will have NO hands in it, so we expect prediction: "Waiting for input..."
dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

# Encode as JPEG
_, buffer = cv2.imencode('.jpg', dummy_img)

# Categories to test
categories = ["alphabets", "basic7", "numbers", "school", "public_places", "workplaces"]

for cat in categories:
    url = f"http://127.0.0.1:8000/api/predict/{cat}"
    files = {"image": ("dummy.jpg", buffer.tobytes(), "image/jpeg")}
    
    try:
        response = requests.post(url, files=files)
        print(f"[{cat}] Status: {response.status_code}, Response: {response.json()}")
    except Exception as e:
        print(f"[{cat}] Error: {e}")
