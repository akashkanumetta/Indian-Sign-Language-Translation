import requests
import sys

# Test with a real image of hands
img_path = "hands2.png" 

# Categories to test
categories = ["alphabets", "basic7", "numbers", "school", "public_places", "workplaces"]

try:
    with open(img_path, "rb") as f:
        img_bytes = f.read()
except FileNotFoundError:
    print(f"Error: {img_path} not found.")
    sys.exit(1)

for cat in categories:
    url = f"http://127.0.0.1:8000/api/predict/{cat}"
    files = {"image": ("test.png", img_bytes, "image/png")}
    
    try:
        response = requests.post(url, files=files)
        print(f"[{cat}] Status: {response.status_code}, Response: {response.json()}")
    except Exception as e:
        print(f"[{cat}] Error: {e}")
