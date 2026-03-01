from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
import base64
import numpy as np
import cv2
import tempfile
import os

from utils import extract_landmarks
from predictor import predict_category
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change to your Vercel URL later for tighter security)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/api/predict/{category}")
async def predict(category: str, image: UploadFile = File(...)):
    try:
        # Read the image bytes
        contents = await image.read()
        
        # Convert to numpy array and decode to OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Failed to decode image"}, status_code=400)

        # Unconditionally mirror the physical camera feed to match training data
        frame = cv2.flip(frame, 1)

        landmarks, _ = extract_landmarks(frame)

        if landmarks is None:
            return {"prediction": "Waiting for input..."}
        
        print(f"[{category}] Landmarks sample: {landmarks[:5]}")
        
        prediction = predict_category(landmarks, category)
        print(f"[{category}] Final prediction: {prediction}")

        return {"prediction": prediction}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

# Mount the current directory strictly for static files
# since our HTML accesses styles.css and converse.html directly.
app.mount("/", StaticFiles(directory=".", html=True), name="static")

