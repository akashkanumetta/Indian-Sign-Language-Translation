import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================
# MODEL CONFIGURATION
# ==============================
model_path = 'hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize global detector
detector = vision.HandLandmarker.create_from_options(options)


# ==============================
# FRAME LANDMARK EXTRACTION
# ==============================
def extract_landmarks(frame):
    # Convert BGR → RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Run detection
    detection_result = detector.detect(mp_image)

    annotated_frame = frame.copy()

    if detection_result.hand_landmarks:
        landmarks = []

        for hand_landmarks in detection_result.hand_landmarks[:2]:
            wx, wy, wz = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
            for lm in hand_landmarks:
                # WRIST-RELATIVE COORDS — The training models all used `0.0` for wrists!
                landmarks.extend([lm.x - wx, lm.y - wy, lm.z - wz])

                # Draw landmark dots on the annotated frame
                h, w, _ = annotated_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (cx, cy), 2, (0, 255, 0), cv2.FILLED)

        landmarks = np.array(landmarks)

        # Force exactly 126 features (2 hands × 21 landmarks × 3 coords)
        if len(landmarks) < 126:
            landmarks = np.concatenate([landmarks, np.zeros(126 - len(landmarks))])
        elif len(landmarks) > 126:
            landmarks = landmarks[:126]

        return landmarks, annotated_frame

    return None, annotated_frame
