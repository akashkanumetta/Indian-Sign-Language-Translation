import cv2, numpy as np, sys, joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

base_options = mp_python.BaseOptions(model_asset_path='d:/signlink-isl/signlink/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

import predictor

def get_landmarks(frame, wrist_relative=False):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks: return None
    landmarks = []
    
    # Pad to 2 hands first
    hl_list = result.hand_landmarks[:2]
    
    for hl in hl_list:
        wx, wy, wz = hl[0].x, hl[0].y, hl[0].z
        for lm in hl:
            if wrist_relative:
                landmarks.extend([lm.x - wx, lm.y - wy, lm.z - wz])
            else:
                landmarks.extend([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks)
    if len(landmarks) < 126:
        landmarks = np.concatenate([landmarks, np.zeros(126 - len(landmarks))])
    else:
        landmarks = landmarks[:126]
    return landmarks

img = cv2.imread('d:/signlink-isl/signlink/hands2.png')

print('--- PERMUTATIONS EVALUATION on hands2.png ---')
for flip in [False, True]:
    f_img = cv2.flip(img, 1) if flip else img
    for wrist in [False, True]:
        for scale in [False, True]:
            lms = get_landmarks(f_img, wrist_relative=wrist)
            if lms is None: continue
            
            # Predict
            # 1. apply scale inv if requested
            f126 = np.copy(lms)
            f63 = np.copy(lms[:63])
            
            if scale:
                m126 = np.max(np.abs(f126)); f126 = f126 / m126 if m126 > 0 else f126
                m63 = np.max(np.abs(f63)); f63 = f63 / m63 if m63 > 0 else f63
            
            print(f"Flip={flip}, Wrist={wrist}, Scale={scale}")
            
            # alphabets
            pred_alph = predictor.predict_category(f126, 'alphabets')
            
            # numbers
            pred_num = predictor.predict_category(f63, 'numbers')
            
            print(f"  Alphabets -> {pred_alph} | Numbers -> {pred_num}")
