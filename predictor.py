import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# ==============================
# BASE MODEL DIRECTORY (pointing to local models folder)
# ==============================
BASE_DIR = Path("models")

# Cache loaded models to avoid reloading
_models_cache = {}

def load_bundle(folder_path):
    model = tf.keras.models.load_model(
        folder_path / "model.h5",
        compile=False   # 🚨 Prevent optimizer errors
    )

    scaler = joblib.load(folder_path / "scaler.pkl")
    labels = joblib.load(folder_path / "labels.pkl")

    model.trainable = False

    return model, scaler, labels

def get_model(category):
    if category in _models_cache:
        return _models_cache[category]
    
    folder_path = BASE_DIR / category
    if not folder_path.exists():
        raise ValueError(f"Model for category '{category}' not found at {folder_path}")

    model, scaler, labels = load_bundle(folder_path)
    _models_cache[category] = (model, scaler, labels)
    return _models_cache[category]

def to_126(features):
    features = np.array(features)
    if len(features) == 63:
        return np.concatenate([features, np.zeros(63)])
    return features

def decode(pred, label_encoder):
    return label_encoder.inverse_transform([np.argmax(pred)])[0]

def predict_category(features, category):
    model, scaler, labels = get_model(category)

    features = np.array(features)

    if category == "numbers":
        features = features[:63]
    else:
        features = to_126(features)

    inp = scaler.transform(features.reshape(1, -1))
    pred = model.predict(inp, verbose=0)

    return decode(pred, labels)
