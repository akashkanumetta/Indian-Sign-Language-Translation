import numpy as np
from predictor import get_model

model, scaler, labels = get_model("numbers")
features = np.random.rand(63)
inp = scaler.transform(features.reshape(1, -1))

with open("scaler_out.txt", "w", encoding="utf-8") as f:
    f.write(f"Input features (first 5): {features[:5]}\n")
    f.write(f"Scaled features (first 5): {inp[0, :5]}\n")
    f.write(f"Scaled min: {np.min(inp)} max: {np.max(inp)}\n")
    f.write(f"Are there NaNs? {np.isnan(inp).any()}\n")
    f.write(f"Scaler mean: {scaler.mean_[:5] if hasattr(scaler, 'mean_') else 'N/A'}\n")
    f.write(f"Scaler scale: {scaler.scale_[:5] if hasattr(scaler, 'scale_') else 'N/A'}\n")
    
    pred = model.predict(inp, verbose=0)
    f.write(f"Prediction distribution: {pred}\n")
    f.write(f"Decoded: {labels.inverse_transform([np.argmax(pred)])[0]}\n")
