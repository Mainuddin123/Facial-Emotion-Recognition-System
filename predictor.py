# ==========================================
# Import Libraries
# ==========================================

import joblib
import numpy as np

from pathlib import Path
from tensorflow.keras.models import load_model

# ==========================================
# Load Model
# ==========================================

MODEL_PATH = Path("models/emotion_model_optuna.keras")
LABEL_ENCODER_PATH = Path("models/label_encoder_optuna.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH.resolve()}"
    )

if not LABEL_ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Label Encoder not found:\n{LABEL_ENCODER_PATH.resolve()}"
    )

MODEL = load_model(MODEL_PATH)

LABEL_ENCODER = joblib.load(
    LABEL_ENCODER_PATH
)

# ==========================================
# Predict Emotion
# ==========================================

def predict_emotion(features):
    """
    Predict facial emotion.

    Parameters
    ----------
    features : numpy.ndarray
        Shape (1, 1434)

    Returns
    -------
    emotion : str
    confidence : float
    probabilities : numpy.ndarray
    """

    if features is None:
        return None, None, None

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # Ensure correct input shape
    features = features.reshape(1, -1)

    if features.shape[1] != 1434:
        raise ValueError(
            f"Expected 1434 features, got {features.shape[1]}"
        )

    prediction = MODEL.predict(
        features,
        verbose=0
    )

    probabilities = prediction[0]

    emotion_index = np.argmax(probabilities)

    emotion = LABEL_ENCODER.inverse_transform(
        [emotion_index]
    )[0]

    confidence = float(
        probabilities[emotion_index]
    )

    return (
        emotion,
        confidence,
        probabilities
    )