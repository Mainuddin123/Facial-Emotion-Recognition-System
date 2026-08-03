# ==========================================
# Import Libraries
# ==========================================

import numpy as np
import joblib

from pathlib import Path

# ==========================================
# Load Saved Scaler
# ==========================================

SCALER_PATH = Path("models/scaler_optuna.pkl")

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_PATH.resolve()}"
    )

SCALER = joblib.load(SCALER_PATH)

# ==========================================
# Preprocess Features
# ==========================================

def preprocess(features):
    """
    Preprocess extracted facial landmark features.

    Parameters
    ----------
    features : numpy.ndarray
        Shape (1434,) or (1, 1434)

    Returns
    -------
    numpy.ndarray
        Shape (1, 1434)
    """

    if features is None:
        return None

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # Ensure correct shape
    features = features.reshape(1, -1)

    # Validate feature size
    if features.shape[1] != 1434:
        raise ValueError(
            f"Expected 1434 features, got {features.shape[1]}"
        )

    # Standardize features
    features = SCALER.transform(features)

    return features