# ==========================================
# Import Libraries
# ==========================================

import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# Load Face Landmarker Model
# ==========================================

MODEL_PATH = Path("models/face_landmarker.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face Landmarker model not found:\n{MODEL_PATH.resolve()}"
    )

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

FACE_DETECTOR = vision.FaceLandmarker.create_from_options(options)

# ==========================================
# Internal Face Detection
# ==========================================

def _detect_face(image):
    """
    Detect face landmarks using MediaPipe Face Landmarker.

    Parameters
    ----------
    image : OpenCV BGR image

    Returns
    -------
    MediaPipe detection result or None
    """

    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = FACE_DETECTOR.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result


# ==========================================
# Extract Landmark Features
# ==========================================

def extract_landmarks(image):
    """
    Extract normalized facial landmarks.

    Returns
    -------
    numpy.ndarray (1434,)
    or
    None
    """

    result = _detect_face(image)

    if result is None:
        return None

    landmarks = result.face_landmarks[0]

    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Center normalization
    center = coords.mean(axis=0)
    coords -= center

    # Scale normalization
    scale = np.max(np.linalg.norm(coords, axis=1))

    if scale > 0:
        coords /= scale

    return coords.flatten()


# ==========================================
# Draw Landmarks
# ==========================================

def draw_landmarks(image):
    """
    Return original image.

    Face mesh visualization is intentionally disabled.
    """

    if image is None:
        return image

    return image.copy()


# ==========================================
# Get Face Bounding Box
# ==========================================

def get_face_bbox(image):
    """
    Get face bounding box.

    Returns
    -------
    (x_min, y_min, x_max, y_max)
    or
    None
    """

    result = _detect_face(image)

    if result is None:
        return None

    landmarks = result.face_landmarks[0]

    h, w = image.shape[:2]

    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x_min = max(0, int(min(xs)))
    y_min = max(0, int(min(ys)))
    x_max = min(w - 1, int(max(xs)))
    y_max = min(h - 1, int(max(ys)))

    return x_min, y_min, x_max, y_max