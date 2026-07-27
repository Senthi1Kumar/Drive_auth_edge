"""Face image preprocessing — shared enroll, inference, and training pipeline."""

from __future__ import annotations

import numpy as np

FACE_MODEL_SIZE = (112, 112)


def preprocess_face_crop_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """CLAHE lighting normalize + mild denoise on a BGR face crop."""
    import cv2  # type: ignore

    img = np.asarray(crop_bgr)
    if img.size == 0:
        return img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    img = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    # Preserve edges for embedding; dampen sensor noise in cabin lighting.
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def face_crop_to_model_blob(
    crop_bgr: np.ndarray,
    *,
    face_size: tuple[int, int] = FACE_MODEL_SIZE,
) -> np.ndarray:
    """Preprocess crop → MobileFaceNet NCHW float32 blob."""
    import cv2  # type: ignore

    proc = preprocess_face_crop_bgr(crop_bgr)
    face_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, face_size)
    blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
    return np.transpose(blob, (2, 0, 1))[np.newaxis]
