"""Unit tests for shared face/voice preprocessing."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from driveauth.preprocess.face import face_crop_to_model_blob, preprocess_face_crop_bgr
from driveauth.preprocess.voice import (
    normalize_capture_audio,
    preprocess,
    resample_audio,
    trim_silence,
    wav_bytes_to_float32,
)


def test_face_preprocess_changes_pixel_stats():
    crop = np.full((80, 80, 3), 90, dtype=np.uint8)
    out = preprocess_face_crop_bgr(crop)
    assert out.shape == crop.shape
    assert float(out.mean()) != float(crop.mean())


def test_face_model_blob_shape():
    crop = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    blob = face_crop_to_model_blob(crop)
    assert blob.shape == (1, 3, 112, 112)
    assert blob.dtype == np.float32


def test_resample_audio_length():
    audio = np.sin(np.linspace(0, 4 * np.pi, 8000, dtype=np.float32))
    out = resample_audio(audio, 8000, 16_000)
    assert out.shape[0] == pytest.approx(16_000, rel=0.02)


def test_trim_silence_removes_edges():
    sr = 16_000
    silent = np.zeros(sr, dtype=np.float32)
    speech = 0.2 * np.sin(np.linspace(0, 40 * np.pi, sr // 2, dtype=np.float32))
    audio = np.concatenate([silent, speech, silent])
    trimmed = trim_silence(audio, sr)
    assert trimmed.size < audio.size
    assert trimmed.size >= speech.size // 2


def test_voice_preprocess_handles_silence():
    silent = np.zeros(1000, dtype=np.float32)
    out = preprocess(silent)
    assert out.shape == silent.shape
    assert np.isfinite(out).all()


def test_normalize_capture_audio_resamples_and_trims():
    sr = 8000
    t = np.linspace(0, 1.5, int(sr * 1.5), dtype=np.float32)
    audio = 0.15 * np.sin(2 * np.pi * 220 * t)
    out, out_sr = normalize_capture_audio(audio, sr)
    assert out_sr == 16_000
    assert out.size > sr


def test_wav_bytes_roundtrip_capture_normalize():
    sr = 16_000
    t = np.linspace(0, 1.2, int(sr * 1.2), dtype=np.float32)
    sig = (0.25 * np.sin(2 * np.pi * 220 * t) * 20000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(sig.tobytes())
    audio = wav_bytes_to_float32(buf.getvalue())
    assert audio.dtype == np.float32
    assert audio.size >= int(sr * 0.8)
