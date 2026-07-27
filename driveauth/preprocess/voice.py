"""Voice audio preprocessing — shared enroll, inference, and training pipeline."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000
_PREEMPH = 0.97
_RMS_TGT = 0.08
_RMS_FLOOR = 1e-6


def resample_audio(
    audio: np.ndarray,
    src_sr: int,
    tgt_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Band-limited linear resample to target sample rate."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_sr == tgt_sr or audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.size / float(src_sr)
    n_out = max(1, int(round(duration * tgt_sr)))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    frame_ms: int = 20,
    energy_ratio: float = 0.08,
    pad_ms: int = 50,
) -> np.ndarray:
    """Energy-VAD trim of leading/trailing silence with short edge padding."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n = (audio.size // frame) * frame
    if n < frame:
        return audio
    chunks = audio[:n].reshape(-1, frame)
    energies = np.sqrt((chunks**2).mean(axis=1) + 1e-12)
    peak = float(energies.max()) if energies.size else 0.0
    thr = max(energy_ratio * peak, 1e-5)
    active = energies >= thr
    if not active.any():
        return audio
    first = int(np.argmax(active))
    last = int(len(active) - 1 - np.argmax(active[::-1]))
    pad = int(sample_rate * pad_ms / 1000)
    i0 = max(0, first * frame - pad)
    i1 = min(audio.size, (last + 1) * frame + pad)
    trimmed = audio[i0:i1]
    min_samples = sample_rate // 4
    if trimmed.size < min_samples and audio.size >= min_samples:
        return audio
    return trimmed.astype(np.float32)


def normalize_capture_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Capture-time cleanup: mono float32 @ target_sr with silence trimmed."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != target_sr:
        audio = resample_audio(audio, sample_rate, target_sr)
        sample_rate = target_sr
    audio = trim_silence(audio, sample_rate)
    return audio, sample_rate


def preprocess(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Matcher pipeline: trim → pre-emphasis → RMS normalize."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    x, _ = normalize_capture_audio(audio, sample_rate)
    out = np.empty_like(x, dtype=np.float32)
    out[0] = x[0]
    out[1:] = x[1:] - _PREEMPH * x[:-1]
    rms = float(np.sqrt(np.mean(out**2)))
    if rms > _RMS_FLOOR:
        out *= _RMS_TGT / rms
    return out


def _decode_wav_bytes(data: bytes) -> tuple[np.ndarray, int, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    return audio.astype(np.float32), sr, sw


def load_wav(
    path: Path | str,
    sr: int = TARGET_SAMPLE_RATE,
    *,
    apply_capture_normalize: bool = True,
) -> np.ndarray:
    """Load WAV → mono float32 @ sr (optional capture-time trim/resample)."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.readframes(w.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            file_sr = w.getframerate()
    except Exception:
        import soundfile as sf  # type: ignore

        audio, file_sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

    if apply_capture_normalize:
        audio, _ = normalize_capture_audio(audio, file_sr, target_sr=sr)
        return audio
    if file_sr != sr:
        audio = resample_audio(audio, file_sr, sr)
    return audio.astype(np.float32)


def wav_bytes_to_float32(
    data: bytes,
    *,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Decode uploaded WAV bytes → mono float32 with capture normalization."""
    if not data:
        raise ValueError("empty audio")
    audio, sr, _ = _decode_wav_bytes(data)
    audio, _ = normalize_capture_audio(audio, sr, target_sr=target_sr)
    return audio


def float32_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> bytes:
    """Encode mono float32 audio as 16-bit PCM WAV bytes."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(sample_rate)
        dst.writeframes(pcm_i16.tobytes())
    return buf.getvalue()
