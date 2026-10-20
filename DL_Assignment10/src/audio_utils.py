"""Audio loading and log-Mel feature extraction shared by training and inference."""
from pathlib import Path
import librosa
import numpy as np
import torch

SAMPLE_RATE = 16_000
DURATION_SECONDS = 5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
TARGET_FRAMES = 157


def waveform_to_logmel(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Return a normalized [1, n_mels, time] log-Mel spectrogram tensor."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1)
    target_length = SAMPLE_RATE * DURATION_SECONDS
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=SAMPLE_RATE // 2,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Per-example standardization makes feature scale less dependent on recording volume.
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)


def load_audio(path, sample_rate: int = SAMPLE_RATE):
    """Load audio as mono float32 at the requested sample rate."""
    waveform, sr = librosa.load(str(path), sr=sample_rate, mono=True, duration=DURATION_SECONDS)
    return waveform.astype(np.float32), sr


def load_audio_bytes(file_bytes: bytes, sample_rate: int = SAMPLE_RATE):
    """Load an in-memory audio upload as mono float32."""
    import io
    waveform, sr = librosa.load(io.BytesIO(file_bytes), sr=sample_rate, mono=True, duration=DURATION_SECONDS)
    return waveform.astype(np.float32), sr
