"""Lazy-loaded pretrained speech-to-text and text-to-speech pipelines."""
import numpy as np
import torch
from transformers import pipeline

_asr_pipeline = None
_tts_pipeline = None


def speech_to_text(audio_path: str) -> str:
    """Transcribe a speech recording with Whisper Tiny."""
    global _asr_pipeline
    if _asr_pipeline is None:
        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=0 if torch.cuda.is_available() else -1,
        )
    result = _asr_pipeline(audio_path)
    return result.get("text", "").strip()


def text_to_speech(text: str):
    """Generate English speech using the pretrained MMS-TTS model."""
    global _tts_pipeline
    if _tts_pipeline is None:
        _tts_pipeline = pipeline(
            "text-to-audio",
            model="facebook/mms-tts-eng",
            device=0 if torch.cuda.is_available() else -1,
        )
    result = _tts_pipeline(text)
    audio = np.asarray(result["audio"]).squeeze()
    return audio, int(result["sampling_rate"])
