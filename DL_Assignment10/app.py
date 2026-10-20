"""Streamlit UI for audio classification, speech-to-text, and text-to-speech."""
import io
import json
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch

from src.audio_utils import waveform_to_logmel
from src.model import AudioCNN
from src.speech import speech_to_text, text_to_speech

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "audio_cnn.pt"

st.set_page_config(page_title="Audio Intelligence Lab", page_icon="🎧", layout="centered")
st.title("🎧 Audio Intelligence Lab")
st.caption("Assignment 10 · Audio classification · Speech-to-text · Text-to-speech")

@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        return None, None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes

model, classes = load_classifier()

tab_classify, tab_stt, tab_tts, tab_about = st.tabs(
    ["Classify Audio", "Speech to Text", "Text to Speech", "About"]
)

with tab_classify:
    st.subheader("Environmental sound classification")
    st.write("Upload an environmental recording such as rain, a dog bark, or a vehicle sound.")
    if model is None:
        st.warning("The classifier has not been trained yet. Run `python download_dataset.py`, then `python train.py` from the project folder.")
    upload = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"], key="classify_upload")
    if upload is not None:
        st.audio(upload)
        if st.button("Classify sound", type="primary"):
            try:
                waveform, _ = librosa.load(io.BytesIO(upload.getvalue()), sr=16000, mono=True, duration=5.0)
                features = waveform_to_logmel(waveform).unsqueeze(0)
                if model is None:
                    st.error("Train the classifier before using this feature.")
                else:
                    with torch.no_grad():
                        probs = torch.softmax(model(features), dim=1)[0]
                    top_probs, top_ids = torch.topk(probs, k=min(5, len(classes)))
                    st.success(f"Predicted class: **{classes[top_ids[0].item()]}**")
                    st.write("Top predictions")
                    for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
                        st.write(f"{classes[idx]} — {prob:.1%}")
            except Exception as exc:
                st.error(f"Could not process this audio file: {exc}")

with tab_stt:
    st.subheader("Speech-to-text")
    st.write("Upload a recording containing spoken words. Whisper Tiny runs locally after its model is downloaded.")
    stt_upload = st.file_uploader("Choose a speech recording", type=["wav", "mp3", "ogg", "flac", "m4a"], key="stt_upload")
    if stt_upload is not None:
        st.audio(stt_upload)
        if st.button("Transcribe speech", type="primary"):
            suffix = Path(stt_upload.name).suffix or ".wav"
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
                    temp.write(stt_upload.getvalue())
                    temp.flush()
                    with st.spinner("Transcribing… the first run may download model files."):
                        transcript = speech_to_text(temp.name)
                st.markdown("**Transcription**")
                st.write(transcript if transcript else "No speech was recognized.")
            except Exception as exc:
                st.error(f"Speech recognition failed: {exc}")

with tab_tts:
    st.subheader("Text-to-speech")
    text = st.text_area("Enter English text", placeholder="Hello! This is a deep learning audio application.", max_chars=1000)
    if st.button("Generate speech", type="primary", disabled=not text.strip()):
        try:
            with st.spinner("Generating speech… the first run may download model files."):
                audio, sample_rate = text_to_speech(text.strip())
            audio = np.asarray(audio, dtype=np.float32)
            audio = np.clip(audio, -1.0, 1.0)
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format="WAV")
            st.audio(buffer.getvalue(), format="audio/wav")
            st.download_button("Download generated WAV", data=buffer.getvalue(), file_name="generated_speech.wav", mime="audio/wav")
        except Exception as exc:
            st.error(f"Text-to-speech failed: {exc}")

with tab_about:
    st.subheader("Project overview")
    st.markdown("""
    - **Classifier:** custom CNN trained on log-Mel spectrograms from ESC-50.
    - **Speech recognition:** pretrained OpenAI Whisper Tiny.
    - **Speech synthesis:** pretrained Facebook MMS-TTS English.
    - **Evaluation:** accuracy, macro precision, macro recall, macro F1-score, classification report, and confusion matrix.
    """)
    st.info("The ESC-50 dataset contains environmental sounds, not reference transcripts. Evaluate speech recognition separately with audio that has a known transcript.")
    if (ROOT / "results" / "metrics.json").exists():
        st.markdown("**Latest local test metrics**")
        metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
        st.json(metrics)
