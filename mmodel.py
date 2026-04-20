import streamlit as st
import numpy as np
import cv2
import librosa
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deepface import DeepFace
import tempfile

st.set_page_config(page_title="MindScope", layout="wide")

st.title("🧠 MindScope: Multimodal Depression Detection")

st.write("Analyze text, audio, and video for emotional indicators.")

# -------------------------------
# TEXT ANALYSIS
# -------------------------------

analyzer = SentimentIntensityAnalyzer()

def analyze_text_emotion(text):

    if not text.strip():
        return 0.5

    score = analyzer.polarity_scores(text)

    compound = score["compound"]

    depression_score = (1 - compound) / 2

    return float(depression_score)

# -------------------------------
# AUDIO ANALYSIS
# -------------------------------

def analyze_audio_emotion(audio_file):

    y, sr = librosa.load(audio_file)

    pitch = np.mean(librosa.yin(y, 50, 300))
    energy = np.mean(librosa.feature.rms(y))

    sadness = 0

    if pitch < 120:
        sadness += 0.4

    if energy < 0.02:
        sadness += 0.4

    return float(min(sadness, 1.0))

# -------------------------------
# VIDEO ANALYSIS
# -------------------------------

def analyze_video_emotion(video_file):

    cap = cv2.VideoCapture(video_file)

    sadness_scores = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        try:

            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            sadness = result[0]["emotion"]["sad"]
            sadness_scores.append(sadness)

        except:
            pass

    cap.release()

    if len(sadness_scores) == 0:
        return 0.5

    return float(np.mean(sadness_scores) / 100)

# -------------------------------
# MULTIMODAL FUSION
# -------------------------------

def fuse_scores(text_score, audio_score, video_score):

    weights = {
        "text": 0.4,
        "audio": 0.3,
        "video": 0.3
    }

    final = (
        text_score * weights["text"]
        + audio_score * weights["audio"]
        + video_score * weights["video"]
    )

    return float(final)

# -------------------------------
# UI INPUT
# -------------------------------

st.header("📥 Input Data")

text_input = st.text_area("📝 Text Input")

audio_file = st.file_uploader(
    "🎙️ Upload Audio",
    type=["wav","mp3","ogg","flac"]
)

video_file = st.file_uploader(
    "🎥 Upload Video",
    type=["mp4","mov","avi","mkv"]
)

# -------------------------------
# ANALYSIS
# -------------------------------

if st.button("Run Analysis"):

    text_score = analyze_text_emotion(text_input)

    audio_score = 0.5
    video_score = 0.5

    if audio_file:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        audio_score = analyze_audio_emotion(audio_path)

    if video_file:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name

        video_score = analyze_video_emotion(video_path)

    final_score = fuse_scores(text_score, audio_score, video_score)

    st.header("📊 Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Text Score", round(text_score,2))
    col2.metric("Audio Score", round(audio_score,2))
    col3.metric("Video Score", round(video_score,2))

    st.write("### Final Depression Risk Score")

    st.progress(float(final_score))

    st.write("Score:", round(final_score*100,2), "%")

    if final_score > 0.65:

        st.error("🔴 Depression Indicators Detected")

        st.markdown("""
        **Recommendations**
        - Consider speaking with a mental health professional
        - Talk to a trusted friend or family member
        - Maintain sleep and physical activity
        """)

    else:

        st.success("🟢 No Significant Depression Indicators")
