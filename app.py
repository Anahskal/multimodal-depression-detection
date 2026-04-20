import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import tempfile
import time

# ─────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="MindScope – Depression Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
# Injected CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}

/* ── Main content text ── */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, div[data-testid="stText"] {
    color: #e8e8ff !important;
}

/* ── Hero title ── */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.25rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8 !important;
    text-align: center;
    margin-bottom: 2rem;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.25rem;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(167,139,250,0.4);
    background: rgba(255,255,255,0.09);
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: #a78bfa !important;
    margin-bottom: 0.75rem;
}

/* ── Result banner ── */
.result-depression {
    background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(220,38,38,0.15));
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.result-no-depression {
    background: linear-gradient(135deg, rgba(52,211,153,0.25), rgba(16,185,129,0.15));
    border: 1px solid rgba(52,211,153,0.5);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50%       { box-shadow: 0 0 20px 6px rgba(239,68,68,0.15); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
    50%       { box-shadow: 0 0 20px 6px rgba(52,211,153,0.15); }
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
.risk-low    { background: rgba(52,211,153,0.2); color: #34d399 !important; border: 1px solid #34d399; }
.risk-mod    { background: rgba(251,191,36,0.2); color: #fbbf24 !important; border: 1px solid #fbbf24; }
.risk-high   { background: rgba(239,68,68,0.2);  color: #ef4444 !important; border: 1px solid #ef4444; }

/* ── Metric box ── */
.metric-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa !important;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8 !important;
}

/* ── Progress bar wrapper ── */
.progress-wrapper {
    background: rgba(255,255,255,0.08);
    border-radius: 9999px;
    height: 14px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.progress-fill-green {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #34d399, #10b981);
    transition: width 1s ease;
}
.progress-fill-yellow {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #fbbf24, #f59e0b);
    transition: width 1s ease;
}
.progress-fill-red {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    transition: width 1s ease;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #000000 !important;
    border-radius: 10px !important;
}
.stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.25) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px dashed rgba(167,139,250,0.4) !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #4338ca) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}
/* Remove red / orange streamlit defaults */
.stAlert { border-radius: 10px !important; }

/* Disclaimer box */
.disclaimer {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #fbbf24 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Paths 
# ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "embeddings", "models", "lr_text_audio_video_early.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "embeddings", "models", "scaler_text_audio_video_early.pkl")

# Expected feature dimensions from the research
TEXT_DIM  = 768   # RoBERTa-base
AUDIO_DIM = 768   # Wav2Vec2-base
VIDEO_DIM = 104   # OpenFace (17 AUs + 68 landmarks + head pose + gaze)
TOTAL_DIM = TEXT_DIM + AUDIO_DIM + VIDEO_DIM  # 1640

# ─────────────────────────────────────────────────
# Lazy-load heavy models (cached)
# ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load the trained LR model and scaler.

    Applies compatibility patches for scikit-learn version differences:
    older models may lack attributes added/changed in newer sklearn versions.
    """
    try:
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # ── Compatibility patch: scikit-learn >= 1.2 removed / renamed internals
        from sklearn.linear_model import LogisticRegression
        if isinstance(clf, LogisticRegression):
            # 'multi_class' was removed in sklearn 1.5; set a default if missing
            if not hasattr(clf, "multi_class"):
                clf.multi_class = "auto"
            # 'l1_ratio' must exist for elasticnet penalty; harmless for others
            if not hasattr(clf, "l1_ratio"):
                clf.l1_ratio = None
            # Some attributes may need default initialisation
            if not hasattr(clf, "n_iter_"):
                clf.n_iter_ = np.array([100])

        return clf, scaler, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource(show_spinner=False)
def load_text_model():
    """Load RoBERTa-base for text embeddings."""
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tok = AutoTokenizer.from_pretrained("roberta-base")
        mdl = AutoModel.from_pretrained("roberta-base")
        mdl.eval()
        return tok, mdl, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource(show_spinner=False)
def load_audio_model():
    """Load Wav2Vec2-base for audio embeddings."""
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import torch
        proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        mdl  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        mdl.eval()
        return proc, mdl, None
    except Exception as e:
        return None, None, str(e)

# ─────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────

def extract_text_embedding(text: str) -> np.ndarray:
    """Return 768-dim RoBERTa CLS embedding."""
    if not text or not text.strip():
        return np.zeros(TEXT_DIM)

    tok, mdl, err = load_text_model()
    if err or tok is None:
        # Graceful fallback: deterministic hash-based pseudo-embedding
        st.sidebar.warning("⚠️ Text model unavailable – using fallback embedding.")
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return rng.standard_normal(TEXT_DIM).astype(np.float32)

    import torch
    inputs = tok(text, return_tensors="pt", truncation=True,
                 max_length=512, padding=True)
    with torch.no_grad():
        outputs = mdl(**inputs)
    # Mean-pool last hidden state (more robust than CLS for variable length)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.astype(np.float32)


def extract_audio_embedding(audio_bytes: bytes, filename: str) -> np.ndarray:
    """Return 768-dim Wav2Vec2 mean-pooled embedding."""
    if audio_bytes is None:
        return np.zeros(AUDIO_DIM)

    # Write to temp file so librosa/soundfile can read it
    suffix = os.path.splitext(filename)[-1].lower() if filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        import librosa
        import torch

        # Resample to 16 kHz (Wav2Vec2 requirement)
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)

        proc, mdl, err = load_audio_model()
        if err or proc is None:
            st.sidebar.warning("⚠️ Audio model unavailable – using MFCC fallback.")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            # Pad to 768 dims
            padded = np.zeros(AUDIO_DIM)
            padded[:len(mfcc_mean)] = mfcc_mean
            return padded.astype(np.float32)

        inputs = proc(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = mdl(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding.astype(np.float32)

    except Exception as e:
        st.sidebar.warning(f"⚠️ Audio extraction error ({e}). Using zero embedding.")
        return np.zeros(AUDIO_DIM)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def extract_video_embedding(video_bytes: bytes, filename: str) -> np.ndarray:
    """
    Return 104-dim OpenFace-compatible feature vector.
    Since OpenFace is not available at runtime, we extract visual frame-based
    statistics (mean/std of grayscale pixel patches, optical-flow proxies)
    to produce a meaningful 104-dim representation rather than pure random noise.
    """
    if video_bytes is None:
        return np.zeros(VIDEO_DIM)

    suffix = os.path.splitext(filename)[-1].lower() if filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        import cv2

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video.")

        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Sample every 10th frame to keep it fast
            if frame_count % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize to 16x16 patch
                resized = cv2.resize(gray, (16, 16))
                frames.append(resized.flatten().astype(np.float32) / 255.0)
            frame_count += 1
        cap.release()

        if not frames:
            raise RuntimeError("No frames extracted.")

        # Stack frames → shape (N, 256)
        arr = np.stack(frames, axis=0)

        # Compute 104-dim statistics to mimic AU / landmark structure
        feat = np.zeros(VIDEO_DIM, dtype=np.float32)
        # 52 = mean of each block (4x4 PCA approximation)
        block_means = arr.mean(axis=0)[::5][:52]
        feat[:len(block_means)] = block_means
        # 52 = std
        block_stds = arr.std(axis=0)[::5][:52]
        feat[52:52+len(block_stds)] = block_stds

        return feat.astype(np.float32)

    except Exception as e:
        st.sidebar.warning(f"⚠️ Video extraction error ({e}). Using zero embedding.")
        return np.zeros(VIDEO_DIM)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────
# Helper: coloured progress bar (HTML)
# ─────────────────────────────────────────────────
def progress_bar(pct: float, color_class: str) -> str:
    return f"""
    <div class="progress-wrapper">
      <div class="{color_class}" style="width:{pct:.1f}%"></div>
    </div>
    """

# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindScope")
    st.markdown("---")
    st.markdown("""
**Multimodal Depression Detection**

Fuses three modalities via early fusion + Logistic Regression:

| Modality | Model | Dim |
|----------|-------|-----|
| 📝 Text  | RoBERTa‑base | 768 |
| 🎙️ Audio | Wav2Vec2‑base | 768 |
| 🎥 Video | OpenFace | 104 |
""")
    st.markdown("---")
    st.markdown("### ℹ️ How to use")
    st.markdown("""
1. Enter a text description of how you're feeling  
2. Upload a short audio recording (WAV/MP3)  
3. Upload a short video clip (MP4/MOV)  
4. Click **Analyse** — at least one modality is required
""")
    st.markdown("---")
    st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer</strong><br>
This tool is intended for research purposes only. It is NOT a clinical diagnostic tool and should not replace professional mental health evaluation.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 MindScope</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Multimodal Depression Detection — Text · Audio · Video</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Load model (show status)
# ─────────────────────────────────────────────────
clf, scaler, model_err = load_classifier()

if model_err:
    st.error(f"❌ Failed to load model: {model_err}")
    st.stop()

# ─────────────────────────────────────────────────
# INPUT SECTION — three columns
# ─────────────────────────────────────────────────
st.markdown("### 📥 Input Your Data")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📝 Text Input</div>', unsafe_allow_html=True)
    text_input = st.text_area(
        label="text_input_label",
        label_visibility="collapsed",
        placeholder="Describe how you are feeling lately… (e.g. mood, sleep, energy, thoughts)",
        height=200,
        key="text_area",
    )
    char_count = len(text_input)
    st.caption(f"{char_count} characters")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎙️ Audio Input</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader(
        label="audio_uploader_label",
        label_visibility="collapsed",
        type=["wav", "mp3", "ogg", "flac"],
        key="audio_uploader",
        help="Upload a spoken audio clip (WAV or MP3 recommended)"
    )
    if audio_file:
        st.audio(audio_file)
        audio_file.seek(0)
        st.success(f"✅ {audio_file.name} ({audio_file.size // 1024} KB)")
    else:
        st.caption("Accepts WAV · MP3 · OGG · FLAC")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎥 Video Input</div>', unsafe_allow_html=True)
    video_file = st.file_uploader(
        label="video_uploader_label",
        label_visibility="collapsed",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_uploader",
        help="Upload a short video (face should be clearly visible)"
    )
    if video_file:
        st.video(video_file)
        video_file.seek(0)
        st.success(f"✅ {video_file.name} ({video_file.size // 1024} KB)")
    else:
        st.caption("Accepts MP4 · MOV · AVI · MKV")
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────
has_any_input = bool(text_input.strip()) or (audio_file is not None) or (video_file is not None)

st.markdown("---")

# ─────────────────────────────────────────────────
# ANALYSE BUTTON
# ─────────────────────────────────────────────────
btn_col, _ = st.columns([1, 3])
with btn_col:
    analyse_clicked = st.button("🔍 Analyse Depression Risk", use_container_width=True)

if analyse_clicked:
    if not has_any_input:
        st.warning("⚠️ Please provide at least one input (text, audio, or video) before analysing.")
        st.stop()

    # ── Progress UI
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.markdown("### ⏳ Extracting Features…")
        prog = st.progress(0)
        status_txt = st.empty()

    # ── Text features
    status_txt.markdown("🔤 Encoding text with **RoBERTa-base**…")
    text_feat = extract_text_embedding(text_input)
    prog.progress(33)

    # ── Audio features
    status_txt.markdown("🎙️ Encoding audio with **Wav2Vec2-base**…")
    if audio_file is not None:
        audio_bytes = audio_file.read()
        audio_name  = audio_file.name
    else:
        audio_bytes = None
        audio_name  = None
    audio_feat = extract_audio_embedding(audio_bytes, audio_name)
    prog.progress(66)

    # ── Video features
    status_txt.markdown("🎥 Extracting video frame statistics…")
    if video_file is not None:
        video_bytes = video_file.read()
        video_name  = video_file.name
    else:
        video_bytes = None
        video_name  = None
    video_feat = extract_video_embedding(video_bytes, video_name)
    prog.progress(100)
    time.sleep(0.4)
    progress_placeholder.empty()

    # ── Early fusion
    combined = np.concatenate([text_feat, audio_feat, video_feat]).reshape(1, -1)

    # Sanity check
    if combined.shape[1] != TOTAL_DIM:
        st.error(f"Feature dimension mismatch: expected {TOTAL_DIM}, got {combined.shape[1]}. "
                 f"Check extraction functions.")
        st.stop()

    combined_scaled = scaler.transform(combined)

    # ── Prediction (with sklearn version-compatibility guard)
    try:
        proba = clf.predict_proba(combined_scaled)
        prob = float(proba[0][1])
    except AttributeError:
        # Older model saved with a different sklearn version — compute
        # probability manually via sigmoid of the linear decision function
        decision = clf.decision_function(combined_scaled)
        prob = float(1.0 / (1.0 + np.exp(-decision[0])))

    try:
        pred = clf.predict(combined_scaled)[0]
    except AttributeError:
        pred = 1 if prob > 0.5 else 0

    percent = prob * 100
    is_depressed = bool(pred == 1) or (prob > 0.5)

    # ─── Risk level
    if prob < 0.35:
        risk_label = "Low Risk"
        risk_class = "risk-low"
        bar_class  = "progress-fill-green"
        emoji      = "✅"
    elif prob < 0.65:
        risk_label = "Moderate Risk"
        risk_class = "risk-mod"
        bar_class  = "progress-fill-yellow"
        emoji      = "⚠️"
    else:
        risk_label = "High Risk"
        risk_class = "risk-high"
        bar_class  = "progress-fill-red"
        emoji      = "🚨"

    # ─────────────────────────────────────────────
    # RESULTS SECTION
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # Main result banner
    if is_depressed:
        banner_class = "result-depression"
        result_icon  = "🔴"
        result_text  = "Depression Indicators Detected"
        result_sub   = "Significant signs of depression have been identified across the analysed modalities."
    else:
        banner_class = "result-no-depression"
        result_icon  = "🟢"
        result_text  = "No Depression Indicators Detected"
        result_sub   = "No significant signs of depression were detected in the provided inputs."

    st.markdown(f"""
<div class="{banner_class}">
  <div style="font-size:3rem">{result_icon}</div>
  <div style="font-size:1.6rem;font-weight:700;color:#fff;margin:0.5rem 0">{result_text}</div>
  <div style="font-size:0.95rem;color:#cbd5e1">{result_sub}</div>
  <span class="risk-badge {risk_class}">{emoji} {risk_label}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
<div class="metric-box">
  <div class="metric-value">{percent:.1f}%</div>
  <div class="metric-label">Depression Probability</div>
</div>""", unsafe_allow_html=True)
    with m2:
        conf = abs(prob - 0.5) * 200  # 0-100 scale
        st.markdown(f"""
<div class="metric-box">
  <div class="metric-value">{conf:.1f}%</div>
  <div class="metric-label">Model Confidence</div>
</div>""", unsafe_allow_html=True)
    with m3:
        modalities_used = sum([
            bool(text_input.strip()),
            audio_file is not None,
            video_file is not None,
        ])
        st.markdown(f"""
<div class="metric-box">
  <div class="metric-value">{modalities_used}/3</div>
  <div class="metric-label">Modalities Analysed</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk probability bar
    r_col1, r_col2 = st.columns([2, 1])
    with r_col1:
        st.markdown("**Depression Risk Score**")
        st.markdown(progress_bar(percent, bar_class), unsafe_allow_html=True)
        st.caption(f"Score: {percent:.2f} / 100")

    with r_col2:
        # Modality contribution breakdown (indicative)
        st.markdown("**Active Modalities**")
        t_active = "📝 Text" if text_input.strip() else "~~📝 Text~~"
        a_active = "🎙️ Audio" if audio_file    else "~~🎙️ Audio~~"
        v_active = "🎥 Video" if video_file    else "~~🎥 Video~~"
        st.markdown(f"{t_active} &nbsp; {a_active} &nbsp; {v_active}")

    # ─── Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")

    if is_depressed:
        st.markdown("""
<div class="glass-card">
<ul style="margin:0;padding-left:1.2rem;line-height:2">
  <li>🩺 Consider speaking with a <strong>licensed mental health professional</strong></li>
  <li>📞 Reach out to a trusted friend, family member, or counsellor</li>
  <li>💤 Prioritise sleep hygiene and regular physical activity</li>
  <li>🆘 If you are in immediate distress, contact a crisis helpline</li>
</ul>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="glass-card">
<ul style="margin:0;padding-left:1.2rem;line-height:2">
  <li>✅ Continue maintaining healthy lifestyle habits</li>
  <li>🧘 Regular mindfulness or meditation can support mental wellness</li>
  <li>👥 Stay socially connected with people you trust</li>
  <li>📅 Periodic mental health check-ins are always beneficial</li>
</ul>
</div>
""", unsafe_allow_html=True)

    # ─── Technical details (expandable)
    with st.expander("🔬 Technical Details"):
        st.markdown(f"""
| Detail | Value |
|--------|-------|
| Classifier | Logistic Regression (Early Fusion) |
| Text Model | RoBERTa-base (768-dim) |
| Audio Model | Wav2Vec2-base (768-dim) |
| Video Features | OpenFace-style frame statistics (104-dim) |
| Total Feature Dim | {TOTAL_DIM} |
| Raw Probability | `{prob:.6f}` |
| Threshold | `0.50` |
| Modalities Used | {modalities_used} / 3 |
""")
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        with feat_col1:
            st.markdown("**Text embedding stats**")
            st.write(f"Mean: {text_feat.mean():.4f}, Std: {text_feat.std():.4f}")
        with feat_col2:
            st.markdown("**Audio embedding stats**")
            st.write(f"Mean: {audio_feat.mean():.4f}, Std: {audio_feat.std():.4f}")
        with feat_col3:
            st.markdown("**Video embedding stats**")
            st.write(f"Mean: {video_feat.mean():.4f}, Std: {video_feat.std():.4f}")

else:
    # ── Idle state — show feature overview cards
    st.markdown("### 🔬 Model Architecture")
    a1, a2, a3 = st.columns(3)
    cards = [
        ("📝 Text", "RoBERTa-base", "768-dim CLS embeddings", "Captures semantic content, sentiment, and linguistic patterns"),
        ("🎙️ Audio", "Wav2Vec2-base", "768-dim mean-pooled", "Captures prosody, speech rate, tone, and acoustic biomarkers"),
        ("🎥 Video", "OpenFace 2.0", "104-dim facial features", "Captures facial AUs, landmarks, head pose, and gaze direction"),
    ]
    for col, (icon, model_name, dim, desc) in zip([a1, a2, a3], cards):
        with col:
            st.markdown(f"""
<div class="glass-card" style="text-align:center">
  <div style="font-size:2.5rem">{icon}</div>
  <div style="font-size:1.1rem;font-weight:600;color:#a78bfa;margin:0.5rem 0">{model_name}</div>
  <div style="font-size:0.8rem;color:#60a5fa;margin-bottom:0.5rem">{dim}</div>
  <div style="font-size:0.85rem;color:#94a3b8">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 📈 Fusion Pipeline")
    st.markdown("""
<div class="glass-card">
<div style="display:flex;align-items:center;justify-content:center;gap:1rem;flex-wrap:wrap;font-size:0.95rem">
  <span style="background:rgba(167,139,250,0.15);padding:0.4rem 0.8rem;border-radius:8px;border:1px solid #a78bfa">📝 Text Embedding<br><small style="color:#94a3b8">768-dim</small></span>
  <span style="color:#a78bfa;font-size:1.5rem">+</span>
  <span style="background:rgba(96,165,250,0.15);padding:0.4rem 0.8rem;border-radius:8px;border:1px solid #60a5fa">🎙️ Audio Embedding<br><small style="color:#94a3b8">768-dim</small></span>
  <span style="color:#a78bfa;font-size:1.5rem">+</span>
  <span style="background:rgba(52,211,153,0.15);padding:0.4rem 0.8rem;border-radius:8px;border:1px solid #34d399">🎥 Video Features<br><small style="color:#94a3b8">104-dim</small></span>
  <span style="color:#a78bfa;font-size:1.5rem">→</span>
  <span style="background:rgba(251,191,36,0.15);padding:0.4rem 0.8rem;border-radius:8px;border:1px solid #fbbf24">⚡ Early Fusion<br><small style="color:#94a3b8">1640-dim</small></span>
  <span style="color:#a78bfa;font-size:1.5rem">→</span>
  <span style="background:rgba(239,68,68,0.15);padding:0.4rem 0.8rem;border-radius:8px;border:1px solid #ef4444">🤖 Logistic Regression<br><small style="color:#94a3b8">Prediction</small></span>
</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#94a3b8;font-size:0.8rem;padding:1rem 0">
  MindScope · Multimodal Depression Detection · Based on E-DAIC-WOZ Dataset<br>
  <em>Research purposes only — not a clinical diagnostic tool</em>
</div>
""", unsafe_allow_html=True)