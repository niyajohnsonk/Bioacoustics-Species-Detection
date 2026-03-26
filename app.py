import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import librosa
import numpy as np
from PIL import Image

st.set_page_config(page_title="Bird Classifier", layout="centered")

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

/* ---- Root Palette ---- */
:root {
    --caramel:   #D4A373;
    --cream:     #FAEDCD;
    --ivory:     #FEFAE0;
    --sage-lt:   #E9EDC9;
    --sage:      #CCD5AE;
    --text-dark: #3d3122;
    --text-mid:  #7a6346;
}

/* ---- Page Background ---- */
.stApp {
    background-color: var(--ivory);
    background-image:
        radial-gradient(ellipse at 10% 20%, rgba(212,163,115,0.18) 0%, transparent 60%),
        radial-gradient(ellipse at 90% 80%, rgba(204,213,174,0.22) 0%, transparent 60%);
    font-family: 'Lato', sans-serif;
}

/* ---- Hide default Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 720px; }

/* ---- Hero Header ---- */
.hero {
    text-align: center;
    padding: 2.8rem 2rem 2rem;
    background: linear-gradient(135deg, var(--cream) 0%, var(--sage-lt) 100%);
    border-radius: 24px;
    border: 1.5px solid var(--sage);
    box-shadow: 0 8px 32px rgba(212,163,115,0.18), 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        -45deg,
        transparent, transparent 18px,
        rgba(212,163,115,0.07) 18px, rgba(212,163,115,0.07) 19px
    );
    pointer-events: none;
}
.hero-icon {
    font-size: 3.6rem;
    line-height: 1;
    filter: drop-shadow(0 4px 12px rgba(212,163,115,0.5));
    animation: float 3.6s ease-in-out infinite;
    display: inline-block;
}
@keyframes float {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-10px); }
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: var(--text-dark);
    margin: 0.4rem 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1rem;
    color: var(--text-mid);
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.3px;
}

/* ---- Upload Zone ---- */
.upload-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--text-dark);
    margin-bottom: 0.4rem;
    display: block;
}
[data-testid="stFileUploader"] {
    background: var(--cream);
    border: 2px dashed var(--caramel);
    border-radius: 16px;
    padding: 1.4rem;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--text-dark);
    background: var(--sage-lt);
}
[data-testid="stFileUploader"] label {
    font-family: 'Lato', sans-serif;
    color: var(--text-mid);
    font-size: 0.95rem;
}

/* ---- Audio player ---- */
[data-testid="stAudio"] {
    background: var(--sage-lt);
    border-radius: 12px;
    padding: 0.6rem 1rem;
    border: 1px solid var(--sage);
    margin: 0.8rem 0;
}
audio {
    width: 100%;
    accent-color: var(--caramel);
}

/* ---- Spinner ---- */
[data-testid="stSpinner"] > div {
    border-top-color: var(--caramel) !important;
}

/* ---- Result Card ---- */
.result-card {
    background: linear-gradient(135deg, var(--caramel) 0%, #c89460 100%);
    border-radius: 20px;
    padding: 2rem 2rem 1.6rem;
    text-align: center;
    box-shadow: 0 12px 40px rgba(212,163,115,0.35);
    margin: 1.4rem 0 0.6rem;
    animation: slideUp 0.5s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-badge {
    display: inline-block;
    background: rgba(255,255,255,0.22);
    border: 1.5px solid rgba(255,255,255,0.5);
    border-radius: 50px;
    padding: 0.25rem 1rem;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.9);
    margin-bottom: 0.7rem;
    font-family: 'Lato', sans-serif;
    font-weight: 700;
}
.result-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-style: italic;
    color: #fff;
    margin: 0 0 0.2rem;
    text-shadow: 0 2px 12px rgba(0,0,0,0.12);
}
.result-conf {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.82);
    font-weight: 300;
    font-family: 'Lato', sans-serif;
}

/* ---- Confidence Arc ---- */
.arc-wrap {
    margin: 0.4rem auto 0;
    width: 100px; height: 55px;
    position: relative;
}

/* ---- Predictions Section ---- */
.pred-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: var(--text-dark);
    margin: 1.6rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.pred-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--sage);
}

/* ---- Prediction Bar ---- */
.pred-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 0.75rem;
    animation: fadeIn 0.4s ease both;
}
.pred-row:nth-child(2) { animation-delay: 0.08s; }
.pred-row:nth-child(3) { animation-delay: 0.16s; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
.pred-rank {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: var(--sage);
    color: var(--text-dark);
    font-size: 0.72rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-family: 'Lato', sans-serif;
}
.pred-rank.top { background: var(--caramel); color: #fff; }
.pred-name {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    color: var(--text-dark);
    font-size: 0.97rem;
    width: 140px;
    flex-shrink: 0;
}
.pred-bar-bg {
    flex: 1;
    background: var(--sage-lt);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.pred-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--caramel), #c89460);
    transition: width 1s cubic-bezier(0.16,1,0.3,1);
}
.pred-pct {
    font-size: 0.82rem;
    color: var(--text-mid);
    font-weight: 700;
    width: 46px;
    text-align: right;
    font-family: 'Lato', sans-serif;
}

/* ---- Divider ---- */
.divider {
    border: none;
    border-top: 1px solid var(--sage);
    margin: 1.8rem 0;
}

/* ---- Info pill ---- */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--sage-lt);
    border: 1px solid var(--sage);
    border-radius: 50px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    color: var(--text-mid);
    font-family: 'Lato', sans-serif;
    margin-bottom: 0.6rem;
}

/* ---- Stagger upload section ---- */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: var(--text-mid);
    margin-bottom: 0.5rem;
    letter-spacing: 0.2px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# CONFIG
# -------------------------------

MODEL_PATH = "models/resnet18_dl.pth"

CLASSES = [
    "bewickii",
    "cardinalis",
    "melodia",
    "migratorius",
    "polyglottos"
]

COMMON_NAMES = {
    "bewickii":    "Bewick's Wren",
    "cardinalis":  "Northern Cardinal",
    "melodia":     "Song Sparrow",
    "migratorius": "American Robin",
    "polyglottos": "Northern Mockingbird",
}

BIRD_EMOJI = {
    "bewickii":    "🪶",
    "cardinalis":  "🔴",
    "melodia":     "🎶",
    "migratorius": "🍊",
    "polyglottos": "🎵",
}

DEVICE = torch.device("cpu")

# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------

def preprocess(audio_file):
    y, sr = librosa.load(audio_file, duration=3)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db -= mel_db.min()
    mel_db /= mel_db.max()
    img = Image.fromarray((mel_db * 255).astype(np.uint8))
    img = img.resize((224, 224)).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0)
    return img

# -------------------------------
# HERO
# -------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-icon">🐦</div>
    <h1>Bird Sound Classifier</h1>
    <p>Upload a bird recording and discover the species within seconds</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# SUPPORTED SPECIES INFO
# -------------------------------

st.markdown("""
<style>
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #3d3122 !important;
}
[data-testid="stFileUploaderFileName"] {
    color: #3d3122 !important;
}
</style>
""", unsafe_allow_html=True)

with st.expander("🌿 Supported species"):
    cols = st.columns(5)
    for i, (k, v) in enumerate(COMMON_NAMES.items()):
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center; padding:0.5rem;'>
                <div style='font-size:1.8rem'>{BIRD_EMOJI[k]}</div>
                <div style='font-family:Playfair Display,serif; font-style:italic;
                            font-size:0.78rem; color:#3d3122; margin-top:0.2rem;'>{v}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

# -------------------------------
# UPLOADER
# -------------------------------

st.markdown("<p class='section-label'>Upload a .wav recording</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["wav"], label_visibility="collapsed")

# -------------------------------
# PREDICTION
# -------------------------------

if uploaded_file is not None:

    st.markdown("<div class='info-pill'>🎙️ Audio loaded — ready to analyse</div>", unsafe_allow_html=True)
    st.audio(uploaded_file, format="audio/wav")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    with st.spinner("Analysing birdsong…"):
        input_tensor = preprocess(uploaded_file).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence_score = confidence.item() * 100
    common = COMMON_NAMES[predicted_class]
    emoji = BIRD_EMOJI[predicted_class]

    # ---- Main result card ----
    st.markdown(f"""
    <div class="result-card">
        <div class="result-badge">✦ Identified species ✦</div>
        <div class="result-name">{emoji} {common}</div>
        <div class="result-conf"><em>{predicted_class}</em> · {confidence_score:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Top predictions ----
    top_probs, top_idxs = torch.topk(probs, 3)

    st.markdown("<div class='pred-header'>Top 3 predictions</div>", unsafe_allow_html=True)

    for i in range(3):
        bird_key = CLASSES[top_idxs[0][i].item()]
        bird_name = COMMON_NAMES[bird_key]
        prob = top_probs[0][i].item() * 100
        rank_class = "top" if i == 0 else ""
        bar_width = f"{prob:.1f}%"

        st.markdown(f"""
        <div class="pred-row">
            <div class="pred-rank {rank_class}">{i+1}</div>
            <div class="pred-name">{bird_name}</div>
            <div class="pred-bar-bg">
                <div class="pred-bar-fill" style="width:{bar_width}"></div>
            </div>
            <div class="pred-pct">{prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align:center; font-size:0.78rem; color:#a89070; font-family:Lato,sans-serif;
              font-style:italic; margin:0;'>
        Model trained on mel-spectrogram representations of bird audio.
    </p>""", unsafe_allow_html=True)