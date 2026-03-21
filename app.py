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

DEVICE = torch.device("cpu")

# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():
    model = models.resnet18()

    # Modify final layer (IMPORTANT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    # Load trained weights
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

    # Generate Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize (important for CNN)
    mel_db -= mel_db.min()
    mel_db /= mel_db.max()

    # Convert to image
    img = Image.fromarray((mel_db * 255).astype(np.uint8))
    img = img.resize((224, 224)).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img = transform(img).unsqueeze(0)
    return img

# -------------------------------
# UI
# -------------------------------



st.title("🐦 Bird Sound Classifier")
st.write("Upload a bird audio file (.wav) to predict the species")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

# -------------------------------
# PREDICTION
# -------------------------------

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Processing..."):
        input_tensor = preprocess(uploaded_file).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)

            confidence, predicted = torch.max(probs, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence_score = confidence.item() * 100

    st.success(f"🎯 Predicted Bird: **{predicted_class}**")
    st.info(f"Confidence: {confidence_score:.2f}%")

    # -------------------------------
    # TOP 3 PREDICTIONS (Nice touch 🔥)
    # -------------------------------
    st.subheader("Top Predictions")

    top_probs, top_idxs = torch.topk(probs, 3)

    for i in range(3):
        bird = CLASSES[top_idxs[0][i].item()]
        prob = top_probs[0][i].item() * 100
        st.write(f"{bird}: {prob:.2f}%")