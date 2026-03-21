# Bioacoustics-Species-Detection

## 📌 Overview
This project classifies bird species from audio recordings using bioacoustic analysis.  
Audio clips are converted into spectrogram images and classified using deep learning.

---

## 📂 Dataset
Kaggle Bird Song Dataset:  
https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set

**Usage:**
- Download directly from Kaggle, OR  
- Upload dataset to Google Drive and mount in Colab
---
## ⚙️ Approach

### 1. Preprocessing
- Convert audio → Mel-Spectrograms (Librosa)
- Resize images to 224×224

### 2. Models Used
- CNN: ResNet18 (PyTorch)
- Random Forest (MFCC features)
---
## 📊 Results
- CNN Accuracy: 90.14%
---
## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the notebook
3. Execute all cells
