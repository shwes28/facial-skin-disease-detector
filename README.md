# 🔬 DermAI: Facial Skin Lesion & Acne Detection Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://facial-skin-disease-detector.streamlit.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision and clinical decision-support web application that classifies facial skin spots as either **Acne Vulgaris** or **Other Skin Lesion** using transfer learning (MobileNetV2), visual attention heatmaps (Grad-CAM), and actionable clinical triage recommendations.

---

## 🌟 Key Highlights

1. **Zero Class Imbalance Pipeline (`prepare_data.py`):**
   * Automated data ingestion that validates file integrity, removes corrupted/non-RGB artifacts, and balances the dataset to a **strict 50.0% / 50.0% ratio** before training.
2. **Transfer Learning Architecture (`model.py`):**
   * Fine-tuned **MobileNetV2** backbone pre-trained on ImageNet. Its depthwise separable convolutions enable ultra-low latency inference on standard CPU hardware.
3. **Visual Explainability (Grad-CAM Saliency):**
   * Computes a spatial feature activation heatmap overlay (red = high focus) to visually verify that the model attends to the lesion texture and borders rather than background skin artifacts.
4. **Clinical Decision Support & Ethical AI:**
   * Distinguishes routine acne care (OTC salicylic acid, gentle cleansing) from atypical lesions requiring formal dermatologist review (ABCDE criteria).
   * Prominent medical safety disclaimer for responsible deployment.

---

## 📂 Project Structure

```
facial-skin-disease-detector/
│
├── app.py              # Interactive Streamlit dashboard & web UI
├── model.py            # MobileNetV2 architecture, inference & Grad-CAM heatmaps
├── prepare_data.py     # Data validation, cleaning & 1:1 balancing pipeline
├── train.py            # PyTorch training script with data augmentation
├── requirements.txt    # Python dependencies for local & cloud deployment
├── .gitignore          # Excludes datasets, checkpoints, and cache
└── README.md           # Project documentation & deployment guide
```

---

## 🚀 Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🌐 Deployment Guide (Streamlit Cloud)

1. Fork or push this repository to your GitHub account.
2. Log in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"** and select `facial-skin-disease-detector`.
4. Set the Main file path to `app.py`.
5. Click **"Deploy!"**

---

## 🏋️ Training Pipeline

1. Place raw downloaded images into:
   * `raw_data/acne/`
   * `raw_data/other/`
2. Run the cleaning & 1:1 balancing script:
   ```bash
   python prepare_data.py
   ```
3. Run the model training pipeline:
   ```bash
   python train.py
   ```
   *(Saves the best-performing weights to `skin_model.pth`, which `app.py` automatically detects).*

---

## 📐 Technical Architecture & Design Decisions

* **Binary Triage Formulation:**
  Designed as a clinical triage decision-support tool to distinguish routine Acne Vulgaris from other atypical lesions requiring dermatologist evaluation.
* **Pre-training & Transfer Learning:**
  MobileNetV2 was chosen for its parameter efficiency (depthwise separable convolutions), allowing edge and CPU deployments with sub-second inference latency.
* **Explainability (Grad-CAM):**
  Feature activation maps are extracted from the final convolutional block to provide visual interpretability, ensuring high transparency for clinical decision support.
* **Data Integrity:**
  Automated pre-filtering removes truncated, non-RGB, and extreme aspect-ratio images to maintain high data quality and a strict 1:1 class balance.
