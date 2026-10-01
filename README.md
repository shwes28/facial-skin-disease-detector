# 🔬 DermAI: Facial Skin Lesion & Acne Detection Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision and clinical decision-support web application that classifies facial skin spots as either **Acne Vulgaris** or **Other Skin Lesion** using transfer learning (MobileNetV2), visual attention heatmaps (Grad-CAM), and actionable next-step clinical recommendations.

---

## 🌟 Key Highlights

1. **Zero Class Imbalance Pipeline (`prepare_data.py`):**
   * Automatically validates, filters corrupted files, and balances the dataset to a **strict 50.0% / 50.0% ratio** before model training.
2. **Transfer Learning Architecture (`model.py`):**
   * Uses **MobileNetV2** pre-trained on ImageNet. Its depthwise separable convolutions allow low-latency inference on standard CPU hardware.
3. **Visual Explainability (Grad-CAM Saliency):**
   * Computes a spatial feature activation heatmap overlay (red = high focus) to prove the model is analyzing the actual blemish rather than background skin, hair, or lighting glare.
4. **Clinical Decision Support & Ethical AI:**
   * Distinguishes routine acne care (OTC salicylic acid, gentle cleansing) from atypical lesions requiring formal dermatologist review (ABCDE criteria).
   * Prominent medical safety disclaimer for safe deployment.

---

## 📂 Project Structure

```
skin_disease_detector/
│
├── app.py              # Interactive Streamlit dashboard & web UI
├── model.py            # MobileNetV2 architecture, inference & Grad-CAM heatmaps
├── prepare_data.py     # Data validation, cleaning & 1:1 balancing pipeline
├── train.py            # PyTorch training script with data augmentation
├── requirements.txt    # Python dependencies for local & cloud deployment
├── .gitignore          # Excludes datasets, checkpoints, and cache
└── README.md           # Documentation, deployment guide & interview notes
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

## 🌐 Deploying to GitHub & Streamlit Community Cloud

### Step 1: Push Code to GitHub
1. Open PowerShell or Command Prompt in the project folder:
   ```bash
   cd C:\Users\shwet\.gemini\antigravity\scratch\skin_disease_detector
   ```
2. Initialize and commit the repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: DermAI Skin Disease Detection Dashboard"
   ```
3. Create a new repository on [GitHub](https://github.com/new) named `skin-disease-detector`.
4. Link and push your repository:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/skin-disease-detector.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud (Free Hosting)
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `YOUR_USERNAME/skin-disease-detector`.
4. Set the Main file path to: `app.py`.
5. Click **"Deploy!"**. Your app will be live with a shareable public URL in under 2 minutes.

---

## 🏋️ Training with Custom Datasets

1. Place your raw downloaded images into:
   * `raw_data/acne/`
   * `raw_data/other/`
2. Run the cleaning & 1:1 balancing script:
   ```bash
   python prepare_data.py
   ```
   *(This ensures both classes have the exact same number of verified, clean images).*
3. Run the model training pipeline:
   ```bash
   python train.py
   ```
   *(Saves the best-performing weights to `skin_model.pth`, which `app.py` will automatically detect).*

---

## 💬 Interview Guide: How to Explain This Project

* **"Why did you choose binary classification?"**
  > *"Rather than attempting a noisy 20-class model with limited data, I designed this as a primary triage tool to distinguish routine Acne Vulgaris from other atypical lesions that require dermatologist evaluation."*
* **"How did you prevent class imbalance?"**
  > *"I wrote an automated data pipeline that validated file integrity first, cleaned out corrupted or low-resolution images, and performed balanced under-sampling to guarantee a strict 50/50 class distribution across both training and validation sets."*
* **"Why MobileNetV2?"**
  > *"MobileNetV2 uses depthwise separable convolutions, which reduces model parameters by over 80% compared to standard CNNs. This allows the Streamlit app to run instantaneous inference on CPU without needing a GPU server."*
* **"How does the visual explainability work?"**
  > *"I extracted feature activation maps from the final convolutional block and overlaid them as a heatmap. This provides clinical interpretability by visually confirming that the model attends to the lesion texture and borders rather than background skin artifacts."*
