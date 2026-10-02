# 🔬 Facial Skin Disease Detector

> End-to-end Computer Vision & Clinical Decision-Support System that classifies facial skin spots as either **Acne Vulgaris** or **Other Skin Lesion** using transfer learning (MobileNetV2), visual attention heatmaps (Grad-CAM), and actionable clinical triage guidelines.

---

## 📋 Features

- **Transfer Learning Backbone** — MobileNetV2 pre-trained on ImageNet with customized 2-class classification head
- **Zero Class Imbalance** — Automated data pipeline enforcing a strict 50.0% / 50.0% class balance
- **Visual Explainability (Grad-CAM)** — Feature activation maps visually highlighting the exact blemish detected
- **Dual Input Modes** — Upload image files (`.jpg`, `.png`) or use live webcam snapshot (`st.camera_input`)
- **Interactive Streamlit UI** — Multi-tab dashboard with interactive Plotly probability distributions
- **Clinical Triage Guide** — Differentiates routine acne care from ABCDE dermatologist referral guidelines

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare sample dataset

```bash
python download_dataset.py
```

### 3. Clean and balance dataset (50/50 ratio)

```bash
python prepare_data.py
```

### 4. Train model

```bash
python train.py
```

### 5. Launch Streamlit app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
facial-skin-disease-detector/
├── models/
│   └── skin_model.pth              # Fine-tuned PyTorch model weights
├── raw_data/
│   ├── acne/                       # Raw acne images
│   └── other/                      # Raw other skin lesion images
├── data/
│   ├── train/                      # Balanced training split (50/50)
│   └── val/                        # Balanced validation split (50/50)
├── app.py                          # Streamlit application
├── model.py                        # Model architecture, inference & Grad-CAM
├── prepare_data.py                 # Data validation, cleaning & 1:1 balancing
├── download_dataset.py             # Dataset downloader / sample generator
├── train.py                        # PyTorch training pipeline
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 📊 Dataset & Class Distribution

| Class | Description | Clinical Action | Split Ratio |
| :--- | :--- | :--- | :--- |
| **Acne Vulgaris** | Inflammatory papules, pustules, comedones | Routine skincare, OTC Salicylic Acid / Benzoyl Peroxide | 50.0% |
| **Other Skin Lesion** | Moles, nevi, dermatitis, rosacea, atypical spots | Monitor ABCDE criteria, consult dermatologist | 50.0% |

---

## 🤖 Technical Architecture & Design Decisions

* **Binary Triage Formulation:**
  Designed as a clinical triage decision-support tool to distinguish routine Acne Vulgaris from other atypical lesions requiring dermatologist evaluation.
* **Pre-training & Transfer Learning:**
  MobileNetV2 was chosen for its parameter efficiency (depthwise separable convolutions), allowing edge and CPU deployments with sub-second inference latency (~25ms).
* **Explainability (Grad-CAM):**
  Feature activation maps are extracted from the final convolutional block to provide visual interpretability, ensuring high transparency for clinical decision support.
* **Data Integrity:**
  Automated pre-filtering removes truncated, non-RGB, and extreme aspect-ratio images to maintain high data quality and a strict 1:1 class balance.

---

## 📦 Tech Stack

- **Python 3.10+**
- **PyTorch & torchvision** — Transfer learning & neural network inference
- **Streamlit** — Web UI & camera capture
- **Plotly** — Interactive probability distributions
- **Matplotlib & Pillow** — Grad-CAM heatmap overlays & image processing
- **NumPy** — Numerical tensor calculations
