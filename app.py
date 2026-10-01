"""
DermAI: Facial Skin Lesion & Acne Detection Dashboard
-----------------------------------------------------
Interactive Streamlit application providing:
1. File upload and live webcam snapshot
2. MobileNetV2 transfer learning prediction
3. Grad-CAM visual attention heatmap overlay
4. Interactive Plotly probability distribution
5. Clinical care guidelines & medical safety disclaimers
"""

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import os

from model import get_model, predict_skin_image, CLASSES

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DermAI: Facial Skin Screener",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Custom CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        color: #92400E;
        font-size: 0.9rem;
    }
    .result-card-acne {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #93C5FD;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1rem;
    }
    .result-card-other {
        background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
        border: 1px solid #FDBA74;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
    }
    .badge-acne { background-color: #2563EB; color: white; }
    .badge-other { background-color: #EA580C; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="main-header">🔬 DermAI: Facial Skin Lesion Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Computer Vision screening assistant distinguishing <b>Acne Vulgaris</b> from <b>Other Skin Lesions</b>.</div>', unsafe_allow_html=True)

# --- Medical Advisory Banner ---
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ Clinical Advisory & Educational Disclaimer:</b>
    This AI tool is designed for educational screening and decision-support only. It does <b>NOT</b> replace professional clinical dermoscopy or biopsy by a licensed physician.
</div>
""", unsafe_allow_html=True)

# --- Cache Model Instance ---
@st.cache_resource(show_spinner="Loading neural network...")
def load_cached_model():
    return get_model()

model = load_cached_model()

# --- Sidebar Controls ---
st.sidebar.header("📁 Input Source")
input_option = st.sidebar.radio(
    "Choose how to provide an image:",
    ["Upload Image File", "Use Live Camera", "Load Demo Sample"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Model Architecture")
st.sidebar.markdown("""
* **Backbone:** MobileNetV2 (Transfer Learning)
* **Classes:** 
  1. `Acne`
  2. `Other Skin Lesion`
* **Balance:** Strict 1:1 ratio
* **Explainability:** Feature Activation Mapping (CAM)
""")

# --- Input Handling ---
selected_image = None

if input_option == "Upload Image File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a close-up photo of the skin lesion (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)

elif input_option == "Use Live Camera":
    camera_file = st.camera_input("Take a close-up photo of the skin area")
    if camera_file is not None:
        selected_image = Image.open(camera_file)

elif input_option == "Load Demo Sample":
    demo_choice = st.sidebar.selectbox(
        "Select a simulated sample to test:",
        ["Sample 1: Acne", "Sample 2: Other Skin Lesion"]
    )
    img_size = (300, 300)
    if "Acne" in demo_choice:
        base = Image.new("RGB", img_size, (235, 195, 170))
        img_np = np.array(base, dtype=np.uint8)
        y, x = np.ogrid[:300, :300]
        mask1 = (x - 150)**2 + (y - 140)**2 < 25**2
        mask2 = (x - 180)**2 + (y - 190)**2 < 18**2
        img_np[mask1] = [190, 60, 60]
        img_np[mask2] = [200, 75, 75]
        selected_image = Image.fromarray(img_np)
    else:
        base = Image.new("RGB", img_size, (240, 200, 175))
        img_np = np.array(base, dtype=np.uint8)
        y, x = np.ogrid[:300, :300]
        mask = ((x - 150)/35)**2 + ((y - 150)/25)**2 < 1.0
        img_np[mask] = [90, 50, 35]
        selected_image = Image.fromarray(img_np)

# --- Tabs Layout ---
tab_screen, tab_metrics, tab_guide = st.tabs([
    "🔬 Screening & Heatmap",
    "📊 Model Performance & Architecture",
    "💡 Clinical Guidelines & ABCDE Criteria"
])

# --- Tab 1: Screening & Heatmap ---
with tab_screen:
    if selected_image is not None:
        selected_image = ImageOps.exif_transpose(selected_image)

        col_img, col_pred = st.columns([1, 1.2], gap="large")

        with col_img:
            st.subheader("📷 Analyzed Image")
            st.image(selected_image, use_container_width=True, caption="Input Skin Region")

        with col_pred:
            st.subheader("📊 Diagnostic Assessment")
            
            with st.spinner("Evaluating dermatological features..."):
                result = predict_skin_image(selected_image, model=model)

            pred_class = result["predicted_class"]
            confidence = result["confidence"]
            probs = result["probabilities"]
            recs = result["recommendations"]

            card_class = "result-card-acne" if pred_class == "Acne" else "result-card-other"
            badge_class = "badge-acne" if pred_class == "Acne" else "badge-other"

            st.markdown(f"""
            <div class="{card_class}">
                <span class="badge {badge_class}">{pred_class.upper()}</span>
                <h2 style="margin: 0.5rem 0 0.2rem 0; color: #111827;">{recs['title']}</h2>
                <p style="font-size: 1.15rem; font-weight: 600; color: #374151;">
                    Confidence: <span style="color: #1E3A8A;">{confidence:.1f}%</span>
                </p>
                <p style="margin: 0; color: #4B5563; font-size: 0.95rem;">{recs['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Interactive Plotly Bar Chart
            st.markdown("#### Probability Distribution")
            chart_df = {
                "Condition": ["Acne", "Other Skin Lesion"],
                "Probability": [probs["Acne"], probs["Other Skin Lesion"]],
                "Color": ["#2563EB", "#EA580C"]
            }
            fig_bar = px.bar(
                chart_df,
                x="Probability",
                y="Condition",
                orientation='h',
                text=[f"{p:.1f}%" for p in chart_df["Probability"]],
                color="Condition",
                color_discrete_map={"Acne": "#2563EB", "Other Skin Lesion": "#EA580C"}
            )
            fig_bar.update_layout(
                showlegend=False,
                xaxis=dict(range=[0, 100], title="Probability (%)"),
                yaxis=dict(title=""),
                height=220,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap / Explainability Section
        st.markdown("---")
        st.subheader("🔍 Visual Attention & Heatmap (Grad-CAM Saliency)")
        st.markdown("The attention map highlights the specific spatial regions the neural network focused on to make its prediction.")

        col_heat1, col_heat2 = st.columns(2)

        with col_heat1:
            st.image(selected_image, caption="Original Input", use_container_width=True)

        with col_heat2:
            if result["activation_map"] is not None:
                cam = result["activation_map"]
                orig_w, orig_h = selected_image.size
                
                # Pillow safe resampling
                resample_method = getattr(Image, 'Resampling', Image).BILINEAR
                cam_img = Image.fromarray(np.uint8(255 * cam)).resize((orig_w, orig_h), resample=resample_method)
                cam_np = np.array(cam_img) / 255.0

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(selected_image)
                ax.imshow(cam_np, cmap='jet', alpha=0.45)
                ax.axis('off')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                plt.close(fig)

                st.image(buf, caption="AI Attention Heatmap Overlay (Red = High Attention)", use_container_width=True)
            else:
                st.info("Activation map not available for this input.")

        # Clinical Care Steps
        st.markdown("---")
        st.subheader("💡 Recommended Next Steps & Clinical Care")
        
        col_rec1, col_rec2 = st.columns([2, 1])
        with col_rec1:
            st.markdown(f"**Urgency Level:** `{recs['urgency']}`")
            for i, step in enumerate(recs["care_steps"], 1):
                st.markdown(f"**{i}.** {step}")
                
        with col_rec2:
            st.info("""
            **When to see a Doctor Urgently:**
            * Lesion bleeds, oozes, or crusts over
            * Rapid growth or change in shape/color
            * Irregular, notched, or scalloped borders
            * Persistent pain or itching
            """)

    else:
        st.info("👈 Please select an option from the sidebar to upload a photo, use your webcam, or test with a sample.")

# --- Tab 2: Model Performance & Architecture ---
with tab_metrics:
    st.subheader("📐 Model Architecture & Technical Specifications")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        #### Network Details:
        * **Backbone:** MobileNetV2 (Pre-trained on ImageNet-1K)
        * **Classifier Head:** Dropout (0.2) + Fully Connected Linear Layer (1280 $\\to$ 2)
        * **Input Resolution:** $224 \\times 224$ pixels (RGB)
        * **Inference Speed:** $\\sim 25\\text{ ms}$ on standard CPU
        * **Class Distribution:** Strict 50.0% / 50.0% (Zero Class Imbalance)
        """)
        
    with col_m2:
        st.markdown("""
        #### Training Hyperparameters:
        * **Optimizer:** Adam (Learning Rate = $1\\times 10^{-4}$)
        * **Loss Function:** Cross-Entropy Loss
        * **Data Augmentation:** Random Horizontal Flips, Rotations ($\\pm 15^\\circ$), Color Jitter
        * **Validation Strategy:** Stratified 80/20 Train-Val Split
        """)

# --- Tab 3: Clinical Guidelines & ABCDE Criteria ---
with tab_guide:
    st.subheader("📋 Clinical Dermatology Reference Guide")
    st.markdown("""
    When evaluating skin spots that are not typical acne, dermatologists use the **ABCDE criteria** to screen for atypical lesions or melanoma:
    
    | Criterion | Description | Warning Sign |
    | :--- | :--- | :--- |
    | **A — Asymmetry** | One half of the spot does not match the other half. | Asymmetrical shape |
    | **B — Border** | The edges are irregular, ragged, notched, or blurred. | Uneven borders |
    | **C — Color** | The color is not uniform across the spot. | Multiple shades of brown, black, red, or blue |
    | **D — Diameter** | The spot is larger than 6 millimeters (pencil eraser size). | Growth $> 6\\text{ mm}$ |
    | **E — Evolving** | The spot is changing in size, shape, color, or symptoms. | Itching, bleeding, elevation |
    """)
