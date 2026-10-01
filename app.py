"""
DermAI: Facial Skin Lesion & Acne Detection Dashboard
-----------------------------------------------------
Modern Dark Mode UI with interactive Before/After Image Comparison Slider
"""

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import os
import base64

from model import get_model, predict_skin_image, CLASSES

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DermAI: Facial Skin Screener",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Dark Mode Custom CSS Styling ---
st.markdown("""
<style>
    /* Dark Theme Core Styles */
    .stApp {
        background-color: #0B0F17;
        color: #F8FAFC;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    
    .disclaimer-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #F59E0B;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: #FCD34D;
        font-size: 0.9rem;
        backdrop-filter: blur(8px);
    }
    
    /* Result Cards */
    .result-card-acne {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.25) 0%, rgba(30, 58, 138, 0.3) 100%);
        border: 1px solid #06B6D4;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.15);
    }
    
    .result-card-other {
        background: linear-gradient(135deg, rgba(194, 65, 12, 0.25) 0%, rgba(154, 52, 18, 0.3) 100%);
        border: 1px solid #F97316;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(249, 115, 22, 0.15);
    }
    
    .badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 700;
        border-radius: 9999px;
        letter-spacing: 0.05em;
    }
    .badge-acne { background-color: #06B6D4; color: #082F49; }
    .badge-other { background-color: #F97316; color: #431407; }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="main-header">🔬 DermAI: Facial Skin Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Computer Vision screening assistant distinguishing <b>Acne</b> from <b>Other Skin Lesion</b>.</div>', unsafe_allow_html=True)

# --- Medical Advisory Banner ---
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ Clinical Advisory:</b>
    This AI tool is designed for educational screening and decision-support only. It does <b>NOT</b> replace professional clinical dermoscopy or biopsy by a licensed physician.
</div>
""", unsafe_allow_html=True)

# --- Cache Model Instance ---
@st.cache_resource(show_spinner="Loading neural network...")
def load_cached_model():
    return get_model()

model = load_cached_model()

# --- Helper: Convert PIL Image to Base64 ---
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Helper: Interactive Before/After Split Slider Component ---
def render_before_after_slider(img_before, img_after):
    b64_before = pil_to_base64(img_before)
    b64_after = pil_to_base64(img_after)
    
    slider_html = f"""
    <div style="max-width: 520px; margin: 0 auto; font-family: sans-serif;">
        <div style="position: relative; width: 100%; height: 380px; overflow: hidden; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
            <!-- After Image (Heatmap) -->
            <img src="data:image/png;base64,{b64_after}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;">
            <div style="position: absolute; top: 12px; right: 12px; background: rgba(0,0,0,0.65); color: #38BDF8; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">HEATMAP</div>
            
            <!-- Before Image (Original) with Clip -->
            <div id="clip-container" style="position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden; border-right: 3px solid #38BDF8;">
                <img src="data:image/png;base64,{b64_before}" style="position: absolute; top: 0; left: 0; width: 520px; height: 380px; max-width: none; object-fit: cover;">
                <div style="position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.65); color: #F8FAFC; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">ORIGINAL</div>
            </div>
            
            <!-- Range Slider Input -->
            <input type="range" min="0" max="100" value="50" id="slider-range" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: ew-resize; z-index: 10;" oninput="updateSlider(this.value)">
        </div>
        <p style="text-align: center; color: #94A3B8; font-size: 13px; margin-top: 8px;">
            ↔ Drag horizontally across the image to compare <b>Original</b> vs. <b>AI Heatmap</b>
        </p>
    </div>
    
    <script>
    function updateSlider(val) {{
        document.getElementById('clip-container').style.width = val + '%';
    }}
    </script>
    """
    st.components.v1.html(slider_html, height=430)

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

        # Run inference first
        with st.spinner("Evaluating dermatological features..."):
            result = predict_skin_image(selected_image, model=model)

        pred_class = result["predicted_class"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        recs = result["recommendations"]

        card_class = "result-card-acne" if pred_class == "Acne" else "result-card-other"
        badge_class = "badge-acne" if pred_class == "Acne" else "badge-other"

        # Generate Heatmap Overlay PIL Image
        orig_w, orig_h = selected_image.size
        resample_method = getattr(Image, 'Resampling', Image).BILINEAR
        
        if result["activation_map"] is not None:
            cam = result["activation_map"]
            cam_img = Image.fromarray(np.uint8(255 * cam)).resize((orig_w, orig_h), resample=resample_method)
            cam_np = np.array(cam_img) / 255.0

            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor('#0B0F17')
            ax.set_facecolor('#0B0F17')
            ax.imshow(selected_image)
            ax.imshow(cam_np, cmap='jet', alpha=0.48)
            ax.axis('off')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
            buf.seek(0)
            heatmap_overlay = Image.open(buf)
            plt.close(fig)
        else:
            heatmap_overlay = selected_image

        # Left Column: Interactive Before & After Slider
        with col_img:
            st.subheader("🖼️ Interactive Comparison")
            render_before_after_slider(selected_image, heatmap_overlay)

        # Right Column: Diagnostic Result Card & Plotly Chart
        with col_pred:
            st.subheader("📊 Diagnostic Assessment")

            st.markdown(f"""
            <div class="{card_class}">
                <span class="badge {badge_class}">{pred_class.upper()}</span>
                <h2 style="margin: 0.6rem 0 0.2rem 0; color: #F8FAFC;">{recs['title']}</h2>
                <p style="font-size: 1.25rem; font-weight: 700; color: #38BDF8;">
                    Confidence: {confidence:.1f}%
                </p>
                <p style="margin: 0; color: #CBD5E1; font-size: 0.95rem;">{recs['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Modern Dark Mode Plotly Bar Chart
            st.markdown("#### Probability Distribution")
            chart_df = {
                "Condition": ["Acne", "Other Skin Lesion"],
                "Probability": [probs["Acne"], probs["Other Skin Lesion"]],
            }
            fig_bar = px.bar(
                chart_df,
                x="Probability",
                y="Condition",
                orientation='h',
                text=[f"{p:.1f}%" for p in chart_df["Probability"]],
                color="Condition",
                color_discrete_map={"Acne": "#06B6D4", "Other Skin Lesion": "#F97316"}
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(range=[0, 100], title="Probability (%)", gridcolor="#1E293B"),
                yaxis=dict(title="", tickfont=dict(color="#F8FAFC", size=13)),
                height=200,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_bar.update_traces(textposition='outside', textfont=dict(color="#F8FAFC", size=13, weight="bold"))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Bottom: Clinical Care Steps
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
