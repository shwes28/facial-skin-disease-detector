"""
Model Architecture & Inference Pipeline
----------------------------------------
- Architecture: MobileNetV2 with Transfer Learning
- Target Classes: 2 classes (Acne vs. Other Skin Lesion)
- Explainability: Feature Activation Mapping (CAM) for visual attention heatmaps
- Clinical Decision Support: Tailored care advice & urgency classification
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

CLASSES = ["Acne", "Other Skin Lesion"]

# Standard ImageNet normalization for transfer learning
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_model(weights_path: str = "skin_model.pth"):
    """
    Loads MobileNetV2 with a customized 2-class classification head.
    If fine-tuned weights exist at weights_path, loads them;
    otherwise uses the pre-trained ImageNet backbone for immediate testing.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Replace final classification layer for 2 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"[INFO] Loaded custom trained weights from {weights_path}")
        except Exception as e:
            print(f"[WARNING] Could not load weights: {e}")
    else:
        print("[INFO] Initialized with MobileNetV2 transfer learning backbone.")
        
    model.eval()
    return model

def generate_activation_map(model, input_tensor):
    """
    Extracts the spatial feature activation map from the last convolutional layer
    to show where the network focused its visual attention.
    """
    try:
        features = model.features(input_tensor)  # Shape: [1, 1280, 7, 7]
        cam = torch.mean(features, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if np.max(cam) != np.min(cam):
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)
        return cam
    except Exception:
        return None

def predict_skin_image(image: Image.Image, model=None, weights_path: str = "skin_model.pth"):
    """
    Processes a PIL image and returns predictions, probabilities, and advice.
    """
    if model is None:
        model = get_model(weights_path)
        
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    input_tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
    pred_idx = int(np.argmax(probabilities))
    predicted_class = CLASSES[pred_idx]
    confidence = float(probabilities[pred_idx]) * 100.0
    
    activation_map = generate_activation_map(model, input_tensor)
    
    if predicted_class == "Acne":
        recommendations = {
            "title": "Acne Vulgaris Detected",
            "summary": "Visual features are consistent with localized inflammatory or non-inflammatory acne lesions (e.g., papules, pustules, or comedones).",
            "care_steps": [
                "Cleanse gently twice daily with a mild, non-comedogenic cleanser.",
                "Consider over-the-counter topical treatments with Salicylic Acid (BHA) or Benzoyl Peroxide.",
                "Avoid picking, squeezing, or popping lesions to prevent scarring and infection.",
                "Consult a dermatologist if blemishes are cystic, painful, or unresponsive to OTC products."
            ],
            "urgency": "Low to Moderate (Routine Skincare / Consultation)"
        }
    else:
        recommendations = {
            "title": "Other Skin Lesion Detected",
            "summary": "Features are not typical for common acne. The image indicates another dermatological lesion (e.g., mole/nevus, dermatitis, rosacea, or atypical lesion).",
            "care_steps": [
                "Schedule an in-person evaluation with a board-certified dermatologist for clinical dermoscopy.",
                "Check for the ABCDE criteria: Asymmetry, Border irregularity, Color variation, Diameter (>6mm), and Evolution over time.",
                "Avoid harsh acne spot treatments which may irritate non-acne skin conditions.",
                "Seek prompt medical review if the spot bleeds, grows rapidly, or has irregular borders."
            ],
            "urgency": "Recommended Dermatologist Evaluation"
        }
        
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "Acne": float(probabilities[0]) * 100.0,
            "Other Skin Lesion": float(probabilities[1]) * 100.0
        },
        "recommendations": recommendations,
        "activation_map": activation_map
    }
