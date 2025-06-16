import streamlit as st
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Backend API URL
API_URL = "http://127.0.0.1:8000/predict"

class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Streamlit page config
st.set_page_config(page_title="Retinal Disease Classifier", layout="wide")
st.sidebar.title("🧭 Navigation")
st.sidebar.info("Upload a retina image to detect diabetic retinopathy stage.")

st.title("👁️ Retinal Disease Classifier")
st.write("This app uses a FastAPI backend with a deep learning model to classify retina images into 5 stages of diabetic retinopathy.")

# Preprocess image locally for Grad-CAM visualization
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0), np.array(image.resize((224, 224)))

# Main Upload Section
uploaded_file = st.file_uploader("📤 Upload a retina image (JPG/PNG)", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🔍 Input Image", use_container_width=True)

    with st.spinner("🔍 Sending image to backend model..."):
        response = requests.post(
            API_URL,
            files={"file": uploaded_file.getvalue()}
        )

    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']
        confidence = float(result['confidence'].replace('%', '')) / 100

        st.markdown(f"### 🩺 **Prediction**: `{prediction}`")
        st.markdown(f"### 🎯 **Confidence**: `{confidence * 100:.2f}%`")
        st.progress(confidence)
    else:
        st.error("❌ Failed to get prediction from backend.")

    # Optional local Grad-CAM (dummy heatmap since model is backend)
    st.markdown("### 🔥 Grad-CAM (Simulated Overlay)")
    _, img_display = preprocess_image(image)
    dummy_heatmap = np.random.uniform(0, 1, (224, 224))  # Replace with real Grad-CAM if needed

    heatmap_color = cv2.applyColorMap(np.uint8(255 * dummy_heatmap), cv2.COLORMAP_JET)
    overlay = heatmap_color * 0.4 + img_display
    overlay = np.uint8(overlay)

    st.image(overlay, caption="Simulated Grad-CAM", use_container_width=True)

# Confusion Matrix Section
st.markdown("---")
if st.button("📈 Show Example Confusion Matrix"):
    st.markdown("### 🧮 Confusion Matrix (Sampled Data)")
    y_true = [0, 1, 2, 3, 4, 2, 1, 0, 4, 3]
    y_pred = [0, 1, 2, 3, 4, 2, 2, 0, 3, 3]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍⚕️ Created for AI Hackathon")
st.sidebar.caption("Frontend: Streamlit | Backend: FastAPI + TensorFlow")
