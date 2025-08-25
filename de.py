import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_detection_model():
    model_path = hf_hub_download(
        repo_id="Aya2012001/Brain_Tumor_Detection_Segmentation",
        filename="brain-tumor-detection-using-resnet50.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_for_detection(image: Image.Image):
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🧠 Brain Tumor Detection")
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    st.subheader("Detection Result")
    detection_model = load_detection_model()
    processed_det = preprocess_for_detection(image)
    prediction = detection_model.predict(processed_det)[0]

    if prediction[0] > 0.5:
        st.error("⚠️ Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")
