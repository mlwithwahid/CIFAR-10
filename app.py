import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("model.h5")
    return model

model = load_trained_model()

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and let the model predict its class!")

# File uploader
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image (CIFAR-10 expects 32x32)
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### ✅ Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")
