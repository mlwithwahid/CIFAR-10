import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = load_model("model.h5")

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.title("CIFAR-10 Image Classification 🚀")
st.write("Upload an image or test with sample CIFAR-10 dataset")

# --- Upload Image Section ---
uploaded_file = st.file_uploader("Upload an image (32x32)", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    img_array = np.array(image) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_input)
    predicted_label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.image(image, caption=f"Prediction: {predicted_label} ({confidence:.2f}%)")
    st.success(f"Predicted Class: {predicted_label} with {confidence:.2f}% confidence")

# --- Test with CIFAR-10 Dataset ---
st.subheader("🔍 Test with CIFAR-10 Sample Images")

if st.button("Load 10 Random CIFAR-10 Images"):
    # Load CIFAR-10 test data
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0

    # Pick 10 random indices
    indices = np.random.choice(len(x_test), 10, replace=False)
    st.session_state.sample_images = x_test[indices]
    st.session_state.sample_labels = y_test[indices]

if "sample_images" in st.session_state:
    # Display images in a grid
    st.write("Here are 10 random test images from CIFAR-10:")
    cols = st.columns(5)
    for i, img in enumerate(st.session_state.sample_images):
        with cols[i % 5]:
            st.image(img, caption=f"Index {i}", use_container_width=True)

    # Pick one image to test
    idx = st.slider("Select image index to check (0-9)", 0, 9, 0)
    img = st.session_state.sample_images[idx]
    true_label = class_names[st.session_state.sample_labels[idx][0]]

    # Predict
    img_input = np.expand_dims(img, axis=0)
    prediction = model.predict(img_input)
    predicted_label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show results
    st.image(img, caption=f"True: {true_label} | Pred: {predicted_label} ({confidence:.2f}%)")
    if true_label == predicted_label:
        st.success("✅ Correct Prediction!")
    else:
        st.error("❌ Wrong Prediction!")
