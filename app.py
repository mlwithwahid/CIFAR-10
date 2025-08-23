import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# CIFAR-10 labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load model once & cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

st.title("🚀 CIFAR-10 Image Classifier")
st.write("Upload an image (32x32 CIFAR-10 style) and the model will predict its class.")

# Allow multiple uploads
uploaded_files = st.file_uploader(
    "Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    # Show all uploaded images
    for idx, uploaded_file in enumerate(uploaded_files):
        st.write(f"### Image {idx+1}")
        image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Ask which image to classify
    option = st.selectbox(
        "Select an image to classify:",
        options=[f"Image {i+1}" for i in range(len(uploaded_files))]
    )

    selected_index = int(option.split()[1]) - 1
    selected_image = Image.open(uploaded_files[selected_index]).convert("RGB").resize((32, 32))

    # Preprocess
    img_array = np.array(selected_image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    st.subheader(f"Prediction: **{predicted_class}**")
    st.bar_chart(prediction[0])
