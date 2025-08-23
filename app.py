import os
import io
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from tensorflow import keras

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="CIFAR-10 Classifier", layout="wide")

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load CIFAR-10 labels
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load the trained model safely (no compile to avoid extra deps)
@st.cache_resource
def load_cifar10_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_cifar10_model()

# Streamlit UI
st.title("CIFAR-10 Image Classification")
st.write("Upload an image and let the model predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader(f"Prediction: **{predicted_class}**")
    st.bar_chart(prediction[0])


# -------------------------------
# Utilities
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    # compile=False prevents optimizer / loss deserialization issues
    model = keras.models.load_model(model_path, compile=False)
    return model

def preprocess_pil(img_pil: Image.Image, target_size=(32, 32)):
    """Convert to RGB, resize to 32x32, scale to [0,1]. Returns numpy array (1,32,32,3)."""
    img = img_pil.convert("RGB").resize(target_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_batch(model, batch_x, top_k=3):
    """batch_x: (N,32,32,3) float32 in [0,1]. Returns (probs, topk_idx, topk_prob)."""
    probs = model.predict(batch_x, verbose=0)
    topk_idx = np.argsort(-probs, axis=1)[:, :top_k]
    topk_prob = np.take_along_axis(probs, topk_idx, axis=1)
    return probs, topk_idx, topk_prob

@st.cache_data(show_spinner=True)
def load_cifar10():
    """Try loading CIFAR-10 test set. If blocked (no internet), return (None, None)."""
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_test = x_test.astype("float32") / 255.0
        y_test = y_test.squeeze().astype(int)
        return x_test, y_test
    except Exception as e:
        return None, None

def show_topk_chart(topk_idx, topk_prob):
    labels = [CLASS_NAMES[i] for i in topk_idx]
    df = pd.DataFrame({"class": labels, "confidence": topk_prob})
    df = df.set_index("class")
    st.bar_chart(df)

def show_card(title, value):
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:16px;border:1px solid #e6e6e6;">
          <div style="font-size:12px;color:#666;">{title}</div>
          <div style="font-size:22px;font-weight:700;margin-top:2px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# Load model once
# -------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Could not find '{MODEL_PATH}' in the app directory. "
             f"Place your trained model file there (or set MODEL_PATH env var).")
    st.stop()

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)
st.success("Model loaded ✔")

st.title("🖼️ CIFAR-10 Image Classifier")
st.caption("Upload your images or use sample CIFAR-10 test images. Then pick one to inspect.")

tab1, tab2 = st.tabs(["🗂️ Upload Images (multiple)", "🧪 CIFAR-10 Samples (optional)"])

# -------------------------------
# Tab 1: Upload multiple images
# -------------------------------
with tab1:
    uploaded_files = st.file_uploader(
        "Upload one or more images (JPG/PNG). We'll show them in a grid and you can choose one to evaluate.",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Show thumbnails with indices
        cols = st.columns(5)
        img_cache = []
        for idx, uf in enumerate(uploaded_files):
            try:
                img = Image.open(io.BytesIO(uf.read()))
                img_cache.append((idx, uf.name, img))
                with cols[idx % 5]:
                    st.image(img, caption=f"#{idx}: {uf.name}", use_column_width=True)
            except Exception as e:
                st.warning(f"Skipped file {uf.name}: {e}")

        if img_cache:
            sel = st.number_input(
                "Choose an image index to evaluate",
                min_value=0, max_value=len(img_cache)-1, value=0, step=1
            )
            _, sel_name, sel_img = img_cache[sel]

            st.subheader(f"Selected: #{sel} — {sel_name}")
            left, right = st.columns([1, 1])

            with left:
                st.image(sel_img, caption="Selected Image", use_column_width=True)

            # Predict
            arr = preprocess_pil(sel_img, (32, 32))
            probs, topk_idx, topk_prob = predict_batch(model, arr, top_k=3)
            pred = int(np.argmax(probs[0]))
            conf = float(probs[0, pred])

            with right:
                show_card("Predicted", CLASS_NAMES[pred])
                show_card("Confidence", f"{conf*100:.2f}%")
                st.markdown("**Top-3 probabilities**")
                show_topk_chart(topk_idx[0], topk_prob[0])
    else:
        st.info("Upload a few images to get started, or switch to the **CIFAR-10 Samples** tab.")

# -------------------------------
# Tab 2: CIFAR-10 samples
# -------------------------------
with tab2:
    x_test, y_test = load_cifar10()
    if x_test is None:
        st.warning(
            "Could not load CIFAR-10 test set (possibly due to no internet on the host). "
            "Use the **Upload Images** tab instead."
        )
    else:
        n = st.slider("How many random samples to load?", 5, 20, 10, 1)
        if st.button("Load random CIFAR-10 test images"):
            rng = np.random.default_rng(seed=None)
            idxs = rng.choice(len(x_test), size=n, replace=False)
            st.session_state["c10_idxs"] = idxs

        if "c10_idxs" in st.session_state:
            idxs = st.session_state["c10_idxs"]
            cols = st.columns(5)
            for j, i in enumerate(idxs):
                with cols[j % 5]:
                    st.image((x_test[i] * 255).astype(np.uint8),
                             caption=f"#{j} → true: {CLASS_NAMES[y_test[i]]}",
                             use_column_width=True)

            sel = st.number_input("Pick an index from the loaded set", 0, len(idxs)-1, 0, 1)
            true_idx = idxs[sel]
            img = (x_test[true_idx] * 255).astype(np.uint8)

            left, right = st.columns([1, 1])
            with left:
                st.image(img, caption=f"Selected #{sel}", use_column_width=True)

            # predict
            arr = np.expand_dims(x_test[true_idx], 0)  # already [0,1]
            probs, topk_idx, topk_prob = predict_batch(model, arr, top_k=3)
            pred = int(np.argmax(probs[0]))
            conf = float(probs[0, pred])

            with right:
                show_card("True label", CLASS_NAMES[y_test[true_idx]])
                show_card("Predicted", CLASS_NAMES[pred])
                show_card("Confidence", f"{conf*100:.2f}%")
                st.markdown("**Top-3 probabilities**")
                show_topk_chart(topk_idx[0], topk_prob[0])
        else:
            st.info("Click **Load random CIFAR-10 test images** to populate the grid.")
