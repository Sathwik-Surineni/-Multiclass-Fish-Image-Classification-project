import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf

# -------------------- Streamlit Page Settings --------------------
st.set_page_config(page_title="Multiclass Fish Image Classification", layout="centered")
st.title("🐟 Multiclass Fish Image Classification Demo")

# -------------------- Paths --------------------
MODEL_PATH = "best_model_single.keras"
  # update if needed
CLASS_MAP_PATH = "class_indices.json"

# -------------------- Load Model & Class Names --------------------
@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()

    # Load model
    loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # If model has multiple inputs, fix it to use only the first
    if isinstance(loaded_model.input, list) and len(loaded_model.input) > 1:
        st.warning("⚠ Model has multiple inputs. Stripping to single input mode.")
        loaded_model = tf.keras.Model(inputs=loaded_model.input[0], outputs=loaded_model.output)

    # Load class names
    if not os.path.exists(CLASS_MAP_PATH):
        st.error(f"❌ Class map file not found: {CLASS_MAP_PATH}")
        st.stop()

    with open(CLASS_MAP_PATH, "r") as f:
        class_indices = json.load(f)
        class_names = list(class_indices.keys())

    return loaded_model, class_names


model, class_names = load_model_and_classes()

# -------------------- Image Preprocessing --------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------- File Uploader --------------------
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)  # updated to avoid warning

    st.write("Classifying... ⏳")
    try:
        input_tensor = preprocess_image(img)
        preds = model.predict(input_tensor)
        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        st.success(f"Prediction: **{pred_class}** ({confidence:.2f}% confidence)")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
