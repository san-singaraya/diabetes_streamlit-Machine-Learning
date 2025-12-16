# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")
st.title("🖼️ CIFAR-10 Image Classification")
st.write("Upload an image and let the CNN predict its class")

# CIFAR-10 class names
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# -------------------------------
# Load & Train Model (Cached)
# -------------------------------
@st.cache_resource
def load_and_train_model():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Simple CNN
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=0
    )

    return model

model = load_and_train_model()

st.success("✅ Model trained successfully!")

# -------------------------------
# Image Upload
# -------------------------------
st.subheader("📤 Upload an Image (32x32 preferred)")

uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize to CIFAR-10 size
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("🔍 Prediction Result")
    st.success(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

# -------------------------------
# Footer
# -------------------------------
st.caption("📌 CNN | CIFAR-10 | TensorFlow | Streamlit")
