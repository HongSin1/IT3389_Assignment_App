import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("finetune_mobilenetv2_new.h5")

model = load_model()

# Define class labels
class_labels = {
    0: 'Alaxan',
    1: 'Bactidol',
    2: 'Bioflu',
    3: 'Biogesic',
    4: 'DayZinc',
    5: 'Decolgen',
    6: 'Fish Oil',
    7: 'Kremil S',
    8: 'Medicol',
    9: 'Neozep'
}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Image Classifier"])

# Homepage
if page == "Homepage":
    st.title("Welcome to the Medicine Image Classifier")
    st.write("Navigate using the sidebar to classify images.")

# Image Classifier
elif page == "Image Classifier":
    st.title("Medicine Image Classifier")
    st.write("Upload an image to classify.")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract actual class from filename (assuming format: 'ClassName_#.jpg')
        actual_class = uploaded_file.name.split("_")[0]  # Gets the first part before "_"

        # Preprocess the image
        image = image.resize((224, 224))
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)  # Get confidence score

        # Show results
        st.subheader("Image Classification Prediction Result")
        st.write(f"### **Actual Class (from filename):** {actual_class}")
        st.write(f"### **Predicted Class:** {class_labels[predicted_class]}")
        st.write(f"### **Confidence:** {confidence:.2f}")

        # Check if prediction is correct
        if actual_class.lower() == class_labels[predicted_class].lower():
            st.success("✅ Prediction is correct!")
        else:
            st.error("❌ Prediction is incorrect.")
