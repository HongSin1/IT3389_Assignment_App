import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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

# Custom CSS for a cleaner navbar without the black shadow
st.markdown("""
    <style>
        .navbar {
            display: flex;
            justify-content: center;
            gap: 30px;
            background-color: white;
            padding: 15px 0;
            border-radius: 10px;
        }
        .navbar button {
            border: none;
            background: none;
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding: 12px 24px;
            border-radius: 30px;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.3s ease;
            text-align: center;
        }
        .navbar button:hover {
            background-color: #388E3C;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Create navigation bar with direct links to deployment
st.markdown('<div class="navbar">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<a href="https://maintrying.streamlit.app/" target="_blank"><button style="width: 100%; padding: 14px 24px;">🏠 Home</button></a>', unsafe_allow_html=True)

with col2:
    st.markdown('<a href="https://your-disease-prediction-deployment-link.com" target="_blank"><button style="width: 100%;">🩺 Disease Prediction</button></a>', unsafe_allow_html=True)

with col3:
    st.markdown('<a href="https://your-outpatient-prediction-deployment-link.com" target="_blank"><button style="width: 100%;">📅 Outpatient Prediction</button></a>', unsafe_allow_html=True)

with col4:
    st.markdown('<a href="https://your-bed-occupancy-deployment-link.com" target="_blank"><button style="width: 100%;">🛏 Bed Occupancy Prediction</button></a>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Default page content for Home Page
page = "📸 Image Classifier"  # Set "Image Classifier" as the default page

if page == "🏠 Home":
    redirect("https://maintrying.streamlit.app/")

elif page == "📸 Image Classifier":
    st.title("📸 Medicine Image Classifier")
    st.write("Upload an image to classify.")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Extract actual class from filename
        actual_class = uploaded_file.name.split("_")[0]

        # Preprocess the image
        image = image.resize((224, 224))
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)  # Get confidence score

        # Show results
        st.subheader("🔍 Classification Result")
        st.write(f"### **Actual Class :** {actual_class}")
        st.write(f"### **Predicted Class:** {class_labels[predicted_class]}")
        st.write(f"### **Confidence:** {confidence:.2f}")

elif page == "🩺 Disease Prediction":
    redirect("https://your-disease-prediction-link")

elif page == "📅 Outpatient Prediction":
    redirect("https://your-outpatient-prediction-link")

elif page == "🛏️ Bed Occupancy Prediction":
    redirect("https://your-bed-occupancy-link")


