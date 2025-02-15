import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configure page layout to use full width
st.set_page_config(layout="wide")

# Load the model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("image_ft_mobilenetv2_new.h5")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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

# Define medicine information
medicine_info = {
    'Alaxan': {
        'description': "Alaxan is a powerful combination pain reliever containing paracetamol and ibuprofen. It effectively treats various types of pain including:",
        'uses': [
            "Headache and migraine",
            "Muscle aches and strains",
            "Joint pain and arthritis",
            "Dental pain",
            "Menstrual cramps"
        ],
        'note': "Should be taken after meals to avoid stomach irritation. Not recommended for children under 12 years old."
    },
    'Bactidol': {
        'description': "Bactidol is an antiseptic mouthwash containing Hexetidine, designed to fight oral infections and provide relief from:",
        'uses': [
            "Sore throat and throat infections",
            "Mouth ulcers and oral sores",
            "Gum inflammation",
            "Bad breath (halitosis)",
            "Post-dental procedure care"
        ],
        'note': "Gargle for 30 seconds twice daily. Avoid eating or drinking for 30 minutes after use."
    },
    'Bioflu': {
        'description': "Bioflu is a combination medication specifically formulated for comprehensive flu relief, containing paracetamol, phenylephrine, and chlorphenamine. It helps with:",
        'uses': [
            "Fever and body pain",
            "Nasal congestion and runny nose",
            "Headache associated with flu",
            "Sneezing and postnasal drip",
            "General flu symptoms"
        ],
        'note': "May cause drowsiness. Avoid alcohol and operating machinery while taking this medication."
    },
    'Biogesic': {
        'description': "Biogesic contains paracetamol (acetaminophen) and is one of the most widely used fever and pain medications. It's effective for:",
        'uses': [
            "Fever reduction",
            "Mild to moderate pain relief",
            "Headache and toothache",
            "General body aches",
            "Safe for most age groups"
        ],
        'note': "Safe to take on an empty stomach. Follow recommended dosage carefully to avoid liver damage."
    },
    'DayZinc': {
        'description': "DayZinc is a zinc supplement fortified with essential vitamins designed to boost immune system function. It's beneficial for:",
        'uses': [
            "Strengthening immune response",
            "Reducing cold duration",
            "Supporting wound healing",
            "Maintaining skin health",
            "Enhancing protein synthesis"
        ],
        'note': "Best absorbed when taken with food. May interact with certain antibiotics."
    },
    'Decolgen': {
        'description': "Decolgen is a multi-action cold medication containing phenylephrine, chlorphenamine, and paracetamol. It effectively treats:",
        'uses': [
            "Nasal congestion and sinus pressure",
            "Runny nose and sneezing",
            "Cold-related headache and fever",
            "Body aches associated with colds",
            "General cold symptoms"
        ],
        'note': "May cause drowsiness. Take with food to minimize stomach upset."
    },
    'Fish Oil': {
        'description': "Fish Oil supplements are rich in omega-3 fatty acids (EPA and DHA), providing numerous health benefits including:",
        'uses': [
            "Heart health support",
            "Brain function enhancement",
            "Joint inflammation reduction",
            "Eye health maintenance",
            "Mood regulation support"
        ],
        'note': "Take with meals to maximize absorption and minimize fishy aftertaste. Keep refrigerated for best quality."
    },
    'Kremil S': {
        'description': "Kremil S is an antacid medication containing aluminum hydroxide, magnesium hydroxide, and simethicone. It provides relief from:",
        'uses': [
            "Heartburn and acid reflux",
            "Indigestion and stomach acidity",
            "Gastric discomfort",
            "Bloating and gas",
            "Stomach pain due to hyperacidity"
        ],
        'note': "Take between meals or before bedtime. Avoid taking with other medications as it may affect their absorption."
    },
    'Medicol': {
        'description': "Medicol contains ibuprofen, a powerful non-steroidal anti-inflammatory drug (NSAID). It's effective for treating:",
        'uses': [
            "Moderate to severe pain",
            "Inflammatory conditions",
            "Sports injuries and sprains",
            "Arthritis pain",
            "Fever reduction"
        ],
        'note': "Always take with food to protect stomach lining. Not recommended for those with stomach ulcers."
    },
    'Neozep': {
        'description': "Neozep is a combination cold medication containing phenylpropanolamine, chlorphenamine, and paracetamol. It helps relieve:",
        'uses': [
            "Cold and flu symptoms",
            "Nasal and sinus congestion",
            "Fever and body aches",
            "Runny nose and sneezing",
            "Headache associated with colds"
        ],
        'note': "May cause drowsiness. Not recommended for those with high blood pressure."
    }
}

def display_medicine_info(medicine_name):
    """Display medicine information in a structured format"""
    try:
        info = medicine_info[medicine_name]
        st.markdown("### 📝 Description")
        st.write(info['description'])
        
        st.markdown("### 💊 Common Uses")
        for use in info['uses']:
            st.write(f"• {use}")
        
        st.markdown("### ⚠️ Important Note")
        st.write(info['note'])
    except Exception as e:
        st.error(f"Error displaying medicine information: {str(e)}")

def process_image(uploaded_file):
    """Process the uploaded image and return the preprocessed version"""
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image_for_model = image.resize((224, 224))
        image_array = np.array(image_for_model) / 255.0
        return image, np.expand_dims(image_array, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    st.title("📸 Medicine Image Classifier")
    st.write("Upload an image to classify.")
    
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please try again later.")
        return

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Process image
        image, processed_image = process_image(uploaded_file)
        if image is None or processed_image is None:
            return

        # Display image
        width, _ = image.size
        display_width = min(width, 900)
        st.image(image, caption="Uploaded Image", width=display_width)
        
        try:
            # Extract actual class from filename
            actual_class = uploaded_file.name.split("_")[0]
            
            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Show results
            st.markdown("## 🔍 Classification Result")
            st.markdown(f"### Actual Class: {actual_class}")
            st.markdown(f"### Predicted Class: {predicted_class}")
            st.markdown(f"### Confidence: {confidence:.2%}")
            
            # Display medicine information based on prediction accuracy
            st.markdown("---")
            
            if predicted_class == actual_class:
                st.success("✅ Correct Prediction!")
                if confidence < 0.75:
                    st.warning("‼️ **Low Confidence Prediction:** While the prediction is correct, the confidence is less than 75%. It's advisable to consult a healthcare professional or pharmacist to verify this medication.")
                display_medicine_info(predicted_class)
            else:
                st.error("❌ Incorrect Prediction")
                st.warning("‼️ **Important:** The model has misidentified this medication. Please consult a healthcare professional or pharmacist to verify any medication before use.")
                
                # Create columns only if we need to display both sets of information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### Actual Medicine: {actual_class}")
                    if actual_class in medicine_info:
                        display_medicine_info(actual_class)
                    else:
                        st.error(f"No information available for {actual_class}")
                
                with col2:
                    st.markdown(f"### Predicted Medicine: {predicted_class}")
                    if predicted_class in medicine_info:
                        display_medicine_info(predicted_class)
                    else:
                        st.error(f"No information available for {predicted_class}")
                
        except Exception as e:
            st.error(f"An error occurred during classification: {str(e)}")

if __name__ == "__main__":
    main()
