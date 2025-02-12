import streamlit as st
import gdown
import diseasepredapp  
import med_imageClassification  
import outpatient_prediction_app  
import roanneapp  

# Set the Streamlit app title
st.set_page_config(page_title="Healthcare Prediction Models", page_icon="🩺", layout="wide")

# Sidebar Navigation
st.sidebar.title("Model Navigation")
selected_model = st.sidebar.radio(
    "Select a Model to Explore:",
    (
        "Home Page",
        "Disease Prediction App",
        "Medical Image Classification",
        "Outpatient Attendance Prediction",
        "Roanne's Prediction App"
    )
)

# Home Page
if selected_model == "Home Page":
    st.title("Welcome to Our Healthcare Prediction Models App")
    st.write(
        """
        This application showcases 4 different healthcare prediction models:
        
        1. Disease Prediction App
        2. Medical Image Classification
        3. Outpatient Attendance Prediction
        4. Roanne's Prediction App
        
        Use the sidebar to navigate between the different models.
        """
    )

# Link to each model's app
elif selected_model == "Disease Prediction App":
    st.title("Disease Prediction Model")
    diseasepredapp.main()

elif selected_model == "Medical Image Classification":
    st.title("Medical Image Classification Model")
    med_imageClassification.main()

elif selected_model == "Outpatient Attendance Prediction":
    st.title("Outpatient Attendance Prediction")
    outpatient_prediction_app.main()

elif selected_model == "Roanne's Prediction App":
    st.title("Roanne's Prediction Model")
    roanneapp.main()
