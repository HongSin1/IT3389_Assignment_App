import streamlit as st

# Set the Streamlit app title (first Streamlit command)
st.set_page_config(page_title="Healthcare Prediction Models", page_icon="🩺", layout="wide")

# Sidebar Navigation
st.sidebar.title("Model Navigation")
selected_model = st.sidebar.radio(
    "Select a Model to Explore:",
    (
        "Home Page",
        "Background Information"
        "Disease Prediction System",
        "Medicine Image Classifier",
        "Outpatient Attendance Prediction",
        "Hospital Capacity Prediction"
    )
)

# Conditional imports for each model to avoid early Streamlit calls
if selected_model == "Home Page":
    st.title("Welcome to Our Healthcare Prediction Models App")
    st.write(
        """
        This application showcases 4 different healthcare prediction models:
        
        1. Disease Prediction System
        2. Medicine Image Classifier
        3. Outpatient Attendance Prediction
        4. Hospital Capacity Prediction
        
        Use the sidebar to navigate between the different models and to learn more about our research!
        """
    )
elif selected_model == "Background Information":
    import background_info
    background_info.main()
    
elif selected_model == "Disease Prediction System":
    import diseasepredapp  
    diseasepredapp.main()

elif selected_model == "Medicine Image Classifier":
    import med_imageClassification  
    med_imageClassification.main()

elif selected_model == "Outpatient Attendance Prediction":
    import outpatient_prediction_app  
    outpatient_prediction_app.main()

elif selected_model == "Hospital Capacity Prediction":
    import roanneapp  
    roanneapp.main()
