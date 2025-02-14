import streamlit as st

# Set the Streamlit app title (first Streamlit command)
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

# Conditional imports for each model to avoid early Streamlit calls
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

elif selected_model == "Disease Prediction App":
    import diseasepredapp  
    diseasepredapp.main()

elif selected_model == "Medical Image Classification":
    import med_imageClassification  
    med_imageClassification.main()

elif selected_model == "Outpatient Attendance Prediction":
    import outpatient_prediction_app  
    outpatient_prediction_app.main()

elif selected_model == "Roanne's Prediction App":
    import roanneapp  
    roanneapp.main()
