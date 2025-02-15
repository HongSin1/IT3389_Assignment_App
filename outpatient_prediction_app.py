# Import necessary libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import datetime
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Define month names to use throughout the application
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

# Load LLM model for personnel recommendation
@st.cache_resource
def load_llm_model():
    try:
        # Add HUGGINGFACE_TOKEN from secrets if available
        hf_token = st.secrets["HUGGINGFACE_TOKEN"] if "HUGGINGFACE_TOKEN" in st.secrets else None

        # Initialise BLOOMZ model with token
        model_name = "bigscience/bloomz-560m"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            model_max_length=512, 
            truncation=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            low_cpu_mem_usage=True,  
            device_map='auto' 
        )
        generator = pipeline('text-generation',
                           model=model,
                           tokenizer=tokenizer,
                           device_map='auto')
        return generator
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Generate personnel recommendations based on outpatient attendance
def generate_personnel_recommendation(attendance, month):
    generator = load_llm_model()
    if generator is None:
        return None
    
    # Calculate average daily attendance
    daily_attendance = int(attendance / 30)
    
    # Define prompt template for the model and apply few shot prompting
    prompt = f"""Task: Given the daily outpatient attendance, recommend the number of healthcare personnel needed.

    Example cases:
    1. Daily attendance: 15000
    Staff per day needed: 300
    Each staff handles 50 patients
    Recommended: 300

    2. Daily attendance: 30000
    Staff per day needed: 545
    Each staff handles 55 patients
    Recommended: 545

    3. Daily attendance: 45000
    Staff per day needed: 750
    Each staff handles 60 patients
    Recommended: 750

    Current case:
    Month: {month}
    Daily attendance: {daily_attendance}
    Staff needed (predict): """

    try:
        # Generate recommendation
        response = generator(prompt,
                           do_sample=True,
                           truncation=True,
                           max_length=len(prompt) + 15,
                           num_return_sequences=1,
                           temperature=0.7,
                           top_p=0.95,
                           top_k=100,
                           repetition_penalty=1.2)
        
        # Extract and format the numerical response
        full_text = response[0]['generated_text']
        number_part = full_text.split("Staff needed (predict):")[-1].strip()
        
        # Extract numbers and add variation
        import re
        numbers = re.findall(r'\d+', number_part)
        if numbers:
            base_number = int(numbers[0])
            variation = base_number * 0.05
            varied_number = int(base_number + np.random.uniform(-variation, variation))
            return "{:,}".format(varied_number)
        return number_part
        
    except Exception as e:
        st.error(f"Error generating recommendation: {e}")
        return None

# Validate and scale input data
def ensure_scaler(df):
    features = ['Total Outpatient Attendance', 'Total Healthcare Personnels', 'Total Healthcare Facilities']
    if not all(feature in df.columns for feature in features):
        return None, "Uploaded CSV file does not contain required features. Please ensure that the CSV file uploaded has 'Total Outpatient Attendance', 'Total Healthcare Personnels', and 'Total Healthcare Facilities' columns."
    
    data = df[features]
    scaler = MinMaxScaler()
    scaler.fit(data)
    joblib.dump(scaler, 'scaler.save')
    return scaler, None

# Load and read CSV files
def load_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Load the 2 prediction models
@st.cache_resource
def load_model(sequence_length):
    try:
        model_file = 'outpatient_prediction_model_1month.h5' if sequence_length == 1 else 'outpatient_prediction_model_12month.h5'
        model = tf.keras.models.load_model(model_file, custom_objects={'mse': MeanSquaredError()})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prepare input data for prediction
def prepare_input_sequence(input_data, scaler, sequence_length):
    if scaler is None or not hasattr(scaler, 'data_min_'):
        st.error("Scaler is not fitted. Please enter valid historical data or upload a CSV.")
        return None
    
    scaled_input = scaler.transform(input_data)
    input_sequence = scaled_input[-sequence_length:].reshape(1, sequence_length, scaled_input.shape[1])
    return input_sequence

# Create line chart for predictions
def create_prediction_plot(results_df):
    min_attendance = results_df[['Predicted Attendance', 'Actual Attendance']].min().min() if 'Actual Attendance' in results_df.columns else results_df['Predicted Attendance'].min()
    max_attendance = results_df[['Predicted Attendance', 'Actual Attendance']].max().max() if 'Actual Attendance' in results_df.columns else results_df['Predicted Attendance'].max()
    y_padding = (max_attendance - min_attendance) * 0.1
    
    fig = px.line(results_df,
                x='Month',
                y=['Predicted Attendance', 'Actual Attendance'] if 'Actual Attendance' in results_df.columns else ['Predicted Attendance'],
                markers=True)
    
    fig.update_layout(
        yaxis=dict(
            range=[min_attendance - y_padding, max_attendance + y_padding],
        ),
        hovermode='x',
        width=800,
        height=500
    )
    
    fig.update_traces(
        hovertemplate='<b>Month</b>: %{x}<br>' +
                    '<b>Attendance</b>: %{y:,.0f}<br>'
    )
    
    return fig

# Reset application state
def reset_session_state():
    st.session_state.predictions_made = False
    st.session_state.results_df = None
    st.session_state.prediction_plot = None
    st.session_state.show_recommendation_button = True
    st.session_state.generating_recommendations = False

# Main application function
def main():
    # Initialise session state variables
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'prediction_plot' not in st.session_state:
        st.session_state.prediction_plot = None
    if 'show_recommendation_button' not in st.session_state:
        st.session_state.show_recommendation_button = True
    if 'generating_recommendations' not in st.session_state:
        st.session_state.generating_recommendations = False
    if 'last_sequence_length' not in st.session_state:
        st.session_state.last_sequence_length = None
    if 'last_prediction_horizon' not in st.session_state:
        st.session_state.last_prediction_horizon = None
    if 'last_file_id' not in st.session_state:
        st.session_state.last_file_id = None

    # Set page layout and title
    st.title('😷 Outpatient Attendance Prediction')
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your CSV file (Test Data)", type=["csv"], key="file_uploader")
    actual_file = st.file_uploader("Upload your CSV file (Actual Data)", type=["csv"], key="actual_file")
    
    # Reset state if file changes
    current_file_id = f"{uploaded_file.name if uploaded_file else ''}{actual_file.name if actual_file else ''}"
    if current_file_id != st.session_state.last_file_id:
        reset_session_state()
        st.session_state.last_file_id = current_file_id
    
    df = None
    actual_df = None
    scaler = None
    error_message = None
    user_input = []
    
    # Process uploaded files
    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
        scaler, error_message = ensure_scaler(df)
        
        if error_message:
            st.error(error_message)
            df = None
            scaler = None
    
    if actual_file is not None:
        actual_df = load_uploaded_file(actual_file)
    
    # Model selection
    sequence_length = st.radio("Select Sequence Length:", [1, 12], index=1)
    prediction_horizon = st.slider("Select Prediction Horizon (Months):", 1, 12, 1)
    
    # Reset state if model selection change
    if (sequence_length != st.session_state.last_sequence_length or
        prediction_horizon != st.session_state.last_prediction_horizon):
        reset_session_state()
        st.session_state.last_sequence_length = sequence_length
        st.session_state.last_prediction_horizon = prediction_horizon
    
    model = load_model(sequence_length)
    
    if model is None:
        st.stop()
    
    # Manual data entry if no file uploaded
    if df is None or error_message:
        st.subheader("Enter Historical Data")
        
        selected_months = ['December', 'November', 'October', 'September', 'August', 'July',
                        'June', 'May', 'April', 'March', 'February', 'January']
        
        if sequence_length == 1:
            selected_months = selected_months[:1]
        
        user_data = pd.DataFrame(
            np.zeros((sequence_length, 3)),
            columns=["Attendance", "Healthcare Personnel", "Healthcare Facilities"]
        )
        user_data.insert(0, "Month", selected_months)
        
        user_data = st.data_editor(user_data, num_rows="fixed")
        
        user_input = user_data.iloc[:, 1:].values.tolist()

    historical_data = df[['Total Outpatient Attendance', 'Total Healthcare Personnels', 'Total Healthcare Facilities']].values[-sequence_length:] if df is not None else user_input
    
    if df is None and user_input:
        scaler = MinMaxScaler()
        scaler.fit(np.array(user_input))
    
    # Prediction logic
    if st.button('Predict Attendance'):
        try:
            input_sequence = prepare_input_sequence(np.array(historical_data), scaler, sequence_length)
            if input_sequence is None:
                st.stop()
            
            predictions = []
            future_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                           'September', 'October', 'November', 'December']
            
            for i in range(prediction_horizon):
                prediction = model.predict(input_sequence)
                dummy_input = np.zeros((1, 3))
                dummy_input[:, 0] = prediction[:, 0]
                predicted_attendance = round(scaler.inverse_transform(dummy_input)[0][0])
                predictions.append(predicted_attendance)
                
                input_sequence = np.roll(input_sequence, shift=-1, axis=1)
                input_sequence[0, -1, 0] = prediction[0, 0]
            
            results_df = pd.DataFrame({'Month': future_months[:prediction_horizon], 'Predicted Attendance': predictions})
            
            if actual_df is not None:
                actual_data = actual_df[['Total Outpatient Attendance']].values[:prediction_horizon]
                results_df['Actual Attendance'] = actual_data.flatten()
                results_df['Difference'] = results_df['Actual Attendance'] - results_df['Predicted Attendance']

            st.session_state.results_df = results_df
            st.session_state.predictions_made = True
            st.session_state.show_recommendation_button = True
            st.session_state.generating_recommendations = False
            
            st.session_state.prediction_plot = create_prediction_plot(results_df)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    # Display prediction results
    if st.session_state.predictions_made and st.session_state.results_df is not None:
        st.subheader('📈 Prediction Results')
        st.write(f'Predicted Outpatient Attendance for the next {prediction_horizon} months:')
        
        if st.session_state.prediction_plot is not None:
            st.plotly_chart(st.session_state.prediction_plot, use_container_width=True)
        
        st.write("Detailed Predictions:")
        st.dataframe(st.session_state.results_df.style.format({
            'Predicted Attendance': '{:,.0f}',
            'Actual Attendance': '{:,.0f}' if 'Actual Attendance' in st.session_state.results_df.columns else 'N/A',
            'Difference': '{:,.0f}' if 'Difference' in st.session_state.results_df.columns else 'N/A'
        }))
    
        # Generate recommendations section
        if st.session_state.show_recommendation_button:
            generate_recommendations = st.button('Generate Healthcare Personnel Recommendations', key='generate_recommendations')
            
            if generate_recommendations:
                st.session_state.show_recommendation_button = False
                st.session_state.generating_recommendations = True
                st.rerun()
        
        if not st.session_state.show_recommendation_button:
            st.subheader('👨🏻‍⚕️ Healthcare Personnel Recommendations')
            
            recommendations = []
            
            if st.session_state.generating_recommendations:
                with st.spinner('Generating recommendations...'):
                    for idx, row in st.session_state.results_df.iterrows():
                        month = row['Month']
                        attendance = row['Predicted Attendance']
                        recommended_personnel = generate_personnel_recommendation(attendance, month)
                        recommendations.append('N/A' if recommended_personnel is None else recommended_personnel)
                
                # Display formatted recommendations
                recommendation_df = st.session_state.results_df.copy()
                recommendation_df['Recommended Personnel'] = recommendations
                
                st.write("Monthly Healthcare Personnel Recommendations:")
                formatted_df = recommendation_df[['Month', 'Predicted Attendance', 'Recommended Personnel']].copy()
                format_dict = {
                    'Predicted Attendance': '{:,.0f}',
                    'Recommended Personnel': lambda x: x if isinstance(x, str) else 'N/A'
                }
                
                st.dataframe(formatted_df.style.format(format_dict))
                st.session_state.generating_recommendations = False

if __name__ == '__main__':
    main()
