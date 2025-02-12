import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError

# Define month names globally to ensure accessibility
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']

# Function to ensure scaler exists
def ensure_scaler(df):
    features = ['Total Outpatient Attendance', 'Total Healthcare Personnels', 'Total Healthcare Facilities']
    if not all(feature in df.columns for feature in features):
        return None, "Uploaded CSV file does not contain required features. Please ensure that the CSV file uploaded has 'Total Outpatient Attendance', 'Total Healthcare Personnels', and 'Total Healthcare Facilities'."
    
    data = df[features]
    scaler = MinMaxScaler()
    scaler.fit(data)
    joblib.dump(scaler, 'scaler.save')  # Save the scaler
    return scaler, None

def load_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_resource
def load_model(sequence_length):
    try:
        model_file = 'outpatient_prediction_model_1month.h5' if sequence_length == 1 else 'outpatient_prediction_model_12month.h5'
        model = tf.keras.models.load_model(model_file, custom_objects={'mse': MeanSquaredError()})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def prepare_input_sequence(input_data, scaler, sequence_length):
    if scaler is None or not hasattr(scaler, 'data_min_'):
        st.error("Scaler is not fitted. Please enter valid historical data or upload a CSV.")
        return None
    
    scaled_input = scaler.transform(input_data)
    input_sequence = scaled_input[-sequence_length:].reshape(1, sequence_length, scaled_input.shape[1])
    return input_sequence

def main():
    st.title('Outpatient Attendance Prediction')
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")
    df = None
    scaler = None
    error_message = None
    user_input = []
    
    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
        scaler, error_message = ensure_scaler(df)
        
        if error_message:
            st.error(error_message)
            df = None
            scaler = None
    
    sequence_length = st.radio("Select Sequence Length:", [1, 12], index=1)
    prediction_horizon = st.slider("Select Prediction Horizon (Months):", 1, 12, 1)
    model = load_model(sequence_length)
    
    if model is None:
        st.stop()
    
    if df is None or error_message:
        st.subheader("Enter Historical Data")
        current_month = datetime.datetime.now().month
        selected_months = [months[(current_month - 2 - i) % 12] for i in range(sequence_length)]
        
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
    
    if st.button('Predict Attendance'):
        try:
            input_sequence = prepare_input_sequence(np.array(historical_data), scaler, sequence_length)
            if input_sequence is None:
                st.stop()
            
            predictions = []
            future_months = []
            current_month = datetime.datetime.now().month  # February (2)
            
            # Start predictions from current month (February)
            for i in range(prediction_horizon):
                prediction = model.predict(input_sequence)
                dummy_input = np.zeros((1, 3))
                dummy_input[:, 0] = prediction[:, 0]
                predicted_attendance = round(scaler.inverse_transform(dummy_input)[0][0])
                predictions.append(predicted_attendance)
                
                input_sequence = np.roll(input_sequence, shift=-1, axis=1)
                input_sequence[0, -1, 0] = prediction[0, 0]
                
                # Calculate future month index starting from current month (no offset needed)
                future_month_idx = (current_month - 1 + i) % 12
                future_months.append(months[future_month_idx])
            
            results_df = pd.DataFrame({'Month': future_months, 'Predicted Attendance': predictions})
            
            # Create a custom month ordering starting from current month
            next_12_months = []
            for i in range(12):
                month_idx = (current_month - 1 + i) % 12
                next_12_months.append(months[month_idx])
            
            # Set the custom month ordering
            results_df['Month'] = pd.Categorical(results_df['Month'], 
                                               categories=next_12_months, 
                                               ordered=True)
            results_df = results_df.sort_values('Month')

            st.subheader('Prediction Results')
            st.write(f'Predicted Outpatient Attendance for the next {prediction_horizon} months:')
            
            # Use Streamlit's interactive line chart
            st.line_chart(results_df.set_index('Month'))
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
if __name__ == '__main__':
    main()