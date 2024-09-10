import streamlit as st
import numpy as np
import pickle

# Load the machine learning model
model_filename = 'diabetes.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
    }
    .main {
        max-width: 700px;
        margin: auto;
        padding: 2rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .input-container {
        margin-bottom: 1rem;
    }
    .input-container label {
        display: block;
        margin-bottom: 0.5rem;
    }
    .input-container input {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .button {
        display: block;
        width: 100%;
        padding: 0.75rem;
        background-color: #28a745;
        border: none;
        border-radius: 4px;
        color: white;
        font-size: 1rem;
        cursor: pointer;
        text-align: center;
    }
    .button:hover {
        background-color: #218838;
    }
    .result {
        margin-top: 1rem;
        font-size: 1.2rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title("Diabetes Prediction")

# Container for input fields
with st.container():
    st.write("<div class='main'>", unsafe_allow_html=True)
    
    # Input fields for user
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1, key='pregnancies')
    glucose = st.number_input('Glucose', min_value=0, max_value=200, key='glucose')
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=140, key='blood_pressure')
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, key='skin_thickness')
    insulin = st.number_input('Insulin', min_value=0, max_value=900, key='insulin')
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, format="%.1f", key='bmi')
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, format="%.3f", key='diabetes_pedigree_function')
    age = st.number_input('Age', min_value=0, max_value=120, key='age')

    # Prediction button
    if st.button('Predict', key='predict'):
        # Collect the inputs in a numpy array
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                                bmi, diabetes_pedigree_function, age]])
        
        # Make predictions
        prediction = model.predict(input_data)
        
        # Display the prediction
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
        st.markdown(f"<div class='result'>Predicted Outcome: {result}</div>", unsafe_allow_html=True)
    
    st.write("</div>", unsafe_allow_html=True)
