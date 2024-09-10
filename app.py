import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Add a custom background color or image using CSS
page_bg_img = '''
<style>
body {
    background-color: #f4f4f4;
    font-family: Arial, Helvetica, sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    text-align: center;
    border-radius: 12px;
    border: none;
}
.stNumberInput>div>input {
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #ccc;
}
.stNumberInput>div>label {
    font-weight: bold;
    color: #333;
}
</style>
'''

# Inject the CSS into the page
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the machine learning model
model_filename = 'diabetes.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)
    
# Title of the app
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)

# Organizing the input fields in two columns
col1, col2 = st.columns(2)

# Input fields
with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=140)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100)

with col2:
    insulin = st.number_input('Insulin', min_value=0, max_value=900)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, format="%.1f")
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, format="%.3f")
    age = st.number_input('Age', min_value=0, max_value=120)

# Prediction button with style
if st.button('Predict'):
    # Collect the inputs in a numpy array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                            bmi, diabetes_pedigree_function, age]])
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Display the prediction with formatted output
    if prediction[0] == 1:
        st.markdown("<h3 style='color: red;'>Prediction: Diabetic</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>Prediction: Not Diabetic</h3>", unsafe_allow_html=True)

# Adding a simple footer
st.markdown("<hr><footer style='text-align: center;'>Developed by Your Name</footer>", unsafe_allow_html=True)
