import streamlit as st
import pandas as pd
import joblib


# ===============================
# Branding Section
# ===============================

st.title("🧠 Chronic Kidney Disease (CKD) Prediction")
st.image("ckd_banner.jpeg", use_container_width=True)
st.markdown("""
Welcome to the **CKD Prediction App**.  
This tool uses a Machine Learning Model trained on clinical data to predict if a patient is likely to have Chronic Kidney Disease.

---

### 📋 Steps to Use the App:

1. **Enter patient data** in all fields below.
2. Click **Predict CKD Status**.
3. View the model's prediction (CKD or Not CKD) and risk confidence.

---
""")

# ===============================
# Load the model and scaler
# ===============================
model = joblib.load("ckd_rf_model.sav")
scaler = joblib.load("ckd_scaler.sav")

# ===============================
# Input Fields
# ===============================
st.header("📝 Enter Patient Clinical Values")

# Collect user input
age = st.number_input("Age", 1, 100)
blood_pressure = st.number_input("Blood Pressure", 60, 180)
specific_gravity = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
albumin = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
sugar = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
red_blood_cells = st.selectbox("Red Blood Cells", [0, 1])
pus_cell = st.selectbox("Pus Cell", [0, 1])
pus_cell_clumps = st.selectbox("Pus Cell Clumps", [0, 1])
bacteria = st.selectbox("Bacteria", [0, 1])
blood_glucose_random = st.number_input("Blood Glucose Random", 50, 500)
blood_urea = st.number_input("Blood Urea", 1, 300)
serum_creatinine = st.number_input("Serum Creatinine", 0.1, 15.0)
sodium = st.number_input("Sodium", 100, 200)
potassium = st.number_input("Potassium", 1.0, 15.0)
haemoglobin = st.number_input("Haemoglobin", 3.0, 17.0)
packed_cell_volume = st.number_input("Packed Cell Volume", 10, 60)
white_blood_cell_count = st.number_input("White Blood Cell Count", 3000, 25000)
red_blood_cell_count = st.number_input("Red Blood Cell Count", 2.5, 6.5)
hypertension = st.selectbox("Hypertension", [0, 1])
diabetes_mellitus = st.selectbox("Diabetes Mellitus", [0, 1])
coronary_artery_disease = st.selectbox("Coronary Artery Disease", [0, 1])
appetite = st.selectbox("Appetite", [0, 1])
peda_edema = st.selectbox("Pedal Edema", [0, 1])
aanemia = st.selectbox("Aanemia", [0, 1])

# Create DataFrame for prediction
# ===============================
# Prediction
# ===============================

input_df = pd.DataFrame([[
    age, blood_pressure, specific_gravity, albumin, sugar,
    red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
    blood_glucose_random, blood_urea, serum_creatinine, sodium,
    potassium, haemoglobin, packed_cell_volume, white_blood_cell_count,
    red_blood_cell_count, hypertension, diabetes_mellitus,
    coronary_artery_disease, appetite, peda_edema, aanemia
]], columns=[
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
    'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
    'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'
])


# Predict
if st.button("Predict CKD Status"):
    scaled_input = scaler.transform(input_df)
    result = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]  # probability of CKD
    
    if result == 1:
        st.error(f" Likely CKD (Chronic Kidney Disease) (Risk: {prob:.2%})")
    else:
        st.success(f"✅ Not CKD (No Chronic Kidney Disease) (Confidence: {1 - prob:.2%})")