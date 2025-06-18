import streamlit as st
import joblib
import numpy as np

# Load model (already contains scaler inside pipeline)
model = joblib.load("stack_kidney_model.pkl")

st.title("Chronic Kidney Disease Predictor")

st.markdown("Enter the patient's medical details:")

# Define all inputs in the correct order
age = st.number_input("Age", 1, 120)
bp = st.number_input("Blood Pressure")
sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
rbc = st.selectbox("Red Blood Cells", ['normal', 'abnormal'])
pc = st.selectbox("Pus Cell", ['normal', 'abnormal'])
pcc = st.selectbox("Pus Cell Clumps", ['notpresent', 'present'])
ba = st.selectbox("Bacteria", ['notpresent', 'present'])
bgr = st.number_input("Blood Glucose Random")
bu = st.number_input("Blood Urea")
sc = st.number_input("Serum Creatinine")
sod = st.number_input("Sodium")
pot = st.number_input("Potassium")
hemo = st.number_input("Hemoglobin")
pcv = st.number_input("Packed Cell Volume")
wc = st.number_input("White Blood Cell Count")
rc = st.number_input("Red Blood Cell Count")
htn = st.selectbox("Hypertension", ['no', 'yes'])
dm = st.selectbox("Diabetes Mellitus", ['no', 'yes'])
cad = st.selectbox("Coronary Artery Disease", ['no', 'yes'])
appet = st.selectbox("Appetite", ['good', 'poor'])
pe = st.selectbox("Pedal Edema", ['no', 'yes'])
ane = st.selectbox("Anemia", ['no', 'yes'])

def encode(val, mapping):
    return mapping[val]

# Prepare input vector
input_data = [
    age, bp, sg, al, su,
    encode(rbc, {'normal': 0, 'abnormal': 1}),
    encode(pc, {'normal': 0, 'abnormal': 1}),
    encode(pcc, {'notpresent': 0, 'present': 1}),
    encode(ba, {'notpresent': 0, 'present': 1}),
    bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
    encode(htn, {'no': 0, 'yes': 1}),
    encode(dm, {'no': 0, 'yes': 1}),
    encode(cad, {'no': 0, 'yes': 1}),
    encode(appet, {'good': 0, 'poor': 1}),
    encode(pe, {'no': 0, 'yes': 1}),
    encode(ane, {'no': 0, 'yes': 1}),
]

if st.button("Predict"):
    prediction = model.predict([input_data])
    if prediction[0] == 1:
        st.error("⚠️ Kidney Disease Detected")
    else:
        st.success("✅ No Kidney Disease")
