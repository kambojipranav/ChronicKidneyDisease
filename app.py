import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("stack_kidney_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="CKD Prediction (Full Features)", layout="centered")
st.title("🩺 Chronic Kidney Disease Prediction")
st.markdown("Enter the patient's medical details (24 features) to predict CKD using a stacked model.")

def user_input_features():
    age = st.number_input("Age", 1, 120)
    bp = st.number_input("Blood Pressure", 50, 200)
    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.slider("Albumin", 0, 5, 0)
    su = st.slider("Sugar", 0, 5, 0)
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
    ba = st.selectbox("Bacteria", ["present", "notpresent"])
    bgr = st.number_input("Blood Glucose Random", 50, 500)
    bu = st.number_input("Blood Urea", 1, 400)
    sc = st.number_input("Serum Creatinine", 0.1, 50.0)
    sod = st.number_input("Sodium", 100, 200)
    pot = st.number_input("Potassium", 2.5, 20.0)
    hemo = st.number_input("Hemoglobin", 3.0, 17.5)
    pcv = st.number_input("Packed Cell Volume", 15, 55)
    wc = st.number_input("White Blood Cell Count", 2000, 30000)
    rc = st.number_input("Red Blood Cell Count", 2.5, 6.5)
    htn = st.selectbox("Hypertension", ["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
    appet = st.selectbox("Appetite", ["good", "poor"])
    pe = st.selectbox("Pedal Edema", ["yes", "no"])
    ane = st.selectbox("Anemia", ["yes", "no"])

    data = [
        age, bp, sg, al, su,
        1 if rbc == "normal" else 0,
        1 if pc == "normal" else 0,
        1 if pcc == "present" else 0,
        1 if ba == "present" else 0,
        bgr, bu, sc, sod, pot,
        hemo, pcv, wc, rc,
        1 if htn == "yes" else 0,
        1 if dm == "yes" else 0,
        1 if cad == "yes" else 0,
        1 if appet == "good" else 0,
        1 if pe == "yes" else 0,
        1 if ane == "yes" else 0
    ]
    return np.array(data).reshape(1, -1)

input_data = user_input_features()

if st.button("Predict"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    result = "✅ The patient is likely to be **Healthy**." if prediction[0] == 0 else "⚠️ The patient is likely to have **Chronic Kidney Disease**."
    st.subheader("Prediction Result")
    st.success(result)
