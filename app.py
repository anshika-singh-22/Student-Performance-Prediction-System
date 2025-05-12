import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("student_performance_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Set up page
st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Prediction System")
st.markdown("<hr style='border:1px solid #f63366'>", unsafe_allow_html=True)

st.markdown(
    "<h4 style='color:#2E8B57;'>Enter Student Details Below:</h4>",
    unsafe_allow_html=True
)

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox("School", ['GP', 'MS'])
        sex = st.radio("Sex", ['F', 'M'])
        age = st.slider("Age", 15, 22, 17)
        address = st.radio("Address", ['U', 'R'])
        famsize = st.selectbox("Family Size", ['LE3', 'GT3'])
        Pstatus = st.selectbox("Parent Status", ['T', 'A'])
        Medu = st.slider("Mother's Education", 0, 4, 2)
        Fedu = st.slider("Father's Education", 0, 4, 2)
        Mjob = st.selectbox("Mother's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
        Fjob = st.selectbox("Father's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
        reason = st.selectbox("Reason to Choose School", ['home', 'reputation', 'course', 'other'])

    with col2:
        guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
        traveltime = st.slider("Travel Time", 1, 4, 1)
        studytime = st.slider("Study Time", 1, 4, 2)
        failures = st.slider("Past Class Failures", 0, 4, 0)
        schoolsup = st.selectbox("School Support", ['yes', 'no'])
        famsup = st.selectbox("Family Support", ['yes', 'no'])
        paid = st.selectbox("Paid Classes", ['yes', 'no'])
        activities = st.selectbox("Extra-curricular Activities", ['yes', 'no'])
        nursery = st.selectbox("Attended Nursery", ['yes', 'no'])
        higher = st.selectbox("Wants Higher Education", ['yes', 'no'])
        internet = st.selectbox("Internet Access", ['yes', 'no'])
        romantic = st.selectbox("In a Romantic Relationship", ['yes', 'no'])

    col3, col4 = st.columns(2)
    with col3:
        famrel = st.slider("Family Relationship Quality", 1, 5, 4)
        freetime = st.slider("Free Time After School", 1, 5, 3)
        goout = st.slider("Going Out Frequency", 1, 5, 3)
        Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 2)

    with col4:
        health = st.slider("Health Status", 1, 5, 5)
        absences = st.slider("Number of Absences", 0, 93, 2)
        G1 = st.slider("First Period Grade (G1)", 0, 20, 15)
        G2 = st.slider("Second Period Grade (G2)", 0, 20, 16)

    submit = st.form_submit_button("🎯 Predict Final Grade (G3)")

# Prediction
if submit:
    input_data = {
        'school': school, 'sex': sex, 'age': age, 'address': address, 'famsize': famsize,
        'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu, 'Mjob': Mjob, 'Fjob': Fjob,
        'reason': reason, 'guardian': guardian, 'traveltime': traveltime, 'studytime': studytime,
        'failures': failures, 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet,
        'romantic': romantic, 'famrel': famrel, 'freetime': freetime, 'goout': goout,
        'Dalc': Dalc, 'Walc': Walc, 'health': health, 'absences': absences, 'G1': G1, 'G2': G2
    }

    # Feature engineering
    input_data["G1_G2_avg"] = (input_data["G1"] + input_data["G2"]) / 2
    input_data["total_parent_edu"] = input_data["Medu"] + input_data["Fedu"]
    input_data["alc_ratio"] = input_data["Walc"] / (input_data["Dalc"] + 1e-6)

    input_df = pd.DataFrame([input_data])
    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)

    st.markdown("<hr style='border:1px solid #f63366'>", unsafe_allow_html=True)
    st.success(f"🎉 Predicted Final Grade (G3): **{prediction[0]:.2f}** / 20")
