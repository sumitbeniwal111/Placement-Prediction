import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("placement_model.pkl")

model = load_model()

# Streamlit UI
st.title("College Placement Prediction")
st.write("Predict whether a student is likely to be placed based on academic profile.")

# Inputs
st.header("Student Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=30, value=21)
    gender = st.radio("Gender", ("Male", "Female"))
    stream = st.selectbox("Stream", [
        "Electronics And Communication", "Computer Science",
        "Information Technology", "Mechanical", "Electrical", "Civil"
    ])

with col2:
    internships = st.number_input("Number of Internships", min_value=0, max_value=5, value=1)
    cgpa = st.selectbox("CGPA", [5, 6, 7, 8, 9])
    hostel = st.radio("Hostel", ("Yes", "No"))
    backlogs = st.radio("History of Backlogs", ("Yes", "No"))

# Encoding input
gender_encoded = 1 if gender == "Male" else 0
stream_encoded = {
    "Electronics And Communication": 1,
    "Computer Science": 2,
    "Information Technology": 3,
    "Mechanical": 4,
    "Electrical": 5,
    "Civil": 6
}[stream]
hostel_encoded = 1 if hostel == "Yes" else 0
backlogs_encoded = 1 if backlogs == "Yes" else 0

input_df = pd.DataFrame([[
    age, gender_encoded, stream_encoded, internships,
    cgpa, hostel_encoded, backlogs_encoded
]], columns=['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'Hostel', 'HistoryOfBacklogs'])

# Prediction
if st.button("Predict Placement"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("Prediction: This student is likely to get placed 🎉")
    else:
        st.warning("Prediction: This student may not get placed 😟")
